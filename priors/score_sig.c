#include <math.h>
#include <stdlib.h>

#include "../sigvisa.h"

/*
  returns log( exp(logval1) + exp(logval2) )
*/
static inline double log_add_exp(double logval1, double logval2)
{
  if (logval1 >= logval2)
    return logval1 + log(1 + exp(logval2 - logval1));
  else
    return logval2 + log(1 + exp(logval1 - logval2));
}


double score_world_sig(SigModel_t * p_sigmodel, 
			   int numevents, 
			   Event_t * p_events,
			   PyArrayObject * arrtimes,
			   int verbose) {

  int numsignals = p_sigmodel->numsignals;
  Signal_t * p_signals = p_sigmodel->p_signals;

  double numsc, locsc=0, magsc=0, atimesc, sigsc;

  if (verbose)
  {
    printf ("%d events: \n", numevents);
    for (int i=0; i < numevents; ++i) {
      Event_t * e = p_events + i;
      printf (" event %d lon %lf lat %lf depth %lf time %lf mag %lf\n", i, e->evlon, e->evlat, e->evdepth, e->evtime, e->evmag);
    }
  }
  
  /* p(events, ttimes | signals) \propto p(signals | events, ttimes)p(ttimes|events)p(events)  */

  /* p(events): prior probabilities of events */

  numsc = NumEventPrior_LogTimeRate(&p_sigmodel->num_event_prior);

  for (int i=0; i<numevents; i++) {
    Event_t * p_event = p_events + i;
    locsc += EventLocationPrior_LogProb(&p_sigmodel->event_location_prior,
					p_event->evlon, p_event->evlat,
					p_event->evdepth);
    magsc += EventMagPrior_LogProb(&p_sigmodel->event_mag_prior,
				   p_event->evmag, 0);
  }
  /* p(ttimes | events): prior probabilities of travel times, given
     event (and station) locations */
  
  atimesc = ArrivalTimeJointPrior_LogProb(&p_sigmodel->arr_time_joint_prior, p_sigmodel->p_earth, numevents, p_events, arrtimes);
  
  /* p(signals| events, ttimes): probability of signals given events and travel times. */
  sigsc = SignalPrior_LogProb(&p_sigmodel->sig_prior, numsignals, p_signals, p_sigmodel->p_earth, numevents, p_events, arrtimes);
  


  double score = numsc + locsc + magsc + atimesc + sigsc;
  fprintf(stdout, "score is %lf + %lf + %lf + %lf + %lf = %lf\n", numsc, locsc, magsc, atimesc, sigsc, score);
  return score;
}




/* returns 0 if the event - phase is impossible to be detected at the
 * site, 1 otherwise 
 * if detected returns log probability of detection given event 
 *                     - log probability noise 
 * else returns log probability of mis-detection given event */
static int score_event_site_phase_int_sig(SigModel_t * p_sigmodel,
					  const Event_t * p_event,
					  const World_t * p_world,
					  int siteid, int phaseid,
					  double distance, double pred_az,
					  double * p_arrtimesc,
					  double * p_arrazsc,
					  double * p_ampsc,
					  double * p_sigsc,)
{
  EarthModel_t * p_earth;
  double pred_arrtime;
  double ttime;
  int numtimedefphases;

  p_earth = p_sigmodel->p_earth;
  
  assert(EarthModel_IsTimeDefPhase(p_earth, phaseid));
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                        p_event->evlat, p_event->evdepth,
                                        p_event->evtime, phaseid,
                                        siteid);
  /* check if the site is in the shadow zone for the event - phase */
  if (pred_arrtime < 0)
    return 0;

  /* check if the site is up */
  if (!SigModel_IsSiteUp(p_sigmodel, siteid, pred_arrtime))
    return 0;
  
  Arrival_t * arr = p_event->p_arrivals[siteid*numtimedefphases + phaseid];
  if (p_arrival == NULL) {
    printf("error, no arrival info for event!\n");
    return 0;
  }

  double pred_slow;


  // TODO: should be calling joint prior, conditional on other current events
  *p_arrtimesc += ArrivalTimePrior_LogProb(&p_sigmodel->arr_time_prior,
                                             arr->time, pred_arrtime,
					   0, siteid,
					   phaseid);


  *p_arrazsc += ArrivalAzimuthPrior_LogProb(&p_sigmodel->arr_az_prior,
					    arr->azi, 
					    pred_az,
					    0,
					    siteid, phaseid);

  ttime = EarthModel_ArrivalTime(p_earth, p_event->evlon, p_event->evlat,
				 p_event->evdepth, 0, phaseid, siteid);
  *p_ampsc += ArrivalAmplitudePrior_LogProb(&p_sigmodel->arr_amp_prior,
					    p_event->evmag, p_event->evdepth,
					    ttime, siteid, phaseid,
					    det->amp_det);
  if (isnan(*p_ampsc))
    {
      printf("nan arr-amp mb %.2lg, dep %.2lg ttime %.2lg siteid %d phaseid %d"
             " amp %.2lg", p_event->evmag, p_event->evdepth,
             ttime, siteid, phaseid, det->amp_det);
      exit(1);
    }
 

  *p_sigsc += SignalPrior_LogProb()

  return 1;
}

/* returns 0 if the event can't be detected at the site */
int score_event_site_phase(SigModel_t * p_sigmodel,
                           const Event_t * p_event,
                           int siteid, int phaseid,
                           double distance, double pred_az,
                           double * p_score)
{
  int      poss;
  int      detcnt;
  double   detsc;
  double   dettimesc;
  double   detslosc;
  double   detazsc;
  double   detphasesc;
  double   detsnrsc;
  double   detampsc;

  detcnt = 0;
  detsc = dettimesc = detslosc = detazsc = detphasesc = detsnrsc = detampsc =0;
  
  poss = score_event_site_phase_int(p_sigmodel, p_event, siteid, phaseid,
                                    distance, pred_az, &detcnt, &detsc,
                                    &dettimesc, &detslosc, &detazsc,
                                    &detphasesc, &detsnrsc, &detampsc);

  *p_score = detsc + dettimesc + detslosc + detazsc + detphasesc + detsnrsc
    + detampsc;

  return poss;
}

/* Scores an event phase, at a site, without requiring the caller to
   specify the distance or azimuth. */
int score_event_site_phase_simple(SigModel_t * p_sigmodel,
                                  const Event_t * p_event, int siteid,
                                  int phaseid, double * p_score)
{
  EarthModel_t * p_earth;
  double distance, pred_az;
  int poss;

  p_earth = p_sigmodel->p_earth;

  distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                              siteid);

  pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                      p_event->evlat, siteid);
  
  poss = score_event_site_phase(p_sigmodel, p_event, siteid, phaseid,
                                distance, pred_az, p_score);

  return poss;
}

/*
static void score_event_int(SigModel_t * p_sigmodel, Event_t * p_event, 
                            double * p_numsc, double * p_locsc, 
                            double * p_magsc,
                            double * p_detsc, double * p_dettimesc,
                            double * p_detazsc, double * p_detslosc,
                            double * p_detphasesc, double * p_detsnrsc,
                            double * p_detampsc,
                            int * p_poss_detcnt, int * p_detcnt)
{
  EarthModel_t * p_earth;
  Detection_t * p_detections;
  int numtimedefphases;
  int numsites;
  int siteid;
  int phaseid;

  p_earth = p_sigmodel->p_earth;
  p_detections = p_sigmodel->p_detections;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  *p_numsc += NumEventPrior_LogTimeRate(&p_sigmodel->num_event_prior);
  
  *p_locsc += EventLocationPrior_LogProb(&p_sigmodel->event_location_prior,
                                         p_event->evlon, p_event->evlat,
                                         p_event->evdepth);
  
  *p_magsc += EventMagPrior_LogProb(&p_sigmodel->event_mag_prior,
                                    p_event->evmag, 0);

  for (siteid=0; siteid<numsites; siteid++)
  {
    double distance, pred_az;
    
    distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                siteid);
    pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                        p_event->evlat, siteid);
    
    for (phaseid=0; phaseid<numtimedefphases; phaseid++)
    {
      *p_poss_detcnt += score_event_site_phase_int(p_sigmodel, p_event,
                                                   siteid, phaseid,
                                                   distance, pred_az,
                                                   p_detcnt,
                                                   p_detsc, p_dettimesc,
                                                   p_detslosc, p_detazsc,
                                                   p_detphasesc,
                                                   p_detsnrsc, p_detampsc);   
    }
  }
}*/

 /*
double score_event(SigModel_t * p_sigmodel, Event_t * p_event)
{
  double numsc, locsc, magsc, detsc, dettimesc, detazsc, detslosc, detphasesc;
  double detsnrsc, detampsc;
  double score;
  int possdetcnt, detcnt;

  numsc = locsc = magsc = detsc = dettimesc = detazsc = detslosc = 
    detphasesc = detsnrsc = detampsc = 0;
  possdetcnt = detcnt = 0;
  
  score_event_int(p_sigmodel, p_event, &numsc, &locsc, &magsc, &detsc, 
                  &dettimesc, &detazsc, &detslosc, &detphasesc, &detsnrsc,
                  &detampsc, &possdetcnt, &detcnt);
  
  score = numsc + locsc + magsc + detsc + dettimesc + detazsc + detslosc
    + detphasesc + detsnrsc + detampsc;

  return score;
  }*/


double prob_event(SigModel_t * p_sigmodel, Event_t * p_event)
{
  Event_t curr_event;
  double logprob;
  double step_z;

  /* make a copy of the event to work with */
  curr_event = *p_event;

  #define STEP_LON 1
  #define STEP_TIME 5
  #define STEP_DEPTH 350
  #define STEP_MAG 2

  step_z = (2.0 / (180.0 / STEP_LON));

  logprob = -INFINITY;
  
  for (curr_event.evlon = -180; curr_event.evlon < 180;
       curr_event.evlon += STEP_LON)
  {
    double z;
    
    for (z = -1; z < 1; z += step_z)
    {
      curr_event.evlat = RAD2DEG * asin(z);
      
      if (simple_distance_deg(curr_event.evlon, curr_event.evlat,
                              p_event->evlon, p_event->evlat) > DELTA_DIST)
        continue;
      
      for (curr_event.evdepth = MIN_DEPTH; curr_event.evdepth < MAX_DEPTH;
           curr_event.evdepth += STEP_DEPTH)
      {
        for (curr_event.evtime = p_event->evtime - DELTA_TIME;
             curr_event.evtime < (p_event->evtime + DELTA_TIME);
             curr_event.evtime += STEP_TIME)
        {
          for (curr_event.evmag = MIN_MAGNITUDE;
               curr_event.evmag < MAX_MAGNITUDE;
               curr_event.evmag += STEP_MAG)
          {
            double score;
            
            score = score_event(p_sigmodel, &curr_event);
            
            logprob = log_add_exp(logprob, score);
          }
        }
      }
    }
  }

  logprob += log(STEP_LON) + log(step_z) + log (STEP_DEPTH) + log(STEP_TIME)\
    + log(STEP_MAG);
  
  return logprob;
}

int compare_double(const double * pnum1, const double * pnum2)
{
  if (*pnum1 < *pnum2)
    return -1;
  else if (*pnum1 > *pnum2)
    return 1;
  else
    return 0;
}

double logprob_false(SigModel_t * p_sigmodel, int numdets, int * p_detids,
                     int verbose)
{
  int i;
 
  double sum_logprob = 0;

  double * p_sum_site_amp;
  
  int * p_cnt_site;

  double worst_amp=0;
  
  /* array of sum logprob amp per site, initialized to zero */
  p_sum_site_amp = (double *)calloc(p_sigmodel->numsites,
                                    sizeof(*p_sum_site_amp));

  /* keep track of the count for each site */
  p_cnt_site = (int *) calloc(p_sigmodel->numsites,
                              sizeof(*p_cnt_site));
  
  
  for (i=0; i<numdets; i++)
  {
    double logprob;
    
    int detid = p_detids[i];
    
    Detection_t * p_det = p_sigmodel->p_detections + detid;

    int siteid = p_det->site_det;
    
    logprob = FalseArrivalAmplitudePrior_LogProb(&p_sigmodel->arr_amp_prior,
                                                 siteid,
                                                 p_det->amp_det);

    p_sum_site_amp[siteid] += logprob;
    p_cnt_site[siteid] ++;

    sum_logprob += logprob;

    if (logprob < worst_amp)
      worst_amp = logprob;
  }

  if (verbose)
  {
    /*
    for (i=0; i<p_sigmodel->numsites; i++)
      if (p_cnt_site[i] > 100)
        printf("Site[%d], cnt %d, Avg amplitude logprob: %lf\n",
               i, p_cnt_site[i], p_sum_site_amp[i] / p_cnt_site[i]);
    */
    printf("Avg. noise amp logprob: %lf   worst %lf\n", sum_logprob / numdets,
           worst_amp);
  }
  
  return sum_logprob;
}
