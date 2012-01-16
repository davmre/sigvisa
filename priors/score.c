#include <math.h>
#include <stdlib.h>

#include "../netvisa.h"

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

/* returns 0 if the event - phase is impossible to be detected at the
 * site, 1 otherwise 
 * if detected returns log probability of detection given event 
 *                     - log probability noise 
 * else returns log probability of mis-detection given event */
static int score_event_site_phase_int(NetModel_t * p_netmodel,
                                      const Event_t * p_event,
                                      int siteid, int phaseid,
                                      double distance, double pred_az,
                                      int * p_detcnt,
                                      double * p_detsc, 
                                      double * p_dettimesc,
                                      double * p_detslosc,
                                      double * p_detazsc,
                                      double * p_detphasesc,
                                      double * p_snrsc,
                                      double * p_ampsc)
{
  EarthModel_t * p_earth;
  double pred_arrtime;
  int numtimedefphases;
  int numdet;
  int detpos;

  p_earth = p_netmodel->p_earth;
  
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
  if (!NetModel_IsSiteUp(p_netmodel, siteid, pred_arrtime))
    return 0;
  
  numdet = p_event->p_num_dets[siteid * numtimedefphases + phaseid];
  
  /* (mis)detection probability for primary detection */
  *p_detsc += EventDetectionPrior_LogProb(&p_netmodel->event_det_prior,
                                          numdet == 0 ? 0 : 1,
                                          p_event->evdepth,
                                          p_event->evmag,
                                          distance, siteid, phaseid);

  for (detpos = 0; detpos < numdet; detpos++)
  {
    Detection_t * det;
    Detection_t * prev_det;
    int detnum;

    detnum = p_event->p_all_detids[(siteid * numtimedefphases
                                    + phaseid) * MAX_PHASE_DET + detpos];

    det = p_netmodel->p_detections + detnum;

    if (detpos > 0)
    {
      int prev_detnum = p_event->p_all_detids[(siteid * numtimedefphases
                                    + phaseid) * MAX_PHASE_DET + detpos-1];
      
      prev_det = p_netmodel->p_detections + prev_detnum;
    }
    else
      prev_det = NULL;
    
    *p_detcnt += 1;

    if (!detpos)
      *p_dettimesc += ArrivalTimePrior_LogProb(&p_netmodel->arr_time_prior,
                                               det->time_det, pred_arrtime,
                                               det->deltim_det, siteid,
                                               phaseid);
    else
      *p_dettimesc += SecDetPrior_Time_LogProb(&p_netmodel->sec_det_prior,
                                               det->time_det,
                                               prev_det->time_det);
    

    *p_dettimesc -= NumFalseDet_LogTimeRate(&p_netmodel
                                            ->num_falsedet_prior,
                                            siteid);

    if (!detpos)
    {
      int pred_slow = EarthModel_ArrivalSlowness(p_earth, p_event->evlon,
                                                 p_event->evlat,
                                                 p_event->evdepth, phaseid,
                                                 siteid);

      *p_detslosc += ArrivalSlownessPrior_LogProb(&p_netmodel
                                                  ->arr_slo_prior,
                                                  det->slo_det, 
                                                  pred_slow,
                                                  det->delslo_det,
                                                  siteid, phaseid);
    }
    else
      *p_detslosc += SecDetPrior_Slow_LogProb(&p_netmodel->sec_det_prior,
                                              det->slo_det, prev_det->slo_det);
    
    *p_detslosc -= LOGPROB_UNIFORM_SLOWNESS;

    if (!detpos)
      *p_detazsc += ArrivalAzimuthPrior_LogProb(&p_netmodel->arr_az_prior,
                                                det->azi_det, 
                                                pred_az,
                                                det->delaz_det,
                                                siteid, phaseid);
    else
      *p_detazsc += SecDetPrior_Azimuth_LogProb(&p_netmodel->sec_det_prior,
                                                det->azi_det,
                                                prev_det->azi_det);
    
    *p_detazsc -= LOGPROB_UNIFORM_AZIMUTH;
    
    if (!detpos)
      *p_detphasesc += ArrivalPhasePrior_LogProb(&p_netmodel
                                                 ->arr_phase_prior,
                                                 det->phase_det,
                                                 phaseid);
    else
      *p_detphasesc += SecDetPrior_Phase_LogProb(&p_netmodel->sec_det_prior,
                                                 det->phase_det);
    
    *p_detphasesc -= FalseArrivalPhasePrior_LogProb(&p_netmodel
                                                    ->arr_phase_prior,
                                                    det->phase_det);

    if (!detpos)
      *p_snrsc += ArrivalSNRPrior_LogProb(&p_netmodel->arr_snr_prior,
                                          det->site_det, phaseid,
                                          det->snr_det);
    else
      *p_snrsc += SecDetPrior_SNR_LogProb(&p_netmodel->sec_det_prior,
                                          det->snr_det, prev_det->snr_det);
    
    *p_snrsc -= FalseArrivalSNRPrior_LogProb(&p_netmodel->arr_snr_prior,
                                             det->site_det, det->snr_det);
    
    /* a -1 amplitude suggests that the amplitude has not been observed */
    if (-1 != det->amp_det)
    {
      double ttime;
      ttime = EarthModel_ArrivalTime(p_earth, p_event->evlon, p_event->evlat,
                                     p_event->evdepth, 0, phaseid, siteid);
    
      if (!detpos)
        *p_ampsc += ArrivalAmplitudePrior_LogProb(&p_netmodel->arr_amp_prior,
                                                  p_event->evmag,
                                                  p_event->evdepth,
                                                  ttime, siteid, phaseid,
                                                  det->amp_det);
      else if (-1 != prev_det->amp_det)
        *p_ampsc += SecDetPrior_Amp_LogProb(&p_netmodel->sec_det_prior,
                                            det->amp_det, prev_det->amp_det);
    
      if (isnan(*p_ampsc))
      {
        printf("nan arr-amp mb %.2lg, dep %.2lg ttime %.2lg siteid %d "
               "phaseid %d amp %.2lg", p_event->evmag, p_event->evdepth,
               ttime, siteid, phaseid, det->amp_det);
        exit(1);
      }
    
      if ((!detpos) || (-1 != prev_det->amp_det))
        *p_ampsc -= FalseArrivalAmplitudePrior_LogProb(&p_netmodel
                                                       ->arr_amp_prior,
                                                       siteid, det->amp_det);
      if (isnan(*p_ampsc))
      {
        printf("nan false-amp siteid %d amp %.2lg", siteid, det->amp_det);
        exit(1);
      }
    }

    /* don't score any of the secondaries if not enabled */
    if (p_netmodel->enable_sec_arr)
      /* the probability of the next secondary detection */
      *p_detsc += SecDetPrior_Det_LogProb(&p_netmodel->sec_det_prior,
                                          (detpos+1) < numdet ? 1 : 0,
                                          det->amp_det);
    else
      break;
  }

  return 1;
}

/* returns 0 if the event can't be detected at the site */
int score_event_site_phase(NetModel_t * p_netmodel,
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
  
  poss = score_event_site_phase_int(p_netmodel, p_event, siteid, phaseid,
                                    distance, pred_az, &detcnt, &detsc,
                                    &dettimesc, &detslosc, &detazsc,
                                    &detphasesc, &detsnrsc, &detampsc);

  *p_score = detsc + dettimesc + detslosc + detazsc + detphasesc + detsnrsc
    + detampsc;

  return poss;
}

int score_event_site_phase_simple(NetModel_t * p_netmodel,
                                  const Event_t * p_event, int siteid,
                                  int phaseid, double * p_score)
{
  EarthModel_t * p_earth;
  double distance, pred_az;
  int poss;

  p_earth = p_netmodel->p_earth;

  distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                              siteid);

  pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                      p_event->evlat, siteid);
  
  poss = score_event_site_phase(p_netmodel, p_event, siteid, phaseid,
                                distance, pred_az, p_score);

  return poss;
}


static void score_event_int(NetModel_t * p_netmodel, Event_t * p_event, 
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

  p_earth = p_netmodel->p_earth;
  p_detections = p_netmodel->p_detections;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  *p_numsc += NumEventPrior_LogTimeRate(&p_netmodel->num_event_prior);
  
  *p_locsc += EventLocationPrior_LogProb(&p_netmodel->event_location_prior,
                                         p_event->evlon, p_event->evlat,
                                         p_event->evdepth);
  
  *p_magsc += EventMagPrior_LogProb(&p_netmodel->event_mag_prior,
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
      *p_poss_detcnt += score_event_site_phase_int(p_netmodel, p_event,
                                                   siteid, phaseid,
                                                   distance, pred_az,
                                                   p_detcnt,
                                                   p_detsc, p_dettimesc,
                                                   p_detslosc, p_detazsc,
                                                   p_detphasesc,
                                                   p_detsnrsc, p_detampsc);   
    }
  }
}

double score_event(NetModel_t * p_netmodel, Event_t * p_event)
{
  double numsc, locsc, magsc, detsc, dettimesc, detazsc, detslosc, detphasesc;
  double detsnrsc, detampsc;
  double score;
  int possdetcnt, detcnt;

  numsc = locsc = magsc = detsc = dettimesc = detazsc = detslosc = 
    detphasesc = detsnrsc = detampsc = 0;
  possdetcnt = detcnt = 0;
  
  score_event_int(p_netmodel, p_event, &numsc, &locsc, &magsc, &detsc, 
                  &dettimesc, &detazsc, &detslosc, &detphasesc, &detsnrsc,
                  &detampsc, &possdetcnt, &detcnt);
  
  score = numsc + locsc + magsc + detsc + dettimesc + detazsc + detslosc
    + detphasesc + detsnrsc + detampsc;

  return score;
}

double score_world(NetModel_t * p_netmodel,
                   int numevents, Event_t * p_events, int verbose)
{
  double score;
  double numsc;
  double magsc;
  double locsc;
  double detsc;
  double dettimesc;
  double detazsc;
  double detslosc;
  double detphasesc;
  double detsnrsc;
  double detampsc;
  
  int poss_detcnt;
  int detcnt;
  
  int i;

  if (verbose)
  {
    printf ("%d events:\n", numevents);
  }
  
  numsc = locsc = magsc = detsc = dettimesc = detazsc = detslosc = 
    detphasesc = detsnrsc = detampsc = 0;
  poss_detcnt = detcnt = 0;
  
  for (i=0; i<numevents; i++)
  {
    score_event_int(p_netmodel, p_events + i, &numsc, &locsc, &magsc, &detsc,
                    &dettimesc, &detazsc, &detslosc, &detphasesc,
                    &detsnrsc, &detampsc,
                    &poss_detcnt, &detcnt);
  }
  
  if (verbose)
  {
    printf("Event #,Time: score %lf, avg %lf\n", numsc, numsc/numevents);
    printf("Event Location: score %lf, avg %lf\n", locsc, locsc/numevents);
    printf("Event Mag: score %lf, avg %lf\n", magsc, magsc/numevents);
    printf("Event Det: score %lf, avg-event %lf avg-det %lf\n", detsc, 
           detsc/numevents, detsc/poss_detcnt);
    printf("Det Time: score %lf, avg-event %lf avg-det %lf\n", dettimesc, 
           dettimesc/numevents, dettimesc/detcnt);
    printf("Det Azimuth: score %lf, avg-event %lf avg-det %lf\n", detazsc, 
           detazsc/numevents, detazsc/detcnt);
    printf("Det Slowness: score %lf, avg-event %lf avg-det %lf\n", detslosc, 
           detslosc/numevents, detslosc/detcnt);
    printf("Det Phase: score %lf, avg-event %lf avg-det %lf\n", detphasesc, 
           detphasesc/numevents, detphasesc/detcnt);
    printf("Det SNR: score %lf, avg-event %lf avg-det %lf\n", detsnrsc, 
           detsnrsc/numevents, detsnrsc/detcnt);
    printf("Det AMP: score %lf, avg-event %lf avg-det %lf\n", detampsc, 
           detampsc/numevents, detampsc/detcnt);
    printf("Avg. # Detections: Possible %lf, Actual %lf\n", 
           ((double) poss_detcnt) / ((double) numevents),
           ((double) detcnt) / ((double) numevents));
  }
  
  score = numsc + locsc + magsc + detsc + dettimesc + detslosc + detazsc +
    detphasesc + detsnrsc + detampsc;
  
  if (verbose)
    printf("Total: %lf Avg. %lf\n", score, score/numevents);
  
  return score;
}

double prob_event(NetModel_t * p_netmodel, Event_t * p_event)
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
            
            score = score_event(p_netmodel, &curr_event);
            
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

double logprob_false(NetModel_t * p_netmodel, int numdets, int * p_detids,
                     int verbose)
{
  int i;
 
  double sum_logprob = 0;

  double * p_sum_site_amp;
  
  int * p_cnt_site;

  double worst_amp=0;
  
  /* array of sum logprob amp per site, initialized to zero */
  p_sum_site_amp = (double *)calloc(p_netmodel->numsites,
                                    sizeof(*p_sum_site_amp));

  /* keep track of the count for each site */
  p_cnt_site = (int *) calloc(p_netmodel->numsites,
                              sizeof(*p_cnt_site));
  
  
  for (i=0; i<numdets; i++)
  {
    double logprob;
    
    int detid = p_detids[i];
    
    Detection_t * p_det = p_netmodel->p_detections + detid;

    int siteid = p_det->site_det;
    
    /* a -1 amplitude signifies that the amplitude was not observed */
    if (-1 != p_det->amp_det)
    {
      logprob = FalseArrivalAmplitudePrior_LogProb(&p_netmodel->arr_amp_prior,
                                                   siteid,
                                                   p_det->amp_det);
      
      p_sum_site_amp[siteid] += logprob;
      p_cnt_site[siteid] ++;
      
      sum_logprob += logprob;
      
      if (logprob < worst_amp)
        worst_amp = logprob;
    }
  }

  if (verbose)
  {
    /*
    for (i=0; i<p_netmodel->numsites; i++)
      if (p_cnt_site[i] > 100)
        printf("Site[%d], cnt %d, Avg amplitude logprob: %lf\n",
               i, p_cnt_site[i], p_sum_site_amp[i] / p_cnt_site[i]);
    */
    printf("Avg. noise amp logprob: %lf   worst %lf\n", sum_logprob / numdets,
           worst_amp);
  }
  
  return sum_logprob;
}

