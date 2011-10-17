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


/* double score_world_sig(SigModel_t * p_sigmodel, 
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
  
  // p(events, ttimes | signals) \propto p(signals | events, ttimes)p(ttimes|events)p(events)  

  // p(events): prior probabilities of events 

  numsc = NumEventPrior_LogTimeRate(&p_sigmodel->num_event_prior);

  for (int i=0; i<numevents; i++) {
    Event_t * p_event = p_events + i;
    locsc += EventLocationPrior_LogProb(&p_sigmodel->event_location_prior,
					p_event->evlon, p_event->evlat,
					p_event->evdepth);
    magsc += EventMagPrior_LogProb(&p_sigmodel->event_mag_prior,
				   p_event->evmag, 0);
  }
  // p(ttimes | events): prior probabilities of travel times, given event (and station) locations
  
  atimesc = ArrivalTimeJointPrior_LogProb(&p_sigmodel->arr_time_joint_prior, p_sigmodel->p_earth, numevents, p_events, arrtimes);
  
  // p(signals| events, ttimes): probability of signals given events and travel times. 
  sigsc = SignalPrior_LogProb(&p_sigmodel->sig_prior, numsignals, p_signals, p_sigmodel->p_earth, numevents, p_events, arrtimes);
  


  double score = numsc + locsc + magsc + atimesc + sigsc;
  fprintf(stdout, "score is %lf + %lf + %lf + %lf + %lf = %lf\n", numsc, locsc, magsc, atimesc, sigsc, score);
  return score;
}
*/



/* returns 0 if the event - phase is impossible to be detected at the
 * site, 1 otherwise 
 * if detected returns log probability of detection given event 
 *                     - log probability noise 
 * else returns log probability of mis-detection given event */
static int score_event_int_sig(SigModel_t * p_sigmodel,
			       const Event_t * p_event,
			       int num_other_events, 
			       const Event_t ** pp_other_events,
			       double * p_arrtimesc,
			       double * p_arrazsc,
			       double * p_arrslosc,
			       double * p_ampsc,
			       double * p_sigsc)
{
  EarthModel_t * p_earth;
  double pred_arrtime;
  double ttime;
  int numtimedefphases, numsites;

  p_earth = p_sigmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  

  for (int siteid = 0; siteid < numsites; ++siteid) {
    for (int phaseid = 0; phaseid < numtimedefphases; ++phaseid) {

      pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
					    p_event->evlat, p_event->evdepth,
					    p_event->evtime, phaseid,
					    siteid);
      /* check if the site is in the shadow zone for the event - phase */
      if (pred_arrtime < 0)
	continue;

      /* check if the site is up */
      //if (!SigModel_IsSiteUp(p_sigmodel, siteid, pred_arrtime))
      //continue;
  
      Arrival_t * arr = p_event->p_arrivals +(siteid*numtimedefphases + phaseid);

      // TODO: use joint prior properly
      *p_arrtimesc += ArrivalTimePrior_LogProb(&p_sigmodel->arr_time_joint_prior.single_prior,
					       arr->time, pred_arrtime,
					       0, siteid,
					       phaseid);
      
      
      double pred_az = EarthModel_ArrivalAzimuth(p_earth, 
						p_event->evlon, 
						p_event->evlat, 
						siteid);
      *p_arrazsc += ArrivalAzimuthPrior_LogProb(&p_sigmodel->arr_az_prior,
						arr->azi, 
						pred_az,
						0,
						siteid, phaseid);
      
      double pred_slo = EarthModel_ArrivalSlowness(p_earth, 
						 p_event->evlon, 
						 p_event->evlat, 
						 p_event->evdepth, 
						 phaseid, siteid); 
      *p_arrslosc += ArrivalSlownessPrior_LogProb(&p_sigmodel->arr_slo_prior,
						  arr->azi, 
						  pred_slo,
						  0,
						  siteid, phaseid);
      
      ttime = EarthModel_ArrivalTime(p_earth, p_event->evlon, p_event->evlat,
				     p_event->evdepth, 0, phaseid, siteid);
      *p_ampsc += ArrivalAmplitudePrior_LogProb(&p_sigmodel->arr_amp_prior,
						p_event->evmag, p_event->evdepth,
						ttime, siteid, phaseid,
						arr->amp);
      if (isnan(*p_ampsc))
	{
	  printf("nan arr-amp mb %.2lg, dep %.2lg ttime %.2lg siteid %d phaseid %d"
		 " amp %.2lg", p_event->evmag, p_event->evdepth,
		 ttime, siteid, phaseid, arr->amp);
	  exit(1);
	}
    }
  }

  *p_sigsc += SignalPrior_Score_Event(&p_sigmodel->sig_prior, p_sigmodel, p_event,  num_other_events, pp_other_events);

  return 1;
}


int score_event_sig(SigModel_t * p_sigmodel,
		     Event_t * p_event,
		     int num_other_events, const Event_t ** pp_other_events)
{
  int      poss;
  double   arrtimesc;
  double   arrslosc;
  double   arrazsc;
  double   arrampsc;
  double   sigsc;


  arrtimesc = arrazsc = arrslosc = arrampsc = sigsc = 0;

  poss = score_event_int_sig(p_sigmodel, p_event, 
			     num_other_events, pp_other_events,
			     &arrtimesc, &arrazsc, 
			     &arrslosc, &arrampsc, &sigsc);

  p_event->evscore = arrtimesc + arrazsc + arrslosc + arrampsc + sigsc;

  return poss;
}


 /*
double prob_event(SigModel_t * p_sigmodel, Event_t * p_event)
{
  Event_t curr_event;
  double logprob;
  double step_z;

  // make a copy of the event to work with 
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
            
            score = score_event_sig(p_sigmodel, &curr_event);
            
            logprob = log_add_exp(logprob, score);
          }
        }
      }
    }
  }

  logprob += log(STEP_LON) + log(step_z) + log (STEP_DEPTH) + log(STEP_TIME)\
    + log(STEP_MAG);
  
  return logprob;
}*/
