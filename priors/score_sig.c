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

double score_event_sta_prior(SigModel_t * p_sigmodel,
				  const Event_t * p_event,
			    int siteid) {
EarthModel_t * p_earth;
  double pred_arrtime;
  double ttime;
  int numtimedefphases, numsites;
 
  double arrtimesc, arrazsc, arrampsc, arrslosc;
  double ev_sta_score = 0;

  p_earth = p_sigmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  
  for (int phaseid = 0; phaseid < MAX_PHASE(numtimedefphases); ++phaseid) {
    if (!USE_PHASE(phaseid)) continue;
    pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
					  p_event->evlat, p_event->evdepth,
					  p_event->evtime, phaseid,
					  siteid-1);
    if (pred_arrtime < 0)
      continue;

    if (!have_signal(p_sigmodel, siteid, pred_arrtime - 5, pred_arrtime+MAX_ENVELOPE_LENGTH)) {
      continue;
    }
    
    Arrival_t * arr = p_event->p_arrivals +((siteid-1)*numtimedefphases + phaseid);
    //    printf("  scoring arrival for siteid %d phaseid %d: ", siteid, phaseid);
    //    print_arrival(arr);
    
    arrtimesc = ArrivalTimePrior_LogProb(&p_sigmodel->arr_time_joint_prior.single_prior, arr->time, pred_arrtime, 0, siteid-1, phaseid);
    
    double pred_az = EarthModel_ArrivalAzimuth(p_earth, 
					       p_event->evlon, 
					       p_event->evlat, 
					       siteid-1);
    arrazsc = ArrivalAzimuthPrior_LogProb(&p_sigmodel->arr_az_prior,
					  arr->azi, 
					  pred_az,
					  0,
					  siteid-1, phaseid);
    
   double pred_slo = EarthModel_ArrivalSlowness(p_earth, 
						p_event->evlon, 
						p_event->evlat, 
						p_event->evdepth, 
						phaseid, siteid-1); 
   arrslosc = ArrivalSlownessPrior_LogProb(&p_sigmodel->arr_slo_prior,
					   arr->slo, 
					   pred_slo,
					   0,
					   siteid-1, phaseid);
   
   ttime = pred_arrtime - p_event->evtime;
   arrampsc = ArrivalAmplitudePrior_LogProb(&p_sigmodel->arr_amp_prior,
					    p_event->evmag, p_event->evdepth,
					    ttime, siteid-1, phaseid,
					    arr->amp);
   if (isnan(arrampsc) || arrampsc < -DBL_MAX) {
     LogTrace("nan arr-amp mb %.2lf, dep %.2lf ttime %.2lf siteid %d phaseid %d"
	    " amp %.2lf", p_event->evmag, p_event->evdepth,
	    ttime, siteid, phaseid, arr->amp);
     //exit(1);
   }

   arr->score = arrtimesc + arrazsc + arrslosc + arrampsc;
   assert(!isnan(arr->score));
   ev_sta_score += arr->score;

   // printf(" score of arrival %d at station %d is time %lf + az %lf + slo %lf + amp %lf, for cumulative event/station score %lf\n", phaseid, siteid, arrtimesc, arrazsc, arrslosc, arrampsc, ev_sta_score);


 }

  return ev_sta_score;

}

double score_event_sta_sig(SigModel_t * p_sigmodel,
				  const Event_t * p_event,
				  int siteid,
				  int num_other_events, 
				  const Event_t ** pp_other_events) {

  double ev_sta_score = score_event_sta_prior(p_sigmodel, p_event, siteid);
  double sigsc = SignalPrior_Score_Event_Site(&p_sigmodel->sig_prior, p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
  ev_sta_score += sigsc;
 
 assert(!isnan(ev_sta_score));

 return ev_sta_score;

}


double score_event_evprior(SigModel_t * p_sigmodel, Event_t * p_event) {
    double numsc, locsc, magsc;

  numsc = NumEventPrior_LogTimeRate(&p_sigmodel->num_event_prior);
  
  locsc = EventLocationPrior_LogProb(&p_sigmodel->event_location_prior,
                                         p_event->evlon, p_event->evlat,
                                         p_event->evdepth);
  
  magsc = EventMagPrior_LogProb(&p_sigmodel->event_mag_prior,
                                    p_event->evmag, 0);

  return numsc + locsc + magsc;
}

double score_event_prior(SigModel_t * p_sigmodel, Event_t * p_event) {
  EarthModel_t * p_earth;
  double pred_arrtime;
  int numsites;

  p_earth = p_sigmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);

  double score = score_event_evprior(p_sigmodel, p_event);

  for (int siteid = 1; siteid <= numsites; ++siteid) {
    
    pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
					  p_event->evlat, p_event->evdepth,
					  p_event->evtime, 0,
					  siteid-1);
    /* check if the site is in the shadow zone for the event - phase */
    if (pred_arrtime < 0)
      continue;

    /* check if the site is up */
    //if (!SigModel_IsSiteUp(p_sigmodel, siteid, pred_arrtime))
    //continue;
    if (!have_signal(p_sigmodel, siteid, pred_arrtime - 5, pred_arrtime+MAX_ENVELOPE_LENGTH)) {
      continue;
    } 

    score += score_event_sta_prior(p_sigmodel,
				   p_event,
				   siteid);
  }

  return score;

}

/* returns 0 if the event - phase is impossible to be detected at the
 * site, 1 otherwise 
 * if detected returns log probability of detection given event 
 *                     - log probability noise 
 * else returns log probability of mis-detection given event */
int score_event_sig(SigModel_t * p_sigmodel,
		    Event_t * p_event,
		    int num_other_events, 
		    const Event_t ** pp_other_events)
{
  EarthModel_t * p_earth;
  double pred_arrtime;
  int numsites;
  
  p_earth = p_sigmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);
    
  p_event->evscore = score_event_evprior(p_sigmodel, p_event);
 
  for (int siteid = 1; siteid <= numsites; ++siteid) {
    
      pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
					    p_event->evlat, p_event->evdepth,
					    p_event->evtime, 0,
					    siteid-1);
      /* check if the site is in the shadow zone for the event - phase */
      if (pred_arrtime < 0)
	continue;

      /* check if the site is up */
      //if (!SigModel_IsSiteUp(p_sigmodel, siteid, pred_arrtime))
      //continue;
      if (!have_signal(p_sigmodel, siteid, pred_arrtime - 5, pred_arrtime+MAX_ENVELOPE_LENGTH)) {
	continue;
      } 

      p_event->evscore += score_event_sta_sig(p_sigmodel,
					      p_event,
					      siteid,
					      num_other_events, 
					      pp_other_events);
  }

  return 1;
}


