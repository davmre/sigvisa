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
    printf ("%d events:\n", numevents);
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

  return score;
}