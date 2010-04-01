#include <math.h>
#include <stdlib.h>

#include "../netvisa.h"

void score_event(NetModel_t * p_netmodel,
                 Event_t * p_event, double * p_locsc, double * p_magsc,
                 double * p_detsc, int * p_detcnt)
{
  EarthModel_t * p_earth;
  int numphases;
  int numsites;
  int siteid;
  int phaseid;

  p_earth = p_netmodel->p_earth;
  
  numsites = p_netmodel->event_det_prior.numsites;
  numphases = p_netmodel->event_det_prior.numphases;
    
  *p_locsc += EventLocationPrior_LogProb(&p_netmodel->event_location_prior,
                                         p_event->evlon, p_event->evlat,
                                         p_event->evdepth);
  
  *p_magsc += EventMagPrior_LogProb(&p_netmodel->event_mag_prior,
                                    p_event->evmag, 0);

  for (siteid=0; siteid<numsites; siteid++)
  {
    double distance;
    
    /* check if the site is up */
    if (!NetModel_IsSiteUp(p_netmodel, siteid, p_event->evtime)
        || !NetModel_IsSiteUp(p_netmodel, siteid,
                              p_event->evtime + UPTIME_QUANT))
      continue;
    
    distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                siteid);
    
    for (phaseid=0; phaseid<numphases; phaseid++)
    {
      /* check if the site is in the shadow zone for the event */
      if (EarthModel_IsTimeDefPhase(p_earth, phaseid) &&
          (EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                  p_event->evlat, p_event->evdepth,
                                  p_event->evtime, phaseid, siteid) < 0))
        continue;
        
      *p_detsc += EventDetectionPrior_LogProb(&p_netmodel->event_det_prior,
                                              p_event->p_detids[siteid 
                                                                * numphases 
                                                                +phaseid]==-1?
                                              0 : 1,
                                              p_event->evdepth, p_event->evmag,
                                              distance, siteid, phaseid);      
      *p_detcnt += 1;
    }
  }
}

double score_world(NetModel_t * p_netmodel,
                   int numevents, Event_t * p_events, int verbose)
{
  double score;
  double numsc;
  double magsc;
  double locsc;
  double detsc;
  int detcnt;
  
  int i;

  score = 0;
  
  /* score the number of events */
  numsc = NumEventPrior_LogProb(&p_netmodel->num_event_prior, numevents, 0);
  
  if (verbose)
  {
    printf ("%d events:\n", numevents);
    printf ("# events: score %lf\n", numsc);
  }
  
  score += numsc;

  detsc = locsc = magsc = 0;
  detcnt = 0;
  
  for (i=0; i<numevents; i++)
  {
    score_event(p_netmodel, p_events + i, &locsc, &magsc, &detsc,
                &detcnt);
  }
  
  if (verbose)
  {
    printf("Event Location: score %lf, avg %lf\n", locsc, locsc/numevents);
    printf("Event Mag: score %lf, avg %lf\n", magsc, magsc/numevents);
    printf("Event Det: score %lf, avg %lf\n", detsc, detsc/detcnt);
  }
  
  score += locsc + magsc + detsc;

  if (verbose)
    printf("Total: %lf\n", score);
  
  return score;
}
