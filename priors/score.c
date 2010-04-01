#include <math.h>

#include "../netvisa.h"

double score_world(EarthModel_t * p_earth, NetModel_t * p_netmodel,
                   int numevents, Event_t * p_events, int verbose)
{
  double score;
  double numsc;
  double magsc;
  double locsc;
  double detsc;
  double detcnt;
  int numphases;
  int numsites;
  
  int i;

  numsites = p_netmodel->event_det_prior.numsites;
  numphases = p_netmodel->event_det_prior.numphases;
  
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
    Event_t * p_event;
    double location[3];
    int siteid;
    int phaseid;
    
    p_event = p_events + i;
    
    Event2R3Vector(p_event, location);
    
    locsc += EventLocationPrior_LogProb(&p_netmodel->event_location_prior,
                                        location, 0);
    magsc += EventMagPrior_LogProb(&p_netmodel->event_mag_prior,
                                   p_event->evmag, 0);
    for (siteid=0; siteid<numsites; siteid++)
    {
      /* TODO: check if the site is up */
      for (phaseid=0; phaseid<numphases; phaseid++)
      {
        /* check if the phase is in range */
        if (EarthModel_InRange(p_earth, p_event->evlon, p_event->evlat,
                               p_event->evdepth, phaseid, siteid))
        {
          detsc += EventDetectionPrior_LogProb(&p_netmodel->event_det_prior,
                                               p_event->p_detids[siteid 
                                                                 * numphases 
                                                                 +phaseid]==-1?
                                               BLOG_FALSE : BLOG_TRUE,
                                               4, siteid, phaseid,
                                               p_event->evmag, location);
          detcnt += 1;
        }
      }
    }
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
