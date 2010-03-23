#include <math.h>

#include "../netvisa.h"

double score_world(NetModel_t * p_netmodel, int numevents, Event_t * p_events,
                   int verbose)
{
  double score;
  double sc;
  double magsc;
  int i;
  
  score = 0;
  
  /* score the number of events */
  sc = NumEventPrior_LogProb(&p_netmodel->num_event_prior, numevents, 0);
  
  if (verbose)
  {
    printf ("%d events:\n", numevents);
    printf ("# events: score %lf\n", sc);
  }
  
  score += sc;

  sc = magsc = 0;
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event;
    double location[3];
    
    p_event = p_events + i;
    
    Event2R3Vector(p_event, location);
    
    sc += EventLocationPrior_LogProb(&p_netmodel->event_location_prior,
                                     location, 0);
    magsc += EventMagPrior_LogProb(&p_netmodel->event_mag_prior,
                                   p_event->evmag, 0);
  }
  
  if (verbose)
  {
    printf("Event Location: score %lf, avg %lf\n", sc, sc/numevents);
    printf("Event Mag: score %lf, avg %lf\n", magsc, magsc/numevents);
  }
  
  score += sc + magsc;

  if (verbose)
    printf("Total: %lf\n", score);
  
  return score;
}
