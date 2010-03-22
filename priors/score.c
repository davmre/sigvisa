
#include "../netvisa.h"

double score_world(NetModel_t * p_netmodel, int numevents, Event_t * p_events,
                   int verbose)
{
  double score;
  double sc;
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

  sc = 0;
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event;
    double location[3];
    
    p_event = p_events + i;
    
    Event2R3Vector(p_event, location);
    
    sc += EventLocationPrior_LogProb(&p_netmodel->event_location_prior,
                                     location, 0);
  }
  
  printf("Event Location: score %lf, avg %lf\n", sc, sc/numevents);
  
  score += sc;

  return score;
}
