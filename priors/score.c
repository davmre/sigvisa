
#include "../netvisa.h"

double score_world(NetModel_t * p_netmodel, int numevents)
{
  double sc;
  
  sc = 0;
  
  /* score the number of events */
  sc += NumEventPrior_LogProb(&p_netmodel->num_event_prior, numevents, 0);

  return sc;
}
