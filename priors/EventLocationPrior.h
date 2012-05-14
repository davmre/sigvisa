typedef struct EventLocationPrior_t
{
  double lonstep;
  double zstep;
  int numlon;
  int numlat;
  
  double * p_bucketprob;  /* numlon x numlat array of probabilities */

  double * p_lonprob; /* the probability of each longitude size = numlon */
  
  /* numlon x numlat array of latitude probabilities normalized for
   * each longitude */
  double * p_latprob;

  double north_pole_prob;                    /* probability of a
                                              * hypothetical bucket at
                                              * the north pole */
} EventLocationPrior_t;

void EventLocationPrior_Init_Params(EventLocationPrior_t * dist,
                                    const char * filename);

double EventLocationPrior_LogProb(const EventLocationPrior_t * dist,
                                  double lon, double lat, double depth);

void EventLocationPrior_Sample(const EventLocationPrior_t * dist,
                               double * p_lon, double * p_lat, 
                               double * p_depth);

void EventLocationPrior_UnInit(EventLocationPrior_t * dist);
