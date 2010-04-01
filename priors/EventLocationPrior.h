typedef struct EventLocationPrior_t
{
  double lonstep;
  double zstep;
  int numlon;
  int numlat;
  
  double * p_bucketprob;
} EventLocationPrior_t;

void EventLocationPrior_Init_Params(EventLocationPrior_t * dist,
                                    const char * filename);

double EventLocationPrior_LogProb(const EventLocationPrior_t * dist,
                                  double lon, double lat, double depth);

void EventLocationPrior_UnInit(EventLocationPrior_t * dist);
