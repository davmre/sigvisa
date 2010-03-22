typedef struct EventLocationPrior_t
{
  double lonstep;
  double zstep;
  int numlon;
  int numlat;
  
  double * p_bucketprob;
} EventLocationPrior_t;

void EventLocationPrior_Init_Params(EventLocationPrior_t * dist, int nparams,
                                    const char * filename);

#define EventLocationPrior_Init_Args while(0)

double EventLocationPrior_LogProb(const EventLocationPrior_t * dist,
                                  const double * location, int nargs);

void EventLocationPrior_UnInit(EventLocationPrior_t * dist);
