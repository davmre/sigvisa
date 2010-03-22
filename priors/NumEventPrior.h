typedef struct NumEventPrior_t
{
  double num_event_rate;
} NumEventPrior_t;


void NumEventPrior_Init_Params(NumEventPrior_t * prior, int nparams, 
                               const char * filename, double time_interval);

#define NumEventPrior_Init_Args while(0)

double NumEventPrior_LogProb(const NumEventPrior_t * dist, int val, int nargs);


