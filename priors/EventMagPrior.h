typedef struct EventMagPrior_t
{
  double min_mag;
  double mag_rate;
} EventMagPrior_t;

void EventMagPrior_Init_Params(EventMagPrior_t * prior, int nparams,
                               const char * filename);

#define EventMagPrior_Init_Args while(0)

double EventMagPrior_LogProb(const EventMagPrior_t * dist, double val, 
                             int nargs);
