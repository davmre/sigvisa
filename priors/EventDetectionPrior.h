typedef struct EventDetectionPrior_t
{
  int numsites;
  int numphases;
  double * p_site_mag;
  double * p_site_phases;
  double * p_site_bias;
} EventDetectionPrior_t;

void EventDetectionPrior_Init_Params(EventDetectionPrior_t * dist, int nparams,
                                     const char * filename);

#define EventDetectionPrior_Init_Args while(0)

double EventDetectionPrior_LogProb(const EventDetectionPrior_t * dist,
                                   int is_detected, int nargs,
                                   int siteid, int phaseid, double evmag,
                                   double * evloc);

void EventDetectionPrior_UnInit(EventDetectionPrior_t * dist);
