typedef struct EventDetectionPrior_t
{
  int numsites;
  int numtimedefphases;
  double * p_coeff_mag;
  double * p_coeff_dist;
  double * p_coeff_phases;
  double * p_coeff_bias;
} EventDetectionPrior_t;

void EventDetectionPrior_Init_Params(EventDetectionPrior_t * dist,
                                     const char * filename);

double EventDetectionPrior_LogProb(const EventDetectionPrior_t * prior,
                                   int is_detected,
                                   double evdepth, double evmag, double dist,
                                   int siteid, int phaseid);

void EventDetectionPrior_UnInit(EventDetectionPrior_t * dist);
