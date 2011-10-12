#include "../infer/infer.h"

typedef struct StationNoiseModel_t {
  double chan_means[NUM_CHANS];
  double chan_vars[NUM_CHANS];
} StationNoiseModel_t;

typedef struct SignalPrior_t
{
  int numsites;
  StationNoiseModel_t * stations;

  double env_height;
  double env_decay; 

} SignalPrior_t;

void SignalPrior_Init_Params(SignalPrior_t * prior, const char * filename, int numsites);

double SignalPrior_Score_Event(SignalPrior_t * prior, SigModel * p_sigmodel, Event_t * event, int num_other_events, Events ** pp_other_events)l

void SignalPrior_UnInit(SignalPrior_t * prior);
