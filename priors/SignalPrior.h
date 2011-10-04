typedef struct SignalPrior_t
{
  int numsites;
  double * p_station_noise_means;
  double * p_station_noise_vars;

  double env_height;
  double env_decay; 

} SignalPrior_t;

void SignalPrior_Init_Params(SignalPrior_t * prior, const char * filename);

double SignalPrior_LogProb(SignalPrior_t * prior, int numsignals, Signal_t * p_signals, EarthModel_t * p_earth, int numevents, Event_t * events, PyArrayObject * arrtimes);

void SignalPrior_UnInit(SignalPrior_t * prior);
