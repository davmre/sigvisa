typedef struct Signal_t
{
  int len;
  double * p_data;

  double start_time;
  double hz;

  int siteid;
  int chan;
  
} Signal_t;

typedef struct SignalPrior_t
{
  int numstations;
  double * p_station_noise_means;
  double * p_station_noise_vars;

  double env_height;
  double env_decay; 

} SignalPrior_t;

void SignalPrior_Init_Params(SignalPrior_t * prior, const char * filename);

void SignalPrior_LogProb(SignalPrior_t * prior, const char * filename);

void SignalPrior_UnInit(SignalPrior_t * prior);
