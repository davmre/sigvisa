



typedef struct StationNoiseModel_t {
  double chan_means[NUM_CHANS];
  double chan_vars[NUM_CHANS];
} StationNoiseModel_t;

typedef struct SignalPrior_t
{
  int numsites;
  StationNoiseModel_t * p_stations;

  double env_height;
  double env_onset; 
  double env_decay; 

} SignalPrior_t;

void SignalPrior_Init_Params(SignalPrior_t * prior, const char * filename, int numsites);

double SignalPrior_Score_Event_Site(SignalPrior_t * prior, void * p_sigmodel_v, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events);

double SignalPrior_Score_Event(SignalPrior_t * prior, void * p_sigmodel_v, const Event_t * event, int num_other_events, const Event_t ** pp_other_events);

void SignalPrior_ThreeAxisEnvelope(SignalPrior_t * prior, 
				   EarthModel_t * p_earth, 
				   int numevents, 
				   const Event_t ** pp_events,
				   ChannelBundle_t * p_segment,
				   ChannelBundle_t * p_wave_segment);

void SignalPrior_UnInit(SignalPrior_t * prior);
