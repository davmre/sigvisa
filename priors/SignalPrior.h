
#include <Python.h>


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

  int ar_n;
  double * p_ar_coeffs;
  double ar_noise_sigma2;
} SignalPrior_t;

void SignalPrior_Init_Params(SignalPrior_t * prior, const char * filename, int numsites);


void arrival_list(EarthModel_t * p_earth, int siteid, double min_time, double max_time, int num_events, const Event_t ** pp_events, int * num_arrivals, Arrival_t *** ppp_arrivals);
void det_arrivals(void * p_sigmodel_v, ChannelBundle_t * p_segment, int * num_arrivals, Arrival_t *** ppp_arrivals);


double det_likelihood(void * p_sigmodel_v);

double SignalPrior_Score_Event_Site(SignalPrior_t * prior, void * p_sigmodel_v, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events);

double SignalPrior_Score_Event(SignalPrior_t * prior, void * p_sigmodel_v, const Event_t * event, int num_other_events, const Event_t ** pp_other_events);

void SignalPrior_ThreeAxisEnvelope(SignalPrior_t * prior, 
				   EarthModel_t * p_earth, 
				   int numevents, 
				   const Event_t ** pp_events,
				   ChannelBundle_t * p_segment,
				   ChannelBundle_t * p_wave_segment);

void SignalPrior_SampleThreeAxisAR(SignalPrior_t * prior, 
				   EarthModel_t * p_earth, 
				   int samplePerturb,
				   int sampleNoise,
				   int num_arrivals,
				   const Arrival_t ** pp_arrivals,
				   ChannelBundle_t * p_segment);

void SignalPrior_UnInit(SignalPrior_t * prior);
