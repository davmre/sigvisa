
#include <Python.h>


typedef struct Envelope_StationModel_t {
  double chan_means[NUM_CHANS];
  double chan_vars[NUM_CHANS];


  // TODO: make distance-dependent
  double env_height;
  double env_p_onset;
  double env_p_decay;
  double env_s_onset;
  double env_s_decay;

  int ar_n;
  double * p_ar_coeffs;
  double ar_noise_sigma2;
} Envelope_StationModel_t;

typedef struct Envelope_SignalModel_t
{
  int numsites;
  Envelope_StationModel_t * p_stations;

} Envelope_SignalModel_t;

void Envelope_SignalModel_Init_Params(void * pv_params, int numsites);

void Envelope_SignalModel_Set_Params(void * pv_params, int station, PyObject * py_dict);

int Envelope_SignalModel_Has_Model(void * pv_model, int siteid, int chan);

double Envelope_SignalModel_Likelihood(void * p_sigmodel, ChannelBundle_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals);


void Envelope_SignalModel_SampleThreeAxis(void * pv_params, 
					  EarthModel_t * p_earth, 
					  ChannelBundle_t * p_segment,
					  int num_arrivals,
					  const Arrival_t ** pp_arrivals,
					  int samplePerturb,
					  int sampleNoise);

void Envelope_SignalModel_UnInit(void * pv_params);
