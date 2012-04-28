
#include <Python.h>


typedef struct Spectral_StationModel_t {
  double chan_means[NUM_CHANS];
  double chan_vars[NUM_CHANS];

  // TODO: make distance-dependent

  double p_b0[NUM_BANDS];
  double p_b1[NUM_BANDS];
  double p_b2[NUM_BANDS];
  double s_b0[NUM_BANDS];
  double s_b1[NUM_BANDS];
  double s_b2[NUM_BANDS];

  double p_gamma0[NUM_BANDS];
  double p_gamma1[NUM_BANDS];
  double p_gamma2[NUM_BANDS];
  double s_gamma0[NUM_BANDS];
  double s_gamma1[NUM_BANDS];
  double s_gamma2[NUM_BANDS];

  double override_b;
  double override_gamma;
  double override_height;

  int ar_n;
  double * p_ar_coeffs;
  double ar_noise_sigma2;

} Spectral_StationModel_t;

typedef struct Spectral_Envelope_Model_t
{
  int numsites;
  Spectral_StationModel_t * p_stations;

} Spectral_Envelope_Model_t;

void Spectral_Envelope_Model_Init_Params(void * pv_params, int numsites);

void Spectral_Envelope_Model_Set_Params(void * pv_params, int station, PyObject * py_dict);

int Spectral_Envelope_Model_Has_Model(void * pv_model, int siteid, int chan);

double Spectral_Envelope_Model_Likelihood(void * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals);


void Spectral_Envelope_Model_SampleThreeAxis(void * pv_params, 
					  EarthModel_t * p_earth, 
					  Segment_t * p_segment,
					  int num_arrivals,
					  const Arrival_t ** pp_arrivals,
					  int samplePerturb,
					  int sampleNoise);

void Spectral_Envelope_Model_UnInit(void * pv_params);
