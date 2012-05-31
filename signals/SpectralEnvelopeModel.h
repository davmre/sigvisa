
#include <Python.h>

#include "SignalModelUtil.h"

typedef struct BandModel_t {

  ARProcess_t channel_noise_models[NUM_CHANS];
  ARProcess_t wiggle_model;

  /* we will eventually need models for all of the envelope params,
     coda decay, etc., but at the moment those are python-only... */
} BandModel_t;

typedef struct Spectral_StationModel_t {

  BandModel_t bands[NUM_BANDS];
  
} Spectral_StationModel_t;

typedef struct Spectral_Envelope_Model_t
{
  int numsites;
  Spectral_StationModel_t * p_stations;

} Spectral_Envelope_Model_t;

void Spectral_Envelope_Model_Init_Params(void * pv_params, int numsites);

int Spectral_Envelope_Model_Has_Model(void * pv_model, int siteid, int chan);

double Spectral_Envelope_Model_Likelihood(void * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals);

double Spectral_Envelope_Model_Sample(void * pv_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int sample_noise, int sample_wiggles);

void Spectral_Envelope_Model_UnInit(void * pv_params);
