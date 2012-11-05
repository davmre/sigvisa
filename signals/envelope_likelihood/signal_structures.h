#ifndef SIGNAL_STRUCTURES_H
#define SIGNAL_STRUCTURES_H

#include "../../signal.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

typedef void(*SMSetParams_f)(void * pv_params, int station, PyObject * py_dict);

typedef int(*SMHasModel_f)(void * p_sigmodel, int siteid, int chan, int band);

typedef double(*SMLikelihood_f)(void * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals);

typedef void(*SMSample_f)(void * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int * bands_to_sample, int samplePerturb, int sampleNoise);

typedef void(*SMUnInit_f)(void * pv_params);

typedef struct SignalModel_t
{
  SMSetParams_f set_params;
  SMHasModel_f has_model;
  SMLikelihood_f likelihood;
  SMSample_f sample;
  SMUnInit_f uninit;

  void * pv_params;

} SignalModel_t;


typedef struct ARProcess_t {

  double mean;
  int order;
  double * coeffs; // most recent first
  double sigma2;

} ARProcess_t;

typedef struct ArrivalWaveform {
  double start_time;
  long idx;
  double end_time;
  long len;

  Trace_t * p_abstract_trace;

  struct ArrivalWaveform * next_start;
  struct ArrivalWaveform * next_end;
  struct ArrivalWaveform * next_active;

  ARProcess_t ar_processes[NUM_CHANS];
  int wiggle_ids[NUM_CHANS];
  double projection_coeffs[NUM_CHANS];


  // used for sampling
  double * last_perturbs;

} ArrivalWaveform_t;

typedef struct ARWLists_t {

  struct ArrivalWaveform * st_head;
  struct ArrivalWaveform * et_head;

  struct ArrivalWaveform * st_ptr;
  struct ArrivalWaveform * et_ptr;
  struct ArrivalWaveform * active_arrivals;

} ARWLists_t;

typedef struct BandModel_t {

  ARProcess_t channel_noise_models[NUM_CHANS];
  ARProcess_t wiggle_model[NUM_CHANS][NUM_TD_PHASES];

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

#endif
