#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdarg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../sigvisa.h"
#include "signal_structures.h"
#include "SignalModelUtil.h"
#include "kalman_filter.h"

void Spectral_Envelope_Model_Init_Params(Spectral_Envelope_Model_t * p_params,  int numsites) {

  p_params->numsites = numsites;

  /* using calloc (instead of malloc) is important here since we
     depend on the p_ar_coeffs pointer being NULL iff unallocated */
  p_params->p_stations = calloc(numsites, sizeof(Spectral_StationModel_t));
}

void Spectral_Envelope_Model_Set_Params(Spectral_Envelope_Model_t * p_params, int siteid, PyObject * py_dict) {

  Spectral_StationModel_t * sta = NULL;
  if (siteid > p_params->numsites) {
    LogError("invalid siteid %d", siteid);
    exit(EXIT_FAILURE);
  } else if (siteid > 0) {
    sta = p_params->p_stations + siteid - 1;
  }

  PyObject *py_key, *py_value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(py_dict, &pos, &py_key, &py_value)) {
    char * key = PyString_AsString(py_key);

    if (sta != NULL) {

      if (strcmp(key, "override_b") == 0) {
	LogError("test test test");
	exit(EXIT_FAILURE);
      } else {
	LogError("unrecognized param %s at siteid %d", key, siteid);
	exit(EXIT_FAILURE);
      }
    } else {
      LogError("unrecognized global param %s", key);
      exit(EXIT_FAILURE);
    }
    CHECK_ERROR;
  }
}

int Spectral_Envelope_Model_Has_Model(SigModel_t * p_sigmodel, int siteid, int chan) {
  SignalModel_t * p_model = &p_sigmodel->signal_model;
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *)p_model->pv_params;

  /* chan < 0 means return true if we have a model for any channel at
     this station */
  if (chan < 0) {
    if (p_params->p_stations[siteid-1].bands[DEFAULT_BAND].wiggle_model.coeffs == NULL) {
      return FALSE;
    }

    for (int i=0; i < NUM_CHANS; ++i) {
      if (p_params->p_stations[siteid-1].bands[DEFAULT_BAND].channel_noise_models[i].coeffs != NULL) {
	return TRUE;
      }
    }
    return FALSE;
  }

  else {
    // LogTrace("checking var %lf for site %d chan %d", p_params->p_stations[siteid-1].chan_vars[chan], siteid, chan);
    // LogTrace("p_model %p sta %p", p_params, p_params->p_stations + siteid-1);
    return (p_params->p_stations[siteid-1].bands[DEFAULT_BAND].wiggle_model.coeffs != NULL) && (p_params->p_stations[siteid-1].bands[DEFAULT_BAND].channel_noise_models[chan].coeffs != NULL);
  }

}


/*
   Input: an arrival specifying envelope parameters, and a trace object specifying the sampling rate.
   Returns: sets p_data, len, and start_time in the Trace.
*/
void abstract_spectral_logenv_raw(const Arrival_t * p_arrival, Trace_t * p_trace) {

  // calculate length of envelope to generate
  p_trace->len = (long)((MIN_LOGENV_CUTOFF- p_arrival->amp) / p_arrival->coda_decay * p_trace->hz);

  // allocate memory
  p_trace->p_data = (double *) calloc(p_trace->len, sizeof(double));
  double * d = p_trace->p_data;
  if (d == NULL) {
    printf("error allocating memory for means in abstract_env, len %ld\n", p_trace->len);
    exit(-1);
  }

  // generate onset
  long peak_idx = (p_arrival->peak_time - p_arrival->time) * p_trace->hz;
  double onset_slope = p_arrival->peak_amp / peak_idx;
  for (long t=0; t < peak_idx; ++t) {
    d[t] = t * onset_slope;
  }

  // generate decay
  /*  double alpha = (exp(p_arrival->peak_amp) - exp(p_arrival->amp)) / exp(p_arrival->amp);
  double gamma = ( log(alpha) - log(0.1)) / log(p_arrival->peak_decay);
  gamma = (gamma > 0) ? gamma : 0;
  for (long t=peak_idx; t < p_trace->len; ++t) {
    double t_off = (t - peak_idx)/p_trace->hz;
    d[t] = p_arrival->amp + log(1+ alpha/(pow(1 + t_off, gamma))) + p_arrival->coda_decay*t_off;
    if (t - peak_idx < 5) {
      printf("peak d[%ld] = %f, prev d[%ld] = %f, alpha = %f, gamma = %f\n", t, d[t], t-1, d[t-1], alpha, gamma);
    }
    }*/


  double A = exp(p_arrival->amp);
  double P = exp(p_arrival->peak_amp);
  double F = p_arrival->peak_decay;
  double b = p_arrival->coda_decay;
  for (long t=peak_idx; t < p_trace->len; ++t) {
    double t_off = (t - peak_idx)/p_trace->hz;
    d[t] = log( (P-A) * exp( -1 * (t_off * t_off) / (F * F)) + A * exp( b * t_off));
  }

  p_trace->start_time = p_arrival->time;

}


void logsum_envelope_obsfn(const gsl_vector * state, gsl_vector *obs, va_list * args) {

  /* Assume that we are passed two arguments: 
     1) an array of length NUM_CHANS, containing for each channel the index of the state variable corresponding to that channel's noise process
     2) the BandModel corresponding to the current band
     2) a linked list of currently active arriving waveforms
  */
  int * noise_indices = va_arg(*args, int *);
  BandModel_t * p_band = va_arg(*args, BandModel_t *);
  ArrivalWaveform_t * active_arrivals = va_arg(*args, ArrivalWaveform_t *);
 
  int obs_i = 0;
  for(int c=0; c < NUM_CHANS; c++) {

    if (noise_indices[c] < 0) {
      continue;
    }

    // the output of each channel starts with the noise process
    double chan_output = gsl_vector_get(state, noise_indices[c]) + p_band->wiggle_model.mean;

    // then, we add in the signal from each active arrival
    for(ArrivalWaveform_t * aa = active_arrivals; 
	aa != NULL; 
	aa = aa->next_active) {

      // signal is the log-envelope, plus AR wiggle process
      double signal = aa->p_abstract_trace->p_data[aa->idx];
      signal += gsl_vector_get(state, aa->active_id);

      // projected onto the current channel
      signal += log(aa->projection_coeffs[c]);

      // added to signals from other channels
      chan_output = LOGSUM(chan_output, signal);
      aa = aa->next_active;
    }

    gsl_vector_set(obs, obs_i++, chan_output);
  }

}

void setup_noise_processes(BandModel_t * p_band, Segment_t * p_segment, KalmanState_t * k, int * noise_indices) {

/* Set up the per-channel noise models. */
  for(int i=0; i < NUM_CHANS; ++i) {

    /* if we are passed an observed segment, we only generate noise
       processes for the observed channels */
    if ((p_segment != NULL) && p_segment->p_channels[i] == NULL) {
      noise_indices[i] = -1;
      continue;
    }

    /* also, we only generate noise processes for channels where we
       actually have a model */
    ARProcess_t * chan_noise = &p_band->channel_noise_models[i];
    if (chan_noise->coeffs == NULL) {
      LogInfo("no noise model for siteid %d, band ?, channel %d!", p_segment->siteid, i);
      noise_indices[i] = -1;
      continue;
    }
    noise_indices[i] = kalman_add_AR_process(k, chan_noise);
  }
}


/* Update our position in the lists of arriving waveforms to reflect
   the current timestep, and update the Kalman state to reflect the
   currently active waveforms. */
void update_active_events(KalmanState_t * k, double time, ARWLists_t * arw) {

    // activate any events that are starting
    while (arw->st_ptr != NULL && time >= arw->st_ptr->start_time) {
      LogTrace(" activating arrivalwaveform w/ st %lf at time %lf", arw->st_ptr->start_time, time);
      arw->active_arrivals = append_active(arw->active_arrivals, arw->st_ptr);
      arw->st_ptr->active_id = kalman_add_AR_process(k, &arw->st_ptr->ar_process);
      arw->st_ptr = arw->st_ptr->next_start;
    }

    // clean up any events that have finished
    while (arw->et_ptr != NULL && time >= arw->et_ptr->end_time) {
      LogTrace(" deactivating arrivalwaveform w/ et %lf at time %lf", arw->et_ptr->end_time, time);
      arw->active_arrivals = remove_active(arw->active_arrivals, arw->et_ptr);

      // remove the wiggle process from the Kalman state
      kalman_remove_AR_process(k, arw->et_ptr->ar_process.order, arw->et_ptr->active_id);

      // surviving events now move up in the Kalman state
      for (ArrivalWaveform_t * a = arw->active_arrivals; a != NULL; a = a->next_active) {
	if (a->active_id > arw->et_ptr->active_id) {
	  LogTrace(" remove arridx %d, decrementing %d", arw->et_ptr->active_id, a->active_id);
	  a->active_id -= arw->et_ptr->ar_process.order;
	} else {
	  LogTrace(" remove arridx %d, not decrementing %d", arw->et_ptr->active_id, a->active_id);
	}

	if (a->idx < a->len) a->idx++;
      }
      arw->et_ptr = arw->et_ptr->next_end;
    }
}



/* populate two linked lists, storing waveform info sorted by
   start_time and end_time respectively */
void init_ArrivalWaveforms(BandModel_t * p_band, int hz, int num_arrivals, const Arrival_t ** pp_arrivals, ARWLists_t * arw) {
  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);

    if (p_arr->amp == 0 || p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = -1; // initialize to -1 since we will increment before the first use

    // for each arrival, get the predicted log-envelope
    w->p_abstract_trace = calloc(1, sizeof(Trace));
    w->p_abstract_trace->hz = hz;
    abstract_spectral_logenv_raw(p_arr, w->p_abstract_trace);

    w->end_time = w->start_time + (double) w->len / hz;

    copy_AR_process(&w->ar_process, &p_band->wiggle_model);

        double iangle;
    if(!slowness_to_iangle(p_arr->slo, p_arr->phase, &iangle)) {
      //LogTrace("iangle conversion failed from slowness %lf phaseid %d, setting default iangle 45.", p_arr->slo, phase);
      iangle = 45;
    }

    // TODO: FIX COEFFS (shouldn't be normalized to BHZ, and should be distributed by some model...)
    w->projection_coeffs[CHAN_BHE] = fabs(SPHERE2X(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->projection_coeffs[CHAN_BHN] = fabs(SPHERE2Y(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->projection_coeffs[CHAN_BHZ] = 1;

    arw->st_head = insert_st(arw->st_head, w);
    arw->et_head = insert_et(arw->et_head, w);

  }

  arw->st_ptr = arw->st_head;
  arw->et_ptr = arw->et_head;
  arw->active_arrivals = NULL;

}


/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + AR(n)
  wiggles + Gaussian iid noise signal model.
 */
double Spectral_Envelope_Model_Likelihood(SigModel_t * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals) {

  SignalModel_t * p_model = &p_sigmodel->signal_model;
  int siteid = p_segment->siteid;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t * )p_model->pv_params;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;
  int band = DEFAULT_BAND;
  BandModel_t * p_band = p_sta->bands + band;
  double ll = 0;

  /* initialize lists of arriving waveforms, sorted by start time and end time
     respectively */
  ARWLists_t arw;
  init_ArrivalWaveforms(p_band, p_segment->hz, num_arrivals, pp_arrivals, &arw);

  // figure out how many channels we will be observing
  int obs_n=0;
  for(int i=0; i < NUM_CHANS; ++i) if (p_segment->p_channels[i] != NULL)  obs_n++;
  if( obs_n == 0 ) return 0;

  // initialize the Kalman filter with AR noise processes for each channel
  KalmanState_t * k = calloc(1, sizeof(KalmanState_t));
  kalman_state_init(k, obs_n, FALSE, NULL, logsum_envelope_obsfn);
  int noise_indices[NUM_CHANS];
  setup_noise_processes(p_band, p_segment, k, noise_indices);

  gsl_vector * p_true_obs = gsl_vector_alloc(obs_n);
  /* MAIN LOOP:
   *  Loop through each timestep within the segment.
   *  Keep track of which arrivals are active at each point in time.
   *  Pass these to the KalmanState to compute a filtering distribution, thus likelihood.
   */
  for (int t = 0; t < p_segment->len; ++t) {
    double time = p_segment->start_time + t/p_segment->hz;

    /* update the set of events active at this timestep (and the
       corresponding Kalman filter state). */
    update_active_events(k, time, &arw);

    // construct the observation vector
    int obs_i = 0;
    for (int c = 0; c < NUM_CHANS; ++c) {
      double obs  = p_segment->p_channels[c]->p_bands[band]->p_data[t];
      gsl_vector_set(p_true_obs, obs_i++, obs);
    }

    /* update the state with the new observation, and return the
       log-likelihood of the observation */
    kalman_predict(k);
    ll -= kalman_nonlinear_update(k, p_true_obs, noise_indices, arw.active_arrivals);
  }

  /* Free memory before returning */
  for(ArrivalWaveform_t * a = arw.st_head; a != NULL; ) {
    a = free_ArrivalWaveform(a);
  }
  kalman_state_free(k);
  gsl_vector_free(p_true_obs);

  LogTrace ("returning ar ll %lf", ll);

  return ll;
}


void Spectral_Envelope_Model_Sample(SigModel_t * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int sample_noise, int sample_wiggles) {

  SignalModel_t * p_model = &p_sigmodel->signal_model;
  int siteid = p_segment->siteid;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t * )p_model->pv_params;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;
  int band = DEFAULT_BAND;
  BandModel_t * p_band = p_sta->bands + band;

  /*make sure segment has all necessary properties defined (really just hz) and arrays allocated */
  if (p_segment->hz == 0) p_segment->hz = DEFAULT_HZ;
  // what to do here will depend on where the segments are coming from...

  /* initialize lists of arriving waveforms, sorted by start time and end time
     respectively */
  ARWLists_t arw;
  init_ArrivalWaveforms(p_band, p_segment->hz, num_arrivals, pp_arrivals, &arw);

  int obs_n=NUM_CHANS;

  // initialize the Kalman filter with AR noise processes for each channel
  KalmanState_t * k = calloc(1, sizeof(KalmanState_t));
  kalman_state_init(k, obs_n, FALSE, NULL, logsum_envelope_obsfn);
  int noise_indices[NUM_CHANS];
  setup_noise_processes(p_band, NULL, k, noise_indices);
  if (!sample_noise) {
    for (int c = 0; c < NUM_CHANS; ++c) {
      gsl_vector_set(k->p_process_noise, noise_indices[c], 0);
    }
  }

  gsl_vector * p_sample_obs = gsl_vector_alloc(obs_n);
  /* MAIN LOOP:
   *  Loop through each timestep within the segment.
   *  Keep track of which arrivals are active at each point in time.
   *  Pass these to the KalmanState to compute a filtering distribution, thus likelihood.
   */
  for (int t = 0; t < p_segment->len; ++t) {
    double time = p_segment->start_time + t/p_segment->hz;

    /* update the set of events active at this timestep (and the
       corresponding Kalman filter state). */
    update_active_events(k, time, &arw);

    /* if we're not sampling wiggles, then set all process noise vars
       to zero except those corresponding to the channel noise models
       (controlled by "sample_noise" above) */ 
    for(int i=0; (!sample_wiggles) && i < k->n; ++i) {
      if (gsl_vector_get(k->p_process_noise, i) != 0) {
	for (int c = 0; c < NUM_CHANS; ++c) if (i == noise_indices[c]) continue;
	gsl_vector_set(k->p_process_noise, i, 0);
      }
    }

    /* update the state with the new observation, and return the
       log-likelihood of the observation */
    kalman_sample_forward(k, p_sample_obs, noise_indices, arw.active_arrivals);

    /* save the sampled observation */
    for (int c=0; c < NUM_CHANS; ++c) {
      p_segment->p_channels[c]->p_bands[band]->p_data[t] = 
	gsl_vector_get(p_sample_obs, c);
    }
  }

  /* Free memory before returning */
  for(ArrivalWaveform_t * a = arw.st_head; a != NULL; ) {
    a = free_ArrivalWaveform(a);
  }
  kalman_state_free(k);
  gsl_vector_free(p_sample_obs);

}

void Spectral_Envelope_Model_UnInit(Spectral_Envelope_Model_t * p_params) {

  double * coeffs;
  for (int i=0; i < p_params->numsites; ++i) {
    for (int j=0; j < NUM_BANDS; ++j) {
      for (int k=0; k < NUM_CHANS; ++k) {
	coeffs = (p_params->p_stations + i)->bands[j].channel_noise_models[k].coeffs;
	if (coeffs != NULL) {
	  free(coeffs);
	}
      }
      coeffs = (p_params->p_stations + i)->bands[j].wiggle_model.coeffs;
      if (coeffs != NULL) {
	free(coeffs);
      }
  }
  free(p_params->p_stations);
  }
}
