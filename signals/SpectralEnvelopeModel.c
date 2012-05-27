#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../sigvisa.h"
#include "SignalModelUtil.h"

void Spectral_Envelope_Model_Set_Params(void * pv_params, int siteid, PyObject * py_dict) {
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *) pv_params;


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
	sta->override_b = PyFloat_AsDouble(py_value);
      }  else if (strcmp(key, "override_gamma") == 0) {
	sta->override_gamma = PyFloat_AsDouble(py_value);
      } else if (strcmp(key, "override_height") == 0) {
	sta->override_height = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "p_b0_", 5) == 0) {
	int band = canonical_band_num(key+5);
	sta->p_b0[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "p_b1_", 5) == 0) {
	int band = canonical_band_num(key+5);
	sta->p_b1[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "p_b2_", 5) == 0) {
	int band = canonical_band_num(key+5);
	sta->p_b2[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "s_b0_", 5) == 0) {
	int band = canonical_band_num(key+5);
	sta->s_b0[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "s_b1_", 5) == 0) {
	int band = canonical_band_num(key+5);
	sta->s_b1[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "s_b2_", 5) == 0) {
	int band = canonical_band_num(key+5);
	sta->s_b2[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "p_gamma0_", 9) == 0) {
	int band = canonical_band_num(key+9);
	sta->p_gamma0[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "p_gamma1_", 9) == 0) {
	int band = canonical_band_num(key+9);
	sta->p_gamma1[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "p_gamma2_", 9) == 0) {
	int band = canonical_band_num(key+9);
	sta->p_gamma2[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "s_gamma0_", 9) == 0) {
	int band = canonical_band_num(key+9);
	sta->s_gamma0[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "s_gamma1_", 9) == 0) {
	int band = canonical_band_num(key+9);
	sta->s_gamma1[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "s_gamma2_", 9) == 0) {
	int band = canonical_band_num(key+9);
	sta->s_gamma2[band] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "chan_mean_", 10) == 0) {
	int chan_num = canonical_channel_num(key+10);
	sta->chan_means[chan_num] = PyFloat_AsDouble(py_value);
      } else if (strncmp(key, "chan_var_", 9) == 0) {
	int chan_num = canonical_channel_num(key+9);
	sta->chan_vars[chan_num] = PyFloat_AsDouble(py_value);

      } else if (strcmp(key, "ar_noise_sigma2") == 0) {
	double val = PyFloat_AsDouble(py_value);

	if (val > 0.0000001) {
	  sta->ar_noise_sigma2 = val;
	} else {
	  LogInfo("ignoring invalid param setting ar_noise_sigma2 = %lf", val);
	}

      } else if (strcmp(key, "ar_coeffs") == 0) {

	int converted_tuple = 0;
	if (PyList_Check(py_value)) {
	  converted_tuple = 1;
	  py_value = PyList_AsTuple(py_value); CHECK_ERROR;
	}

	if (!PyTuple_Check(py_value)) {
	  LogFatal("expected Python tuple for ar_coeffs!\n");
	  exit(EXIT_FAILURE);
	}
	CHECK_ERROR;

	sta->ar_n = PyTuple_Size(py_value); CHECK_ERROR;
	LogTrace("setting %s to tuple of length %d at siteid %d", key, sta->ar_n, siteid);
	if (sta->p_ar_coeffs != NULL) free(sta->p_ar_coeffs);
	sta->p_ar_coeffs = calloc(sta->ar_n, sizeof(double));
	for (int i=0; i < sta->ar_n; ++i) {
	  sta->p_ar_coeffs[i] = PyFloat_AsDouble(PyTuple_GetItem(py_value, i)); CHECK_ERROR;
	}

	if (converted_tuple) {
	  Py_DECREF(py_value);
	}

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

void Spectral_Envelope_Model_Init_Params(void * pv_params,  int numsites) {
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *) pv_params;

  p_params->numsites = numsites;

  /* using calloc (instead of malloc) is important here since we
     depend on the p_ar_coeffs pointer being NULL iff unallocated */
  p_params->p_stations = calloc(numsites, sizeof(Spectral_StationModel_t));
}

int Spectral_Envelope_Model_Has_Model(void * pv_sigmodel, int siteid, int chan) {

  SigModel_t * p_sigmodel = (SigModel_t *) pv_sigmodel;
  SignalModel_t * p_model = &p_sigmodel->signal_model;
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *)p_model->pv_params;

  /* chan < 0 means return true if we have a model for any channel at
     this station */
  if (chan < 0) {
    for (int i=0; i < NUM_CHANS; ++i) {
      if (p_params->p_stations[siteid-1].chan_vars[i] > 0) {
	return 1;
      }
    }
    return 0;
  }

  else {
    // LogTrace("checking var %lf for site %d chan %d", p_params->p_stations[siteid-1].chan_vars[chan], siteid, chan);
    // LogTrace("p_model %p sta %p", p_params, p_params->p_stations + siteid-1);
    return (p_params->p_stations[siteid-1].chan_vars[chan] > 0);
  }

}


/*
   Input: an arrival specifying envelope parameters, and a trace object specifying the sampling rate.
   Returns: sets p_data, len, and start_time in the Trace.
*/
void abstract_spectral_logenv_raw(Arrival_t * p_arrival, Trace_t * p_trace) {

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


void * logsum_envelope_obsfn(const gsl_vector * state, gsl_vector *obs, va_list * args) {

  /* Assume that we are passed two arguments: 
     1) an array of length NUM_CHANS, containing for each channel the index of the state variable corresponding to that channel's noise process
     2) the BandModel corresponding to the current band
     2) a linked list of currently active arriving waveforms
  */
  int noise_indices[] = va_arg(*args, int[]);
  BandModel_t * p_band = va_arg(*args, (BandModel_t *));
  ArrivalWaveform * active_arrivals = va_arg(*args, (ArrivalWaveform *));
 
  int obs_i = 0;
  for(int c=0; c < NUM_CHANS; i++) {

    if (noise_indices[c] < 0) {
      continue;
    }

    // the output of each channel starts with the noise process
    double chan_output = gsl_vector_get(state, noise_indices[c]) + p_band.wiggle_model.mean;

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

void setup_noise_processes(BandModel_t * p_band, Segment_t * p_segment, KalmanState * k, int * noise_indices) {

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
    ARProcess * chan_noise = &p_band.channel_noise_models[i];
    if (chan_noise->params == NULL) {
      LogInfo("no noise model for siteid %d, band %d, channel %d!", siteid, band, i);
      noise_indices[i] = -1;
      continue;
    }
    noise_indices[i] = kalman_add_AR_process(k, chan_noise);
  }
}


/* Update our position in the lists of arriving waveforms to reflect
   the current timestep, and update the Kalman state to reflect the
   currently active waveforms. */
void update_active_events(KalmanState * k, double time, ARWLists_t * arw) {

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

/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + AR(n)
  wiggles + Gaussian iid noise signal model.
 */
double Spectral_Envelope_Model_Likelihood(void * pv_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals) {

  SigModel_t * p_sigmodel = (SigModel_t *) pv_sigmodel;
  SignalModel_t * p_model = &p_sigmodel->signal_model;
  int siteid = p_segment->siteid;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t * )p_model->pv_params;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;
  int band = BB_ENVELOPE;
  BandModel_t * p_band = p_sta.bands + band;
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
  KalmanState * k = calloc(1, sizeof(KalmanState));
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
    ll -= kalman_nonlinear_update(k, p_true_obs, noise_indices, active_arrivals);
  }

  /* Free memory before returning */
  for(ArrivalWaveform_t * a = arw.st_arrivals; a != NULL; ) {
    a = free_ArrivalWaveform(a);
  }
  kalman_state_free(k);
  gsl_vector_free(p_true_obs);

  LogTrace ("returning ar ll %lf", ll);

  return ll;
}

/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + AR(n)
  wiggles + Gaussian iid noise signal model.
 */
double Spectral_Envelope_Model_Sample(void * pv_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int sample_noise, int sample_wiggles) {

  SigModel_t * p_sigmodel = (SigModel_t *) pv_sigmodel;
  SignalModel_t * p_model = &p_sigmodel->signal_model;
  int siteid = p_segment->siteid;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t * )p_model->pv_params;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;
  int band = BB_ENVELOPE;
  BandModel_t * p_band = p_sta.bands + band;

  /*make sure segment has all necessary properties defined (really just hz) and arrays allocated */
  if (p_segment->hz == 0) p_segment->hz = DEFAULT_HZ;
  // what to do here will depend on where the segments are coming from...

  /* initialize lists of arriving waveforms, sorted by start time and end time
     respectively */
  ARWLists_t arw;
  init_ArrivalWaveforms(p_band, p_segment->hz, num_arrivals, pp_arrivals, &arw);

  int obs_n=NUM_CHANS;

  // initialize the Kalman filter with AR noise processes for each channel
  KalmanState * k = calloc(1, sizeof(KalmanState));
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
    void kalman_sample_forward(k, p_sample_obs, noise_indices, active_arrivals);

    /* save the sampled observation */
    for (int c=0; c < NUM_CHANS; ++c) {
      p_segment->p_channels[c]->p_bands[band]->p_data[t] = 
	gsl_vector_get(p_sample_obs, c);
    }
  }

  /* Free memory before returning */
  for(ArrivalWaveform_t * a = arw.st_arrivals; a != NULL; ) {
    a = free_ArrivalWaveform(a);
  }
  kalman_state_free(k);
  gsl_vector_free(p_true_obs);

}

void Spectral_Envelope_Model_UnInit(void * pv_params) {
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *) pv_params;

  for (int i=0; i < p_params->numsites; ++i) {
    free((p_params->p_stations + i)->p_ar_coeffs);
  }
  free(p_params->p_stations);

}
