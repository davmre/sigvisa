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

#include "../../sigvisa.h"
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

int Spectral_Envelope_Model_Has_Model(SigModel_t * p_sigmodel, int siteid, int chan, int band) {
  SignalModel_t * p_model = &p_sigmodel->signal_model;
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *)p_model->pv_params;

  /* chan < 0 means return true if we have a model for *any* channel
     at this station. similarly band < 0 means return true if we have
     any band models at this station. */

  int chan_model = FALSE;
  int wiggle_model = FALSE;

  for (int i=0; i < NUM_CHANS; ++i) {
    for (int j=0; j < NUM_BANDS; ++j) {

      if (chan >= 0 && chan != i) continue;
      if (band >= 0 && chan != j) continue;

      if (p_params->p_stations[siteid-1].bands[j].channel_noise_models[i].coeffs != NULL) {
	chan_model = TRUE;
      }

      for(int p=0; p < NUM_TD_PHASES; ++p) {
	if (p_params->p_stations[siteid-1].bands[j].wiggle_model[i][p].coeffs != NULL) {
	  wiggle_model = TRUE;
	}
      }

      if (chan_model && wiggle_model) return TRUE;
    }
  }
  return FALSE;

}


/*
   Input: an arrival specifying envelope parameters, and a trace object specifying the sampling rate.
   Returns: sets p_data, len, and start_time in the Trace.
*/
void abstract_spectral_logenv_raw(const Arrival_t * p_arrival, Trace_t * p_trace) {

  // calculate length of envelope to generate
  p_trace->len = (long)((MIN_LOGENV_CUTOFF- p_arrival->amp) / p_arrival->coda_decay * p_trace->hz);

  if (p_trace->len > 800*(p_trace->hz) || p_trace->len < 0) {
    LogTrace("WARNING: truncating arrival length from %d to %d, (length based on amp %f, coda_decay %f, hz %f)", p_trace->len, (int) ((500)*(p_trace->hz)), p_arrival->amp, p_arrival->coda_decay, (p_trace->hz));
    p_trace->len = 500*p_trace->hz;
  }

  // allocate memory
  p_trace->p_data = (double *) calloc(p_trace->len, sizeof(double));
  double * d = p_trace->p_data;
  if (d == NULL) {
    printf("error allocating memory for means in abstract_env, len %ld from amp %f, decay %f, hz %f\n", p_trace->len, p_arrival->amp, p_arrival->coda_decay, p_trace->hz);
    exit(-1);
  }

  // generate onset
  long peak_idx = (p_arrival->peak_time - p_arrival->time) * p_trace->hz;
  double onset_slope;
  if (peak_idx != 0) {
    onset_slope = (p_arrival->amp - MIN_LOGENV_CUTOFF) / peak_idx;
  }
  for (long t=0; t < peak_idx && t < p_trace->len; ++t) {
    d[t] = MIN_LOGENV_CUTOFF + t * onset_slope;
  }

  double b = p_arrival->coda_decay;
  for (long t=MAX(peak_idx, 0); t < p_trace->len; ++t) {
    double t_off = (t - peak_idx)/p_trace->hz;
    d[t] = p_arrival->amp + b * t_off;
    //    printf("trace %ld = %f * exp(%lf * %lf) = %lf\n", t, A, b, t_off, d[t]);
  }

  p_trace->start_time = p_arrival->time;

}

void sum_envelope_obsfn(const gsl_vector *state, gsl_vector *obs, void *void_k, va_list * args) {

  KalmanState_t *k = (KalmanState_t *)void_k;

  /* Assume that we are passed three arguments:
     1) an array of length NUM_CHANS, containing for each channel the index of the state variable corresponding to that channel's noise process
     2) the BandModel corresponding to the current band
     2) a linked list of currently active arriving waveforms
  */
  int * noise_indices = va_arg(*args, int *);
  BandModel_t * p_band = va_arg(*args, BandModel_t *);
  ArrivalWaveform_t * active_arrivals = va_arg(*args, ArrivalWaveform_t *);

  int obs_i = 0;
  for(int c=0; c < NUM_CHANS; c++) {

    // an index of -1 corresponds to a channel that is not being output, so we skip it
    if (noise_indices[c] == -1 ) continue;

    /* but an index of -2 corresponds to a channel that is being
       output, but just doesn't have a model defined (this happens
       during sampling since we always output a full segment
       containing every channel). So here we don't output anything,
       but increment the output counter. */
    if (noise_indices[c] == -2 ) {
      obs_i++;
      continue;
    }

    // the output of each channel starts with the noise process
    int id = gsl_vector_get(k->p_permanent_indices, noise_indices[c]);
    double chan_output = gsl_vector_get(state, id);

    // then, we add in the signal from each active arrival
    for(ArrivalWaveform_t * aa = active_arrivals;
	aa != NULL;
	aa = aa->next_active) {

      // signal is the log-envelope, plus AR wiggle process
      double signal = exp(aa->p_abstract_trace->p_data[aa->idx]);
      id = gsl_vector_get(k->p_permanent_indices, aa->wiggle_ids[c]);
      signal *= gsl_vector_get(state, id);

      // projected onto the current channel
      signal *= aa->projection_coeffs[c];

      // added to signals from other channels
      chan_output += signal;
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
      LogTrace("no noise model for siteid %d, band ?, channel %d!", p_segment->siteid, i);
      noise_indices[i] = -2;
      continue;
    }
    noise_indices[i] = kalman_add_AR_process(k, chan_noise);
  }

  if (k->np == 0) {
    LogError("No noise processes added - are you sure you want this?");
    exit(EXIT_FAILURE);
  }
}


void setup_noise_process(BandModel_t * p_band, int chan, KalmanState_t * k, int * noise_indices) {

/* Set up the per-channel noise models. */
  for(int i=0; i < NUM_CHANS; ++i) {

    /* if we are passed an observed segment, we only generate noise
       processes for the observed channels */
    if (i != chan) {
      noise_indices[i] = -1;
      continue;
    }

    /* also, we only generate noise processes for channels where we
       actually have a model */
    ARProcess_t * chan_noise = &p_band->channel_noise_models[i];
    if (chan_noise->coeffs == NULL) {
      LogError("no noise model for channel %d!", i);
      exit(1);
    }
    noise_indices[i] = kalman_add_AR_process(k, chan_noise);
  }

  if (k->np == 0) {
    LogError("No noise processes added - are you sure you want this?");
    exit(EXIT_FAILURE);
  }
}

/* Update our position in the lists of arriving waveforms to reflect
   the current timestep, and update the Kalman state to reflect the
   currently active waveforms. */
int update_active_events(KalmanState_t * k, double time, ARWLists_t * arw, int * chan_indices, int obs_n) {

  int changed = 0;

    // activate any events that are starting
  while (arw->st_ptr != NULL && time >= arw->st_ptr->start_time) {
      LogTrace(" activating arrivalwaveform w/ st %lf at time %lf", arw->st_ptr->start_time, time);
      changed = 1;
      arw->active_arrivals = append_active(arw->active_arrivals, arw->st_ptr);

      for(int ci=0; ci < obs_n; ++ci) {
	int c = chan_indices[ci];
	arw->st_ptr->wiggle_ids[c] = kalman_add_AR_process(k, &(arw->st_ptr->ar_processes[c]));
      }
      arw->st_ptr = arw->st_ptr->next_start;
    }

    // clean up any events that have finished
    while (arw->et_ptr != NULL && time >= arw->et_ptr->end_time) {
      LogTrace(" deactivating arrivalwaveform w/ et %lf at time %lf", arw->et_ptr->end_time, time);
      arw->active_arrivals = remove_active(arw->active_arrivals, arw->et_ptr);

      // remove the wiggle process from the Kalman state
      for(int ci=0; ci < obs_n; ++ci) {
	int c = chan_indices[ci];
	kalman_remove_AR_process(k, arw->et_ptr->ar_processes[c].order, arw->et_ptr->wiggle_ids[c]);
      }
      arw->et_ptr = arw->et_ptr->next_end;

    }

    // advance all active arrivals
    for (ArrivalWaveform_t * a = arw->active_arrivals; a != NULL; a = a->next_active){
      if (a->idx < a->len) a->idx++;
    }

    return changed;
}



/* populate two linked lists, storing waveform info sorted by
   start_time and end_time respectively */
void init_ArrivalWaveforms(BandModel_t * p_band, double hz, int num_arrivals, const Arrival_t ** pp_arrivals, ARWLists_t * arw) {

  arw->st_head = NULL;
  arw->et_head = NULL;

  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);

    if (p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = -1; // initialize to -1 since we will increment before the first use

    // for each arrival, get the predicted log-envelope
    w->p_abstract_trace = calloc(1, sizeof(Trace_t));
    w->p_abstract_trace->hz = hz;
    abstract_spectral_logenv_raw(p_arr, w->p_abstract_trace);
    w->len = w->p_abstract_trace->len;

    w->end_time = w->start_time + (double) w->len / hz;

    for(int c = 0; c < NUM_CHANS; ++c) {
      copy_AR_process(&w->ar_processes[c], &p_band->wiggle_model[c][p_arr->phaseid-1]);
    }

        double iangle;
    if(!slowness_to_iangle(p_arr->slo, p_arr->phaseid-1, &iangle)) {
      //LogTrace("iangle conversion failed from slowness %lf phaseid %d, setting default iangle 45.", p_arr->slo, phase);
      iangle = 45;
    }

    // TODO: FIX COEFFS (shouldn't be normalized to BHZ, and should be distributed by some model...)
    w->projection_coeffs[CHAN_BHE] = 1;
      //fabs(SPHERE2X(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->projection_coeffs[CHAN_BHN] = 1;
      //fabs(SPHERE2Y(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->projection_coeffs[CHAN_BHZ] = 1;
    w->projection_coeffs[CHAN_HORIZ_AVG] = 1;

    arw->st_head = (arw->st_head == NULL) ? w : insert_st(arw->st_head, w);
    arw->et_head = (arw->et_head == NULL) ? w : insert_et(arw->et_head, w);
  }

  arw->st_ptr = arw->st_head;
  arw->et_ptr = arw->et_head;
  arw->active_arrivals = NULL;

}


double Envelope_Model_Likelihood(SigModel_t * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int band) {
  SignalModel_t * p_model = &p_sigmodel->signal_model;
  int siteid = p_segment->siteid;
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t * )p_model->pv_params;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;

  BandModel_t * p_band = p_sta->bands + band;
  double ll = 0;

  /* initialize lists of arriving waveforms, sorted by start time and end time
     respectively */
  ARWLists_t arw;

  init_ArrivalWaveforms(p_band, p_segment->hz, num_arrivals, pp_arrivals, &arw);

  // figure out how many channels we will be observing
  int obs_n=0;
  int chan_indices[NUM_CHANS];
  for(int i=0; i < NUM_CHANS; ++i) if (p_segment->p_channels[i] != NULL) chan_indices[obs_n++] = i;
  if( obs_n == 0 ) return 0;

  // initialize the Kalman filter with AR noise processes for each channel
  KalmanState_t * k = calloc(1, sizeof(KalmanState_t));
  kalman_state_init(k, obs_n, sum_envelope_obsfn, (obs_n == 1) ? 0 : 0.0001, NULL);
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
    k->verbose = update_active_events(k, time, &arw, chan_indices, obs_n);

    // construct the observation vector
    int obs_i = 0;
    for (int c = 0; c < obs_n; ++c) {
      double obs  = p_segment->p_channels[chan_indices[c]]->p_bands[band]->p_data[t];
      gsl_vector_set(p_true_obs, obs_i++, obs);
    }

    /* update the state with the new observation, and return the
       log-likelihood of the observation */
    kalman_predict(k);

    // ll += kalman_nonlinear_update(k, p_true_obs, noise_indices, p_band, arw.active_arrivals);
    ll += kalman_linear_update(k, p_true_obs, noise_indices, arw.active_arrivals);
  }

  /* Free memory before returning */
  for(ArrivalWaveform_t * a = arw.st_head; a != NULL; a = free_ArrivalWaveform(a));
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
double Spectral_Envelope_Model_Likelihood(SigModel_t * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals) {

  SignalModel_t * p_model = &p_sigmodel->signal_model;
  int siteid = p_segment->siteid;
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t * )p_model->pv_params;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;

  double ll = 0;
  int bands_considered = 0;
  for(int band=0; band < NUM_BANDS; ++band) {

    if (!segment_contains_band(p_segment, band)) {
      continue;
    }

    BandModel_t * p_band = p_sta->bands + band;
    if (p_band == NULL) {
      LogError("segment contains data for band '%s', but no model is loaded (siteid %d)!", canonical_band_name(band), siteid);
      exit(1);
    }

    bands_considered++;
    ll += Envelope_Model_Likelihood(p_sigmodel, p_segment, num_arrivals, pp_arrivals, band);
  }

  if (bands_considered==0) {
    LogError("No valid bands found in segment, can't compute likelihood!");
    return NAN;
  }

  return ll;
}


void Spectral_Envelope_Model_Sample_Trace(SigModel_t * p_sigmodel, Trace_t * p_trace, int num_arrivals, const Arrival_t ** pp_arrivals, int sample_noise, int sample_wiggles) {

  SignalModel_t * p_model = &p_sigmodel->signal_model;
  int siteid = p_trace->siteid;

  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t * )p_model->pv_params;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;
  int band = p_trace->band;
  BandModel_t * p_band = p_sta->bands + band;

  if (p_trace->hz == 0) p_trace->hz = DEFAULT_HZ;

  /* initialize lists of arriving waveforms, sorted by start time and end time
     respectively */
  ARWLists_t arw;
  init_ArrivalWaveforms(p_band, p_trace->hz, num_arrivals, pp_arrivals, &arw);

  int obs_n=1;
  int chan_indices[1];
  chan_indices[0] = p_trace->chan;

  // initialize the Kalman filter with AR noise processes for each channel
  KalmanState_t * k = calloc(1, sizeof(KalmanState_t));
  kalman_state_init(k, obs_n, sum_envelope_obsfn, 0, NULL);
  int noise_indices[NUM_CHANS];
  setup_noise_process(p_band, p_trace->chan, k, noise_indices);

  if (!sample_noise) {
    for (int c = 0; c < NUM_CHANS; ++c) {
      if (noise_indices[c] >= 0) {
	gsl_vector_set(k->p_process_noise, noise_indices[c], 0);
      }
    }
  }

  gsl_vector * p_sample_obs = gsl_vector_alloc(obs_n);
  /* MAIN LOOP:
   *  Loop through each timestep within the segment.
   *  Keep track of which arrivals are active at each point in time.
   *  Pass these to the KalmanState to compute a filtering distribution, thus likelihood.
   */
  for (int t = 0; t < p_trace->len; ++t) {
    double time = p_trace->start_time + t/p_trace->hz;

    /* update the set of events active at this timestep (and the
       corresponding Kalman filter state). */
    update_active_events(k, time, &arw, chan_indices, obs_n);

    /* if we're not sampling wiggles, then set all process noise vars
       to zero except those corresponding to the channel noise models
       (controlled by "sample_noise" above) */
    for(int i=0; (!sample_wiggles) && i < k->n; ++i) {
      if (gsl_vector_get(k->p_process_noise, i) != 0) {
	int is_noise=FALSE;
	for (int c = 0; c < NUM_CHANS; ++c) {if (i == noise_indices[c]) {is_noise = TRUE; break;} }
	if (!is_noise) gsl_vector_set(k->p_process_noise, i, 0);
      }
    }

    /* update the state with the new observation, and return the
       log-likelihood of the observation */
    kalman_sample_forward(k, p_sample_obs, noise_indices, p_band, arw.active_arrivals);

    /* save the sampled observation */
    for (int c=0; c < obs_n; ++c) {
      p_trace->p_data[t] =
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

void Spectral_Envelope_Model_Sample(SigModel_t * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int * bands_to_sample, int sample_noise, int sample_wiggles) {

  /*
    bands_to_sample is a boolean array of length NUM_BANDS.
   */

  for(int chan=0; chan < NUM_CHANS; ++chan) {
    Channel_t * p_chan = p_segment->p_channels[chan];
    if (p_chan != NULL) {
      for (int band=0; band < NUM_BANDS; ++band) {
	if (bands_to_sample[band]) {
	  Trace_t * p_trace;
	  p_trace = p_chan->p_bands[band];
	  assert(p_trace != NULL);
	  Spectral_Envelope_Model_Sample_Trace(p_sigmodel, p_trace, num_arrivals, pp_arrivals, sample_noise, sample_wiggles);
	}
      }
    }
  }
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

	for(int p=0; p < NUM_TD_PHASES; ++p) {
	  coeffs = (p_params->p_stations + i)->bands[j].wiggle_model[k][p].coeffs;
	  if (coeffs != NULL) {
	    free(coeffs);
	  }
	}
      }
    }
  }
  free(p_params->p_stations);
}
