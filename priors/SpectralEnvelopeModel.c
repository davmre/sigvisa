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
	  py_value = PyList_AsTuple(py_value);
	}

	if (!PyTuple_Check(py_value)) {
	  LogFatal("expected Python tuple for ar_coeffs!\n");
	  exit(EXIT_FAILURE);
	}

	sta->ar_n = PyTuple_Size(py_value); CHECK_ERROR;
	LogTrace("setting %s to tuple of length %d at siteid %d", key, sta->ar_n, siteid);      
	if (sta->p_ar_coeffs != NULL) free(sta->p_ar_coeffs);
	sta->p_ar_coeffs = calloc(sta->ar_n, sizeof(double));
	for (int i=0; i < sta->ar_n; ++i) {
	  sta->p_ar_coeffs[i] = PyFloat_AsDouble(PyTuple_GetItem(py_value, i));
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




void abstract_spectral_env_raw(double gamma, double b, double height, double hz, double ** pp_envelope, long *len) {
  
  *len = 200 * hz;
  double * means = (double *) calloc(*len, sizeof(double));

  if (means == NULL) {
    printf("error allocating memory for means in abstract_env, len %ld\n", *len);
    exit(-1);
  }

  for (long t=0; t < *len; ++t) {
    means[t] = height * pow(t/hz, -1 * gamma) * exp(b * t/hz);
    // LogInfo("%d %d %d %lf %d %lf %lf", i, peak_idx, *len, env_decay, end_idx, hz, means[i]);
  }
  
  *pp_envelope = means;
}

void abstract_spectral_env(Spectral_StationModel_t * p_sta, int band, const Arrival_t * p_arr, double hz, double ** pp_envelope, long *len) {
  
  double b, gamma, height; 

  assert(p_arr->dist != 0);

  if (IS_P_PHASE(p_arr->phase)) {
    b = p_sta->p_b0[band] - p_sta->p_b1[band] / (p_sta->p_b2[band] + p_arr->dist);
    gamma = p_sta->p_gamma0[band] - p_sta->p_gamma1[band] / (p_sta->p_gamma2[band] + p_arr->dist);
  } else {    
    b = p_sta->s_b0[band] - p_sta->s_b1[band] / (p_sta->s_b2[band] + p_arr->dist);
    gamma = p_sta->s_gamma0[band] - p_sta->s_gamma1[band] / (p_sta->s_gamma2[band] + p_arr->dist);
  } 

  height = p_arr->amp;

  abstract_spectral_env_raw(b, gamma, height, hz, pp_envelope, len);

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


  double ll = 0;
  double iidll = 0;

  int n = p_sta->ar_n;

  ArrivalWaveform_t * st_arrivals = NULL;
  ArrivalWaveform_t * et_arrivals = NULL;

  /*
  Channel_t filtered_waveform;
  filtered_waveform.p_data = calloc(p_segment->len, sizeof(double));
  filtered_waveform.len = p_segment->len;
  filtered_waveform.start_time = p_segment->start_time;
  filtered_waveform.hz = p_segment->hz;
  filtered_waveform.siteid = p_segment->siteid;
  filtered_waveform.py_array=NULL;
  filtered_waveform.chan=CHAN_BHZ;

  Channel_t filtered_perturbs;
  filtered_perturbs.p_data = calloc(p_segment->len, sizeof(double));
  filtered_perturbs.len = p_segment->len;
  filtered_perturbs.start_time = p_segment->start_time;
  filtered_perturbs.hz = p_segment->hz;
  filtered_perturbs.siteid = p_segment->siteid;
  filtered_perturbs.py_array=NULL;
  filtered_perturbs.chan=CHAN_BHZ;

  Channel_t noise;
  noise.p_data = calloc(p_segment->len, sizeof(double));
  noise.len = p_segment->len;
  noise.start_time = p_segment->start_time;
  noise.hz = p_segment->hz;
  noise.siteid = p_segment->siteid;
  noise.py_array=NULL;
  noise.chan=CHAN_BHZ;*/

  // TODO: figure this out
  int band = BB_ENVELOPE;


  /* populate two linked lists, storing waveform info sorted by
     start_time and end_time respectively */  
  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);
      
    if (p_arr->amp == 0 || p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = 0;

    if (p_sta->override_b != 0) {
      abstract_spectral_env_raw(p_sta->override_b, p_sta->override_gamma, p_sta->override_height, p_segment->hz, &w->p_envelope, &w->len);
    } else {
      abstract_spectral_env(p_sta, band, p_arr, p_segment->hz, &w->p_envelope, &w->len);
    }
    w->end_time = w->start_time + (double) w->len / p_segment->hz;

    double iangle;
    if(!slowness_to_iangle(p_arr->slo, p_arr->phase, &iangle)) {
      //LogTrace("iangle conversion failed from slowness %lf phaseid %d, setting default iangle 45.", p_arr->slo, phase);
      iangle = 45;
    }
    
    w->bhe_coeff = fabs(SPHERE2X(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->bhn_coeff = fabs(SPHERE2Y(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->bhz_coeff = 1;

    st_arrivals = insert_st(st_arrivals, w);
    et_arrivals = insert_et(et_arrivals, w);

  }


  gsl_vector * p_means = NULL;
  gsl_matrix * p_covars = NULL;
  gsl_matrix * p_transition = NULL;
  gsl_matrix * p_observation = NULL;

  ArrivalWaveform_t * starting_next = st_arrivals;
  ArrivalWaveform_t * ending_next = et_arrivals;
  ArrivalWaveform_t * active_arrivals = NULL;
  int n_active = 0;

  // k = number of channels we will be observing
  int k=0;
  for(int i=0; i < NUM_CHANS; ++i) {
    if (p_segment->p_channels[i] != NULL) {
      k++;
    }
  }
  if( k == 0 ) {
    return 0;
  }

  gsl_vector * obs_perturb = gsl_vector_alloc(k);
  gsl_matrix * obs_covar = gsl_matrix_alloc(k,k);
  gsl_matrix_set_identity(obs_covar);
  int curr_chan=0;
  for(int i=0; i < NUM_CHANS; ++i) {
    if (p_segment->p_channels[i] != NULL) {
      if (p_sta->chan_vars[i] <= 0) {
	LogError("invalid variance %lf for siteid %d, channel %d!", p_sta->chan_vars[i], siteid, i);
	exit(EXIT_FAILURE);
      }

      gsl_matrix_set(obs_covar, curr_chan, curr_chan, p_sta->chan_vars[i]);
      curr_chan++;
    }

  }
  gsl_vector * residuals = gsl_vector_alloc(k);
  gsl_vector * residuals_tmp = gsl_vector_alloc(k);
  gsl_matrix * residual_covars = gsl_matrix_alloc(k,k);
  gsl_matrix * residual_covars_inv = gsl_matrix_alloc(k,k);

  for (int t = 0; t < p_segment->len; ++t) {
    double time = p_segment->start_time + t/p_segment->hz;

    // activate the next event, if needed
    while (starting_next != NULL && time >= starting_next->start_time) {
      LogTrace(" activating arrivalwaveform w/ st %lf at time %lf t %d", starting_next->start_time, time, t);
      active_arrivals = append_active(active_arrivals, starting_next);
      n_active++;
      AR_transitions(n, k, p_sta->p_ar_coeffs, n_active, &p_transition);
      starting_next->active_id = AR_add_arrival(&p_means, &p_covars, n);
      starting_next = starting_next->next_start;
    }



    // clean up any events that have finished
    while (ending_next != NULL && time >= ending_next->end_time) {
      LogTrace(" deactivating arrivalwaveform w/ et %lf at time %lf", ending_next->end_time, time);
      active_arrivals = remove_active(active_arrivals, ending_next);
      n_active--;
      AR_transitions(n, k, p_sta->p_ar_coeffs, n_active, &p_transition);
      AR_remove_arrival(&p_means, &p_covars, n, ending_next->active_id);
      for (ArrivalWaveform_t * a = active_arrivals; a != NULL; a = a->next_active) {
	if (a->active_id > ending_next->active_id) {
	  LogTrace(" remove arridx %d, decrementing %d", ending_next->active_id, a->active_id);
	  a->active_id--;
	} else {
	  LogTrace(" remove arridx %d, not decrementing %d", ending_next->active_id, a->active_id);
	}
      }
      ending_next = ending_next->next_end;
    }

    AR_observation(n, k, n_active, active_arrivals, p_segment, &p_observation);

    // compute the predicted envelope for each component

    double env_bhz = 0;
    double env_bhe = 0;
    double env_bhn = 0;

    for(ArrivalWaveform_t * a = active_arrivals; a != NULL; a = a->next_active) {
      if (a->idx >= a->len) continue;
      env_bhz += a->p_envelope[a->idx] * a->bhz_coeff;
      env_bhn += a->p_envelope[a->idx] * a->bhn_coeff;
      env_bhe += a->p_envelope[a->idx] * a->bhe_coeff;
      a->idx++;
      LogTrace("getting envelope from active id %d st %lf coeffs z %lf e %lf n %lf idx %d env %lf", a->active_id, a->start_time, a->bhz_coeff, a->bhe_coeff, a->bhn_coeff, a->idx, a->p_envelope[a->idx]);
    }

    double pred_bhz = p_sta->chan_means[CHAN_BHZ] + env_bhz;
    double pred_bhe = p_sta->chan_means[CHAN_BHE] + env_bhe;
    double pred_bhn = p_sta->chan_means[CHAN_BHN] + env_bhn;

    double obs_bhz, obs_bhe, obs_bhn;
    int obs_perturb_n = 0;
    if (p_segment->p_channels[CHAN_BHZ] != NULL) {
      obs_bhz = p_segment->p_channels[CHAN_BHZ]->p_bands[band][t] - pred_bhz;
      gsl_vector_set(obs_perturb, obs_perturb_n++, obs_bhz);
    }
    if (p_segment->p_channels[CHAN_BHE] != NULL) {
      obs_bhe = p_segment->p_channels[CHAN_BHE]->p_bands[band][t] - pred_bhe;
      gsl_vector_set(obs_perturb, obs_perturb_n++, obs_bhe);
    }
    if (p_segment->p_channels[CHAN_BHN] != NULL) {
      obs_bhn = p_segment->p_channels[CHAN_BHN]->p_bands[band][t] - pred_bhn;
      gsl_vector_set(obs_perturb, obs_perturb_n++, obs_bhn);
    }


    if (n_active > 0) {
      AR_predict(p_means, p_covars, p_transition, p_sta->ar_noise_sigma2, n);
      AR_update(p_means, p_covars, p_observation, obs_perturb, obs_covar, n, k, residuals, residual_covars);
      //LogTrace("pz %lf pe %lf pn %lf rz %lf re %lf rn %lf", gsl_vector_get(obs_perturb, 0), gsl_vector_get(obs_perturb, 1), gsl_vector_get(obs_perturb, 2), gsl_vector_get(residuals, 0), gsl_vector_get(residuals, 1), gsl_vector_get(residuals, 2));
    } else {
      gsl_vector_memcpy(residuals, obs_perturb);
      gsl_matrix_memcpy(residual_covars, obs_covar);
    }

    double det = matrix_inv_det(residual_covars, residual_covars_inv);
    gsl_blas_dgemv(CblasNoTrans, 1, residual_covars_inv, residuals, 0, residuals_tmp);
    double ex;

    gsl_blas_ddot(residuals, residuals_tmp, &ex);
    double thisll = 0.5 * k*log(2*PI) + .5 * log(det) + 0.5 * ex;
    LogTrace(" ll minus %lf is %lf (%lf = 0.5 * %d * log(2pi) + .5 * log(%lf) + 0.5 * %lf", thisll, ll, thisll, k, det, ex);
    assert(!isnan(thisll) && thisll > -1*DBL_MAX);
    ll -= thisll;


    //filtered_perturbs.p_data[t] = obs_perturb_mean-residual;
    //filtered_waveform.p_data[t] = env_bhz + obs_perturb_mean-residual;
    //noise.p_data[t] = residual;

  }

  for(ArrivalWaveform_t * a = st_arrivals; a != NULL; ) {
    free(a->p_envelope);
    ArrivalWaveform_t * next_a = a->next_start;
    free(a);
    a = next_a;
  }

  if (p_means != NULL) gsl_vector_free(p_means);
  if (p_covars != NULL) gsl_matrix_free(p_covars);
  if (p_transition != NULL) gsl_matrix_free(p_transition);
  if (p_observation != NULL) gsl_matrix_free(p_observation);
  gsl_vector_free(obs_perturb);
  gsl_matrix_free(obs_covar);
  gsl_vector_free (residuals);
  gsl_vector_free (residuals_tmp);
  gsl_matrix_free (residual_covars);
  gsl_matrix_free (residual_covars_inv);

  /*  save_pdf_plot(p_sigmodel, &filtered_perturbs, "filtered_perturbs");
  save_pdf_plot(p_sigmodel, &filtered_waveform, "filtered_waveform");
  save_pdf_plot(p_sigmodel, &noise, "noise");

  free(filtered_perturbs.p_data);
  free(filtered_waveform.p_data);*/


  LogTrace ("returning ar ll %lf", ll);

  return ll;
}



/* Fills in the signal envelope for a set of event arrivals at a
   three-axis station. p_segment must set start_time, hz, and
   siteid. */
void Spectral_Envelope_Model_SampleThreeAxis(void * pv_params,
				   EarthModel_t * p_earth, 
				   Segment_t * p_segment,
				   int num_arrivals, 
				   const Arrival_t ** pp_arrivals,
				   int samplePerturb,
				   int sampleNoise) {
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *) pv_params;
  int siteid = p_segment->siteid;
  Spectral_StationModel_t * p_sta = p_params->p_stations + siteid - 1;

  double end_time = p_segment->start_time + p_segment->len / p_segment->hz;
   
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  int n = p_sta->ar_n;

  ArrivalWaveform_t * st_arrivals = NULL;
  ArrivalWaveform_t * et_arrivals = NULL;

  // initialize random number generator
  const gsl_rng_type * T;
  gsl_rng * r;   
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set(r, time(NULL));

  double stddev = sqrt(p_sta->ar_noise_sigma2);

  for (int band=0; band < NUM_BANDS; ++band) {

    p_segment->p_channels[CHAN_BHZ] = alloc_channel(p_segment);
    p_segment->p_channels[CHAN_BHZ]->chan = CHAN_BHZ;
    p_segment->p_channels[CHAN_BHZ]->p_bands[band] = calloc(p_segment->len, sizeof(double));
    p_segment->p_channels[CHAN_BHN] = alloc_channel(p_segment);
    p_segment->p_channels[CHAN_BHN]->chan = CHAN_BHN;
    p_segment->p_channels[CHAN_BHN]->p_bands[band] = calloc(p_segment->len, sizeof(double));
    p_segment->p_channels[CHAN_BHE] = alloc_channel(p_segment);
    p_segment->p_channels[CHAN_BHE]->chan = CHAN_BHE;
    p_segment->p_channels[CHAN_BHE]->p_bands[band] = calloc(p_segment->len, sizeof(double));

  /* populate two linked lists, storing waveform info sorted by
     start_time and end_time respectively */
  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);
      
    if (p_arr->amp == 0 || p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = 0;

    if (p_sta->override_b != 0) {
      abstract_spectral_env_raw(p_sta->override_b, p_sta->override_gamma, p_sta->override_height, p_segment->hz, &w->p_envelope, &w->len);
    } else {
      abstract_spectral_env(p_sta, band, p_arr, p_segment->hz, &w->p_envelope, &w->len);
    }

    if (samplePerturb) {
      w->last_perturbs = calloc(p_sta->ar_n, sizeof(double));
      for(int t=0; t < w->len; ++t) {
	double newperturb=0;
	for(int j=0; j < p_sta->ar_n; ++j) {
	  newperturb += w->last_perturbs[j] * p_sta->p_ar_coeffs[j]; 
	  //printf("inc newperturb by %lf * %lf = %lf to %lf\n", w->last_perturbs[j], p_sta->p_ar_coeffs[j] , w->last_perturbs[j] *  p_sta->p_ar_coeffs[j], newperturb);
	  if (j > 0) w->last_perturbs[j-1] = w->last_perturbs[j];
	}
	double epsilon = gsl_ran_gaussian(r, stddev);
	newperturb += epsilon;
      
	w->last_perturbs[p_sta->ar_n-1] = newperturb;
	newperturb *= w->p_envelope[t];
	w->p_envelope[t] = w->p_envelope[t] + newperturb;
      }
      free(w->last_perturbs);
    }
    w->end_time = w->start_time + (double) w->len / p_segment->hz;

    double iangle;
    if(!slowness_to_iangle(p_arr->slo, p_arr->phase, &iangle)) {
      //LogTrace("iangle conversion failed from slowness %lf phaseid %d, setting default iangle 45.", p_arr->slo, phase);
      iangle = 45;
    }
    
    w->bhe_coeff = fabs(SPHERE2X(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->bhn_coeff = fabs(SPHERE2Y(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->bhz_coeff = 1;

    LogTrace("azi %lf slo %lf iangle %lf bhe %lf bhn %lf bhz %lf", p_arr->azi, p_arr->slo, iangle, w->bhe_coeff, w->bhn_coeff, w->bhz_coeff);

    st_arrivals = insert_st(st_arrivals, w);
    et_arrivals = insert_et(et_arrivals, w);

  }



  ArrivalWaveform_t * starting_next = st_arrivals;
  ArrivalWaveform_t * ending_next = et_arrivals;
  ArrivalWaveform_t * active_arrivals = NULL;
  int n_active = 0;

  for (int t = 0; t < p_segment->len; ++t) {
    double time = p_segment->start_time + t/p_segment->hz;

    // activate the next event, if needed
    while (starting_next != NULL && time >= starting_next->start_time) {
      //LogTrace(" activating arrivalwaveform w/ st %lf at time %lf t %d", starting_next->start_time, time, t);
      active_arrivals = append_active(active_arrivals, starting_next);
      starting_next->active_id = n_active++;
      starting_next = starting_next->next_start;
    }

    // clean up any events that have finished
    while (ending_next != NULL && time >= ending_next->end_time) {
      //LogTrace(" deactivating arrivalwaveform w/ et %lf at time %lf", ending_next->end_time, time);
      active_arrivals = remove_active(active_arrivals, ending_next);
      n_active--;
      for (ArrivalWaveform_t * a = active_arrivals; a != NULL; a = a->next_active) {
	if (a->active_id > ending_next->active_id) {
	  a->active_id--;
	} else {
	}
      }
      ending_next = ending_next->next_end;
    }

    
    // compute the predicted envelope for each component
    double env_bhz = p_sta->chan_means[CHAN_BHZ];
    double env_bhe = p_sta->chan_means[CHAN_BHE];
    double env_bhn = p_sta->chan_means[CHAN_BHN];
    for(ArrivalWaveform_t * a = active_arrivals; a != NULL; a = a->next_active) {
      if (a->idx >= a->len) continue;
      env_bhz += a->p_envelope[a->idx] * a->bhz_coeff;
      env_bhn += a->p_envelope[a->idx] * a->bhn_coeff;
      env_bhe += a->p_envelope[a->idx] * a->bhe_coeff;
      a->idx++;
      LogTrace("getting envelope from active id %d st %lf coeffs z %lf e %lf n %lf idx %d env %lf", a->active_id, a->start_time, a->bhz_coeff, a->bhe_coeff, a->bhn_coeff, a->idx, a->p_envelope[a->idx]);
    }
    p_segment->p_channels[CHAN_BHZ]->p_bands[band][t] = env_bhz; 
    p_segment->p_channels[CHAN_BHE]->p_bands[band][t] = env_bhe;
    p_segment->p_channels[CHAN_BHN]->p_bands[band][t] = env_bhn;
									   
    if(sampleNoise) {
      printf("sampling gaussian noise with var %lf\n", p_sta->chan_vars[CHAN_BHZ]);
      p_segment->p_channels[CHAN_BHZ]->p_bands[band][t] += 
	fabs(gsl_ran_gaussian(r,sqrt(p_sta->chan_vars[CHAN_BHZ]))); 
      p_segment->p_channels[CHAN_BHE]->p_bands[band][t] += 
	fabs(gsl_ran_gaussian(r,sqrt(p_sta->chan_vars[CHAN_BHE]))); 
      p_segment->p_channels[CHAN_BHN]->p_bands[band][t] += 
	fabs(gsl_ran_gaussian(r,sqrt(p_sta->chan_vars[CHAN_BHN]))); 
    }

  }

  for(ArrivalWaveform_t * a = st_arrivals; a != NULL; a = a->next_start) {
    free(a->p_envelope);
    free(a);
  }


  }


  gsl_rng_free(r);

}

void Spectral_Envelope_Model_UnInit(void * pv_params) {
  Spectral_Envelope_Model_t * p_params = (Spectral_Envelope_Model_t *) pv_params;
  
  for (int i=0; i < p_params->numsites; ++i) {
    free((p_params->p_stations + i)->p_ar_coeffs);
  }
  free(p_params->p_stations);

}
