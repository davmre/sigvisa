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

void Envelope_SignalModel_Set_Params(void * pv_params, int siteid, PyObject * py_dict) {
  Envelope_SignalModel_t * p_params = (Envelope_SignalModel_t *) pv_params;
  

  Envelope_StationModel_t * sta = NULL;
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

      if (strcmp(key, "env_p_height") == 0) {
	LogTrace("setting %s to %lf at siteid %d", key, PyFloat_AsDouble(py_value), siteid);      
	sta->env_p_height = PyFloat_AsDouble(py_value); CHECK_ERROR;
      }  else if (strcmp(key, "env_s_height") == 0) {
	LogTrace("setting %s to %lf at siteid %d", key, PyFloat_AsDouble(py_value), siteid);      
	sta->env_s_height = PyFloat_AsDouble(py_value); CHECK_ERROR;
      } else if (strcmp(key, "env_p_onset") == 0) {
	LogTrace("setting %s to %lf at siteid %d", key, PyFloat_AsDouble(py_value), siteid);      
	sta->env_p_onset = PyFloat_AsDouble(py_value); CHECK_ERROR;
      } else if (strcmp(key, "env_p_decay") == 0) {
	LogTrace("setting %s to %lf at siteid %d", key, PyFloat_AsDouble(py_value), siteid);      
	sta->env_p_decay = PyFloat_AsDouble(py_value); CHECK_ERROR;
      } else if (strcmp(key, "env_s_onset") == 0) {
	LogTrace("setting %s to %lf at siteid %d", key, PyFloat_AsDouble(py_value), siteid);      
	sta->env_s_onset = PyFloat_AsDouble(py_value); CHECK_ERROR;
      } else if (strcmp(key, "env_s_decay") == 0) {
	LogTrace("setting %s to %lf at siteid %d", key, PyFloat_AsDouble(py_value), siteid);      
	sta->env_s_decay = PyFloat_AsDouble(py_value); CHECK_ERROR;
      } else if (strncmp(key, "chan_mean_", 10) == 0) {
	int chan_num = canonical_channel_num(key+10); CHECK_ERROR;
	LogTrace("setting chan mean %lf for chan str %s int %d at siteid %d", PyFloat_AsDouble(py_value), key+10, chan_num, siteid);
	sta->chan_means[chan_num] = PyFloat_AsDouble(py_value); CHECK_ERROR;
      } else if (strncmp(key, "chan_var_", 9) == 0) {
	int chan_num = canonical_channel_num(key+9);
	LogTrace("setting chan var %lf for chan str %s int %d siteid %d", PyFloat_AsDouble(py_value), key+9, chan_num, siteid);
	sta->chan_vars[chan_num] = PyFloat_AsDouble(py_value); CHECK_ERROR;

      } else if (strcmp(key, "ar_noise_sigma2") == 0) {
	LogTrace("setting %s to %lf at siteid %d", key, PyFloat_AsDouble(py_value), siteid);      
	double val = PyFloat_AsDouble(py_value); CHECK_ERROR;
	
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

void Envelope_SignalModel_Init_Params(void * pv_params,  int numsites) {
  Envelope_SignalModel_t * p_params = (Envelope_SignalModel_t *) pv_params;

  p_params->numsites = numsites;

  /* using calloc (instead of malloc) is important here since we
     depend on the p_ar_coeffs pointer being NULL iff unallocated */
  p_params->p_stations = calloc(numsites, sizeof(Envelope_StationModel_t));
}

int Envelope_SignalModel_Has_Model(void * pv_sigmodel, int siteid, int chan) {

  SigModel_t * p_sigmodel = (SigModel_t *) pv_sigmodel;
  SignalModel_t * p_model = &p_sigmodel->signal_model;
  Envelope_SignalModel_t * p_params = (Envelope_SignalModel_t *)p_model->pv_params;

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


typedef struct ArrivalWaveform {
  double start_time;
  long idx;
  double end_time;
  long len;
  double * p_envelope;

  struct ArrivalWaveform * next_start;
  struct ArrivalWaveform * next_end;
  struct ArrivalWaveform * next_active;
  int active_id;

  double bhe_coeff;
  double bhn_coeff;
  double bhz_coeff;

  // used for sampling
  double * last_perturbs;

} ArrivalWaveform_t;

ArrivalWaveform_t * append_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_active=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->next_active == NULL) {
    // if at the end of the list, add to the end
    p_head->next_active = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_active = append_active(p_head->next_active, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * remove_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  if (p_head == NULL) {
    p_new_head = NULL;
  } else if (p_head == p_arr) {
    p_new_head = p_arr->next_active;
  } else if (p_head->next_active == NULL) {
    p_new_head = p_head;
  } else {
    // otherwise, delete recursively
    p_head->next_active = remove_active(p_head->next_active, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * insert_st(ArrivalWaveform_t * p_head, 
			      ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_start=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->start_time > p_arr->start_time ) {
    // if the new addition comes before the head, make it the new head
    p_arr->next_start = p_head;
    p_new_head = p_arr;
  } else if (p_head->next_start == NULL) {
    // if at the end of the list, add to the end
    p_head->next_start = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_start = insert_st(p_head->next_start, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * insert_et(ArrivalWaveform_t * p_head, 
				 ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_end=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->end_time > p_arr->end_time ) {
    // if the new addition comes before the head, make it the new head
    p_arr->next_end = p_head;
    p_new_head = p_arr;
  } else if (p_head->next_end == NULL) {
    // if at the end of the list, add to the end
    p_head->next_end = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_end = insert_et(p_head->next_end, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}


void abstract_env(Envelope_StationModel_t * p_sta, const Arrival_t * p_arr, double hz, double ** pp_envelope, long *len) {

  double env_onset, env_decay, env_height;
  if (IS_S_PHASE(p_arr->phase)) {
    env_onset = p_sta->env_s_onset;
    env_decay = p_sta->env_s_decay;
    env_height = p_sta->env_s_height;
  } else {
    env_onset = p_sta->env_p_onset;
    env_decay = p_sta->env_p_decay;
    env_height = p_sta->env_p_height;
  }
  LogTrace("env for siteid %d phase %d height %lf onset %lf decay %lf hz %lf amp %lf", p_arr->siteid, p_arr->phase, env_height, env_onset, env_decay, hz, p_arr->amp);

  double peak_height = env_height * p_arr->amp;
  long peak_idx = (long) (log(peak_height) / env_onset * hz);
  long end_idx = peak_idx + (long)(log(peak_height)/env_decay * hz);

  if (peak_idx < 0 || end_idx <=0) {
    peak_idx = 0;
    end_idx = 1;
  }
  
  *len = MIN(end_idx+15*hz, 240*hz);
  double * means = (double *) calloc(*len, sizeof(double));

  if (means == NULL) {
    printf("error allocating memory for means in abstract_env, len %ld\n", *len);
    exit(-1);
  }

  for (long i=0; i < MIN(peak_idx, *len); ++i) {
    means[i] = exp(env_onset * (i/hz));
  }
  for (long i=MIN(peak_idx, *len); i < *len; ++i) {
    means[i] = exp(env_decay * ((end_idx - i)/hz));
    // LogInfo("%d %d %d %lf %d %lf %lf", i, peak_idx, *len, env_decay, end_idx, hz, means[i]);
  }
  
  *pp_envelope = means;
}

int AR_add_arrival(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n) {

  int arridx = 0;

  if (*pp_means == NULL || *pp_covars == NULL) {

    *pp_means = gsl_vector_alloc(n);
    gsl_vector_set_zero(*pp_means);

    *pp_covars = gsl_matrix_alloc(n,n);
    gsl_matrix_set_identity(*pp_covars);

  } else {
    
    int l = (*pp_means)->size;
    arridx = l/n;
    gsl_vector * new_means = gsl_vector_alloc(l + n);
    gsl_vector_set_zero(new_means);
    for(int i=0; i < l; ++i) {
      gsl_vector_set(new_means, i,  gsl_vector_get(*pp_means, i));
    }
    gsl_vector_free(*pp_means);
    *pp_means = new_means;

    gsl_matrix * new_covars = gsl_matrix_alloc(l + n, l + n);
    gsl_matrix_set_identity(new_covars);
    for (int i=0; i < l; ++i) {
      for(int j=0; j < l; ++j) {
	gsl_matrix_set(new_covars, i, j, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    gsl_matrix_free(*pp_covars);
    *pp_covars = new_covars;
  }
  return arridx;
}

void AR_remove_arrival(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n, int arridx) {
  if (*pp_means == NULL || *pp_covars == NULL) {
    return;
  } 

  int l = (*pp_means)->size;
  if (l == n) {
    gsl_vector_free(*pp_means);
    gsl_matrix_free(*pp_covars);
    *pp_means = NULL;
    *pp_covars = NULL;
  } else {
    gsl_vector * new_means = gsl_vector_alloc(l - n);
    for(int i=0; i < arridx*n; ++i) {
      gsl_vector_set(new_means, i,  gsl_vector_get(*pp_means, i));
    }
    for(int i=(arridx+1)*n; i < l; ++i) {
      gsl_vector_set(new_means, i-n,  gsl_vector_get(*pp_means, i));
    }
    gsl_vector_free(*pp_means);
    *pp_means = new_means;

    gsl_matrix * new_covars = gsl_matrix_alloc(l - n, l - n);

    for (int i=0; i < arridx*n; ++i) {
      for(int j=0; j < arridx*n; ++j) {
	gsl_matrix_set(new_covars, i, j, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    // move everything up by n rows
    for (int i=(arridx+1)*n; i < l; ++i) {
      for(int j=0; j < (arridx+1)*n; ++j) {
	gsl_matrix_set(new_covars, i-n, j, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    // move everything in by n columns
    for (int i=0; i < (arridx+1)*n; ++i) {
      for(int j=(arridx+1)*n; j < l; ++j) {
	gsl_matrix_set(new_covars, i, j-n, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    // move everything in by n columns
    for (int i=i=(arridx+1)*n; i < l; ++i) {
      for(int j=(arridx+1)*n; j < l; ++j) {
	gsl_matrix_set(new_covars, i-n, j-n, gsl_matrix_get(*pp_covars, i, j));
      }
    }
    gsl_matrix_free(*pp_covars);
    *pp_covars = new_covars;

  }
}

void AR_transitions(int n, int k, double * ar_coeffs, 
		    int n_arrs, 
		    gsl_matrix ** pp_transition) {
  if (*pp_transition != NULL) {
    gsl_matrix_free(*pp_transition);
  } 
  

  if (n_arrs == 0) {
    *pp_transition = NULL;
    return;
  } 

  *pp_transition = gsl_matrix_alloc(n*n_arrs, n*n_arrs);

  gsl_matrix_set_zero(*pp_transition);

  for (int arr=0; arr < n_arrs; ++arr) {
    gsl_matrix_set(*pp_transition, arr*n+n-1, arr*n, ar_coeffs[0]);
    for (int i=1; i < n; ++i) {
      gsl_matrix_set(*pp_transition, arr*n+i-1, arr*n+i, 1);
      gsl_matrix_set(*pp_transition, arr*n+n-1, arr*n+i, ar_coeffs[i]);
    }
   
  }

}

void AR_observation(int n, int k, 
		    int n_arrs, ArrivalWaveform_t * active_arrivals, 
		    ChannelBundle_t * p_segment,
		    gsl_matrix ** pp_obs) {
  if (*pp_obs != NULL) {
    gsl_matrix_free(*pp_obs);
  }
  
  if (n_arrs == 0) {
    *pp_obs = NULL;
    return;
  } 

  *pp_obs = gsl_matrix_alloc(k, n*n_arrs);
  gsl_matrix_set_zero(*pp_obs);


  for (int arr=0; arr < n_arrs; ++arr) {
    ArrivalWaveform_t * p_a = active_arrivals;
   int curr_chan=0;
    for (int i=0; i < NUM_CHANS; ++i) {
      if (p_segment->p_channels[i] != NULL) {
	switch (i) {
	case CHAN_BHZ:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]); 
	  break;
	case CHAN_BHN:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]); 
	  break;
	case CHAN_BHE:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]); 
	  break;
	}
	curr_chan++;
      }
    }

    active_arrivals=active_arrivals->next_active;
  }


}

void AR_predict(gsl_vector * p_means, gsl_matrix * p_covars, gsl_matrix * p_transition, double noise_sigma2, int n) {
  int num_arrs = p_means->size/n;
  
  gsl_vector * tmp = gsl_vector_alloc(p_means->size);
  gsl_blas_dgemv (CblasNoTrans, 1, p_transition, p_means, 0, tmp);
  gsl_vector_memcpy(p_means, tmp);
  gsl_vector_free(tmp);

  gsl_matrix * mtmp = gsl_matrix_alloc(p_covars->size1, p_covars->size2);
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, p_covars, p_transition, 0, mtmp);  
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, p_transition, mtmp, 0, p_covars);
  
  gsl_matrix_free(mtmp);
  for(int i=0; i < num_arrs; ++i) {
    int idx = (i+1)*n-1;
    gsl_matrix_set(p_covars, idx, idx, gsl_matrix_get(p_covars, idx, idx) + noise_sigma2);
  }
}

double matrix_inv_det(gsl_matrix * A, gsl_matrix * invA) {
  gsl_matrix * LU = gsl_matrix_alloc(A->size1,A->size2);
  gsl_permutation *p = gsl_permutation_alloc(A->size1);
  int signum;
  gsl_matrix_memcpy(LU, A);
  gsl_linalg_LU_decomp(LU, p, &signum);
  gsl_linalg_LU_invert(LU, p, invA);

  double det = gsl_linalg_LU_det (LU, signum);

  gsl_matrix_free(LU);
  gsl_permutation_free(p);
  return det;
}

void AR_update(gsl_vector * p_means, gsl_matrix * p_covars, 
	       gsl_matrix * p_obs, gsl_vector * obs_perturb, 
	       gsl_matrix * obs_covar, int n, int k, 
	       gsl_vector * y, gsl_matrix * S) {

  gsl_blas_dgemv (CblasNoTrans, 1, p_obs, p_means, 0, y);
  //LogTrace("predicted perturb %lf versus obs %lf",  pred_obs_perturb, obs_perturb_mean);
  gsl_vector_sub(y, obs_perturb);
  gsl_vector_scale(y, -1);


  gsl_matrix * Sinv = gsl_matrix_alloc(k,k);
  gsl_matrix * PHt = gsl_matrix_alloc(p_means->size, k);
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, p_covars, p_obs, 0, PHt);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, p_obs, PHt, 0, S);
  gsl_matrix_add (S, obs_covar);
  matrix_inv_det(S, Sinv);

  gsl_matrix * K = gsl_matrix_alloc(p_means->size, k);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, PHt, Sinv, 0, K);

  gsl_vector *mean_update = gsl_vector_alloc(p_means->size);
  gsl_blas_dgemv(CblasNoTrans, 1, K, y, 0, mean_update);
  gsl_vector_add(p_means, mean_update);

  gsl_matrix * I = gsl_matrix_alloc(p_means->size, p_means->size);
  gsl_matrix_set_identity (I);

  gsl_matrix * new_covar = gsl_matrix_alloc(p_means->size, p_means->size);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, -1, K, p_obs, 1, I);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, I, p_covars, 0, new_covar);
  gsl_matrix_memcpy(p_covars, new_covar);

  gsl_matrix_free(Sinv);
  gsl_matrix_free(PHt);
  gsl_matrix_free(K);
  gsl_vector_free(mean_update);
  gsl_matrix_free(I);
  gsl_matrix_free(new_covar);
}


/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + AR(n)
  wiggles + Gaussian iid noise signal model.
 */
double Envelope_SignalModel_Likelihood(void * pv_sigmodel, ChannelBundle_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals) {

  SigModel_t * p_sigmodel = (SigModel_t *) pv_sigmodel;
  SignalModel_t * p_model = &p_sigmodel->signal_model;

  int siteid = p_segment->siteid;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);

  Envelope_SignalModel_t * p_params = (Envelope_SignalModel_t * )p_model->pv_params;
  Envelope_StationModel_t * p_sta = p_params->p_stations + siteid - 1;


  double ll = 0;
  double iidll = 0;

  int n = p_sta->ar_n;

  ArrivalWaveform_t * st_arrivals = NULL;
  ArrivalWaveform_t * et_arrivals = NULL;

  /*
  Signal_t filtered_waveform;
  filtered_waveform.p_data = calloc(p_segment->len, sizeof(double));
  filtered_waveform.len = p_segment->len;
  filtered_waveform.start_time = p_segment->start_time;
  filtered_waveform.hz = p_segment->hz;
  filtered_waveform.siteid = p_segment->siteid;
  filtered_waveform.py_array=NULL;
  filtered_waveform.chan=CHAN_BHZ;

  Signal_t filtered_perturbs;
  filtered_perturbs.p_data = calloc(p_segment->len, sizeof(double));
  filtered_perturbs.len = p_segment->len;
  filtered_perturbs.start_time = p_segment->start_time;
  filtered_perturbs.hz = p_segment->hz;
  filtered_perturbs.siteid = p_segment->siteid;
  filtered_perturbs.py_array=NULL;
  filtered_perturbs.chan=CHAN_BHZ;

  Signal_t noise;
  noise.p_data = calloc(p_segment->len, sizeof(double));
  noise.len = p_segment->len;
  noise.start_time = p_segment->start_time;
  noise.hz = p_segment->hz;
  noise.siteid = p_segment->siteid;
  noise.py_array=NULL;
  noise.chan=CHAN_BHZ;*/

  /* populate two linked lists, storing waveform info sorted by
     start_time and end_time respectively */
  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);
      
    if (p_arr->amp == 0 || p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = 0;
    abstract_env(p_sta, p_arr, p_segment->hz, &w->p_envelope, &w->len);
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
      obs_bhz = p_segment->p_channels[CHAN_BHZ]->p_data[t] - pred_bhz;
      gsl_vector_set(obs_perturb, obs_perturb_n++, obs_bhz);
    }
    if (p_segment->p_channels[CHAN_BHE] != NULL) {
      obs_bhe = p_segment->p_channels[CHAN_BHE]->p_data[t] - pred_bhe;
      gsl_vector_set(obs_perturb, obs_perturb_n++, obs_bhe);
    }
    if (p_segment->p_channels[CHAN_BHN] != NULL) {
      obs_bhn = p_segment->p_channels[CHAN_BHN]->p_data[t] - pred_bhn;
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
void Envelope_SignalModel_SampleThreeAxis(void * pv_params,
				   EarthModel_t * p_earth, 
				   ChannelBundle_t * p_segment,
				   int num_arrivals, 
				   const Arrival_t ** pp_arrivals,
				   int samplePerturb,
				   int sampleNoise) {
  Envelope_SignalModel_t * p_params = (Envelope_SignalModel_t *) pv_params;
  int siteid = p_segment->siteid;
  Envelope_StationModel_t * p_sta = p_params->p_stations + siteid - 1;


  double end_time = p_segment->start_time + p_segment->len / p_segment->hz;


  p_segment->p_channels[CHAN_BHZ] = alloc_signal(p_segment);
  p_segment->p_channels[CHAN_BHZ]->chan = CHAN_BHZ;
  p_segment->p_channels[CHAN_BHZ]->p_data = calloc(p_segment->len, sizeof(double));
  p_segment->p_channels[CHAN_BHN] = alloc_signal(p_segment);
  p_segment->p_channels[CHAN_BHN]->chan = CHAN_BHN;
  p_segment->p_channels[CHAN_BHN]->p_data = calloc(p_segment->len, sizeof(double));
  p_segment->p_channels[CHAN_BHE] = alloc_signal(p_segment);
  p_segment->p_channels[CHAN_BHE]->chan = CHAN_BHE;
  p_segment->p_channels[CHAN_BHE]->p_data = calloc(p_segment->len, sizeof(double));
    
  
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

  /* populate two linked lists, storing waveform info sorted by
     start_time and end_time respectively */
  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);
      
    if (p_arr->amp == 0 || p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = 0;

    abstract_env(p_sta, p_arr, p_segment->hz, &w->p_envelope, &w->len);

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
    p_segment->p_channels[CHAN_BHZ]->p_data[t] = env_bhz; 
    p_segment->p_channels[CHAN_BHE]->p_data[t] = env_bhe;
    p_segment->p_channels[CHAN_BHN]->p_data[t] = env_bhn;
									   
    if(sampleNoise) {
      printf("sampling gaussian noise with var %lf\n", p_sta->chan_vars[CHAN_BHZ]);
      p_segment->p_channels[CHAN_BHZ]->p_data[t] += 
	fabs(gsl_ran_gaussian(r,sqrt(p_sta->chan_vars[CHAN_BHZ]))); 
      p_segment->p_channels[CHAN_BHE]->p_data[t] += 
	fabs(gsl_ran_gaussian(r,sqrt(p_sta->chan_vars[CHAN_BHE]))); 
      p_segment->p_channels[CHAN_BHN]->p_data[t] += 
	fabs(gsl_ran_gaussian(r,sqrt(p_sta->chan_vars[CHAN_BHN]))); 
    }

  }


  for(ArrivalWaveform_t * a = st_arrivals; a != NULL; a = a->next_start) {
    free(a->p_envelope);
    free(a);
  }

  gsl_rng_free(r);

}

void Envelope_SignalModel_UnInit(void * pv_params) {
  Envelope_SignalModel_t * p_params = (Envelope_SignalModel_t *) pv_params;
  
  for (int i=0; i < p_params->numsites; ++i) {
    free((p_params->p_stations + i)->p_ar_coeffs);
  }
  free(p_params->p_stations);

}
