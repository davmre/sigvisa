#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "../sigvisa.h"

void SignalPrior_Init_Params(SignalPrior_t * prior, const char * filename, int numsites) {
  
  FILE * fp;

  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (3 != fscanf(fp, "%lf %lf %lf\n", &prior->env_height, &prior->env_decay, &prior->env_onset))
    {
    fprintf(stderr, "error reading envelope coefficients from %s\n", filename);
    exit(1);
  }


  if (1 != fscanf(fp, "%d", &prior->ar_n)) {
    fprintf(stderr, "error reading AR_n from %s\n", filename);
    exit(1);
  }
  prior->p_ar_coeffs = calloc(prior->ar_n, sizeof(double));
  for(int i=0; i < prior->ar_n; ++i) {
    if (1 != fscanf(fp, "%lf", prior->p_ar_coeffs+i)) {
      fprintf(stderr, "error reading AR coefficient %d of %d from %s\n", i, prior->ar_n, filename);
      exit(1);
    }
  }
  if (1 != fscanf(fp, "%lf\n", &prior->ar_noise_sigma2)) {
    fprintf(stderr, "error reading AR_noise_sigma2 from %s\n", filename);
    exit(1);
  }
 
  prior->numsites = numsites;  
  prior->p_stations = (StationNoiseModel_t *) calloc(numsites, sizeof(StationNoiseModel_t));

  for (int i=0; i < numsites; ++i) {
    for (int j=0; j < NUM_CHANS; ++j) {
      (prior->p_stations + i)->chan_means[j] = 0;
      (prior->p_stations + i)->chan_vars[j] = 10;
      LogTrace(" set %d, %d, vars to %lf", i, j, (prior->p_stations + i)->chan_vars[j]);
    }
  }

  int numentries;
  if (1 != fscanf(fp, "%d\n", &numentries))
  {
    LogFatal("error reading num entries from %s\n", filename);
    exit(1);
  }

  for (int i=0; i < numentries; ++i) {

    int siteid;
    int num_chans;
    if (2 != fscanf(fp, "%d %d", &siteid, &num_chans)) {
      LogFatal("error reading siteid and num_chans from station line %d of %s\n", i, filename);
      exit(1);
    }

    for (int j=0; j < num_chans; ++j) {
      int chan_num;
      double mean, var;
      if (3 != fscanf(fp, " %d %lf %lf", &chan_num, &mean, &var))
	{
	  LogFatal("error reading mean and variance for site %d from %s\n", i, filename);
	  exit(1);
	}
    
      LogTrace("%d: loaded %d %d %lf %lf", i, siteid, chan_num, mean, var);
      (prior->p_stations + siteid-1)->chan_means[chan_num] = mean;
      (prior->p_stations + siteid-1)->chan_vars[chan_num] = var;
    }
  }

  fclose(fp);
  fflush(stdout);
}

void vector_times_scalar_inplace(int n, double * vector, double scalar) {
  for (int i=0; i < n; ++i) {
    *(vector++) *= scalar;
  }
}

void vector_abs_inplace(int n, double * vector) {
  for (int i=0; i < n; ++i) {
    vector[i] = fabs(vector[i]);
  }
}

/*
double * vector_times_scalar_copy(int n, double * vector, double scalar) {
  double * nvector = malloc(n * sizeof(double));
  for (int i=0; i < n; ++i) {
    nvector[i] = vector[i]*scalar;
  }
  return nvector;
}
*/

void print_vector(int n, double * vector) {
  for(int i=0; i < n; ++i) {
    if (i == 0 || vector[i] != vector[i-1])
      fprintf(stdout, "%d %lf \n", i, vector[i]);
  }
}

long time2idx(double t, double start_time, double hz) {
  double delta_t = t - start_time;
  long result = floorl(delta_t * hz);

  return result;
}



void arrival_list(EarthModel_t * p_earth, int siteid, double min_time, double max_time, int num_events, const Event_t ** pp_events, int * num_arrivals, Arrival_t *** ppp_arrivals) {
  
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  *num_arrivals = 0;
  *ppp_arrivals = calloc(num_events*MAX_PHASE(numtimedefphases), sizeof(Arrival_t *));

  for (int i=0; i < num_events; ++i) {
    const Event_t * p_event = *(pp_events+i);
    for (int j=0; j < MAX_PHASE(numtimedefphases); ++j) {
      Arrival_t * p_arr = p_event->p_arrivals + (siteid-1)*numtimedefphases + j;
      if (p_arr->amp == 0 || p_arr->time <= 0) {
	continue;
      }
      if (p_arr->time < min_time - MAX_ENVELOPE_LENGTH && p_arr->time > max_time) {
	continue;
      }
      *(*ppp_arrivals + (*num_arrivals)++) = p_arr;
    }
  }
}

int test_identity(int n, double * x, double * means) {

  for (int i=0; i < n; ++i) {
    double diff = fabs(x[i] - means[i]);
    if (diff > 0.0001) {
      printf("NOT IDENTICAL: discrepancy at %d\n", i);
      print_vector(n, means);
      printf("NOT IDENTICAL: discrepancy at %d\n", i);
      print_vector(n, x);
      exit(0);
    }
  }
  printf("identity check passed!\n");
  return 1;
}

double indep_Gaussian_LogProb(int n, double * x, double * means, double * vars) {

  double lp = 0;

  for (int i=0; i < n; ++i) {
    assert(vars[i] >= 0);
    if (isnan(x[i]) || x[i] < 0) continue;    
    lp -= 0.5 * log(2*PI * vars[i]) + 0.5 * (x[i] - means[i])*(x[i] - means[i]) / vars[i];
  }

  return lp;
}


void phase_env_doubleexp(SignalPrior_t * prior, 
			 const Arrival_t * p_arr, 
			 double hz,
			 int chan_num,
			 int phaseid,
			 double ** p_envelope,
			 int * t) {

  double peak_height = prior->env_height * p_arr->amp;
  long peak_idx = (long) (log(peak_height) / prior->env_onset * hz);
  long end_idx = peak_idx + (long)(log(peak_height)/prior->env_decay * hz);

  if (peak_idx < 0 || end_idx <=0) {
    peak_idx = 0;
    end_idx = 1;
  }

  //printf("double-exp height %lf peak_idx %ld end_idx %ld\n", peak_height, peak_idx, end_idx);


  double component_coeff = 0;
  double iangle;
  int res = slowness_to_iangle(p_arr->slo, phaseid, &iangle);
  if (!res) {
    //    LogTrace("iangle conversion failed, setting default 45");
    iangle=45;
  }
  //LogInfo("iangle %lf slowness %lf amp %lf phase %d ", iangle, p_arr->slo, p_arr->amp, phaseid);
  switch (chan_num) {
  case CHAN_BHE:
    component_coeff = SPHERE2X(p_arr->azi, iangle)/SPHERE2Z(p_arr->azi, iangle); break;
  case CHAN_BHN:
    component_coeff = SPHERE2Y(p_arr->azi, iangle)/SPHERE2Z(p_arr->azi, iangle); break;
  case CHAN_BHZ:
    component_coeff = 1; break;
  }
  
  //  printf("generating event signal with arrival azi %lf and slo %lf\n", p_arr->azi, p_arr->slo);
  //  printf("channel is %d and ratio is %lf\n", chan_num, component_coeff);

  /*if (len >= 30 * hz) {
    printf("event lasting more than 30 seconds! env_decay = %lf, hz = %lf, step = %lf, amp = %lf, env_height = %lf, newmean = %lf, len = %ld\n", prior->env_decay, hz, step, p_arr->amp, prior->env_height, newmean, len);
    step = newmean /(30.0 * hz);
    len = ceil( newmean / step );
    printf("resetting step to %lf , len to %ld\n", step, len);
    }*/
  long len = end_idx;
  double * means = (double *) calloc(len, sizeof(double));

  if (means == NULL) {
    printf("error allocating memory for means in phase_env, len %ld\n", len);
    exit(-1);
  }

  for (int i=0; i < peak_idx; ++i) {
    means[i] = exp(prior->env_onset * (i/hz)) * component_coeff;
  }
  for (int i=peak_idx; i < end_idx; ++i) {
    means[i] = exp(prior->env_decay * ((end_idx - i)/hz)) * component_coeff;
  }
  
  *p_envelope = means;
  *t = len;
}

int add_signals(double * dest, long dest_len, double * source, long source_len, long offset) {
  
  long max_i = dest_len - offset;
  long i;

  //fprintf(stdout, " dest len %ld, source len %ld, offset %ld, max i %ld\n", dest_len, source_len, offset, max_i);

  for (i=0; (i < source_len) && (i < max_i); ++i) {
    if (offset + i < 0 || offset+i > dest_len || i < 0 || i > source_len) continue;
    dest[offset+i] = dest[offset+i] + source[i];
  }
  return i;
}

void envelope_means_vars(SignalPrior_t * prior, 
			 double hz,
			 double start_time,
			 double end_time,
			 EarthModel_t * p_earth, 
			 int num_arrivals,
			 const Arrival_t ** pp_arrivals,
			 int siteid,
			 int chan_num, 
			 long * p_len,
			 double ** pp_means,
			 double ** pp_vars,
			 int abs_env) {
 
  // force our samples to align with second boundaries

  start_time = ((long) (start_time * hz))/hz;


  (*p_len) = time2idx(end_time, start_time, hz);



  (*pp_means) = (double *) calloc(*p_len, sizeof(double));
  double *p_means = (*pp_means);
  if (p_means == NULL) {
    printf("error allocating memory for signal means: siteid %d, chan %d, start %lf, end %lf, hz %lf, len %ld\n", siteid, chan_num, start_time, end_time, hz, *p_len);
    exit(-1);
  }

  double *p_vars=0;
  if (pp_vars != NULL) {
    (*pp_vars) = (double *) calloc(*p_len, sizeof(double));
    p_vars = (*pp_vars);
    if (p_vars == NULL) {
      printf("error allocating memory for signal vars: siteid %d, chan %d, start %lf, end %lf, hz %lf, len %ld\n", siteid, chan_num, start_time, end_time, hz, *p_len);
      exit(-1);
    }
  }

  double noise_mean = (prior->p_stations + siteid-1)->chan_means[chan_num];
  double noise_var = (prior->p_stations + siteid-1)->chan_vars[chan_num];
  for (int i=0; i < *p_len; ++i) {
    p_means[i] = noise_mean;
    if (pp_vars != NULL) p_vars[i] = noise_var;
  }

  for (int i=0; i < num_arrivals; ++i) {
    
    const Arrival_t * p_arr = *(pp_arrivals+i);

    //    printf("event %d at siteid %d, ratios n/z %lf e/z %lf\n", i, siteid, SPHERE2Y(p_arr->azi, p_arr->slo)/SPHERE2Z(p_arr->azi, p_arr->slo), SPHERE2X(p_arr->azi, p_arr->slo)/SPHERE2Z(p_arr->azi, p_arr->slo)   );

      double arrtime = p_arr->time;
      if (arrtime < 0) continue;
    
      long idx = time2idx(arrtime, start_time, hz);

      // skip events which don't arrive during the current time period
      if (idx < 0 - MAX_ENVELOPE_LENGTH * hz || idx >= *p_len) {
	//printf("skipping event arrival at time %lf vs start_time %lf idx %ld\n", arrtime, start_time, idx);
	continue;
      }

      double * p_envelope;
      int env_len;
      phase_env_doubleexp(prior, p_arr, hz, chan_num, p_arr->phase, &p_envelope, &env_len);
      // printf("got envelope of len %d (idx = %ld)\n", env_len, idx);

      if (abs_env) {
	vector_abs_inplace(env_len, p_envelope);
      }
      add_signals(p_means, *p_len, p_envelope, env_len, idx);
      if (pp_vars != NULL) {
	vector_times_scalar_inplace(env_len, p_envelope, 0.5);
	vector_abs_inplace(env_len, p_envelope);
	add_signals(p_vars, *p_len, p_envelope, env_len, idx);
      }
      
      free(p_envelope);
    
  }

}

/* Fills in the signal envelope for a set of event arrivals at a
   three-axis station. p_segment must set start_time, hz, and
   siteid. */
void SignalPrior_ThreeAxisEnvelope(SignalPrior_t * prior, 
				   EarthModel_t * p_earth, 
				   int numevents, 
				   const Event_t ** pp_events,
				   ChannelBundle_t * p_segment,
				   ChannelBundle_t * p_wave_segment) {
  int chan_nums[3];
  chan_nums[0] = CHAN_BHE;
  chan_nums[1] = CHAN_BHN;
  chan_nums[2] = CHAN_BHZ;

  double end_time = p_segment->start_time + p_segment->len / p_segment->hz;


    int num_arrivals;
    Arrival_t ** pp_arrivals;
    arrival_list(p_earth, p_segment->siteid, p_segment->start_time, ChannelBundle_EndTime(p_segment), numevents, pp_events, &num_arrivals, &pp_arrivals);
    LogInfo("generating signal, numarrivals %d", num_arrivals);
  for (int i=0; i < 3; ++i) {
    
    int chan_num = chan_nums[i];
    
    p_segment->p_channels[chan_num] = alloc_signal(p_segment);
    p_segment->p_channels[chan_num]->chan = chan_num;
    // printf("tae calling envelope_means_vars(%lf, %lf, %lf, %d, %d, %d)\n", p_segment->hz, p_segment->start_time, end_time, numevents, p_segment->siteid, chan_num);


    envelope_means_vars(prior, 
			p_segment->hz, p_segment->start_time, end_time,
			p_earth, num_arrivals, (const Arrival_t **)pp_arrivals, 
			p_segment->siteid,chan_num, 
			&(p_segment->p_channels[chan_num]->len),
			&(p_segment->p_channels[chan_num]->p_data),
			NULL,
			1);
    assert(p_segment->p_channels[chan_num]->len == p_segment->len);



    if (p_wave_segment != NULL) {
      p_wave_segment->p_channels[chan_num] = alloc_signal(p_wave_segment);
      p_wave_segment->p_channels[chan_num]->chan = chan_num;
      envelope_means_vars(prior, 
			  p_segment->hz, p_segment->start_time, end_time,
			  p_earth, num_arrivals, (const Arrival_t **)pp_arrivals, 
			  p_segment->siteid,chan_num, 
			  &(p_wave_segment->p_channels[chan_num]->len),
			  &(p_wave_segment->p_channels[chan_num]->p_data),
			  NULL,
			  0);
    }



    //printf("generated signal of length %ld:\n", p_segment->p_channels[chan_num]->len);
    //print_vector(p_segment->p_channels[chan_num]->len, p_segment->p_channels[chan_num]->p_data);
  }

  free(pp_arrivals);

}

double vector_sum(int n, double * vector) {
  double result = 0;
  for (int i=0; i < n; ++i) {
    result += *(vector++);
  }
  return result;
}

void evt_arrival_times(const Event_t * p_event, int siteid, int numtimedefphases, double * first_arrival, double *last_arrival) {
  *first_arrival = DBL_MAX;
  *last_arrival = DBL_MIN;
  //  printf("called with siteid %d\n", siteid);
  fflush(stdout);
  for (int i=0; i < MAX_PHASE(numtimedefphases); ++i) {
    double phase_arr_time = (p_event->p_arrivals + (siteid-1)*numtimedefphases + i)->time;
    if (phase_arr_time <= 0) continue;
    if (phase_arr_time < *first_arrival) {
      *first_arrival = phase_arr_time;
    }
    if (phase_arr_time > *last_arrival) {
      *last_arrival = phase_arr_time;
    }
  }
}

const Event_t ** augment_events(int numevents, const Event_t ** events, const Event_t *event) {
  const Event_t ** augmented =  (const Event_t **)malloc((numevents+1)* sizeof(Event_t *));
  CHECK_PTR(augmented);
  memcpy(augmented, events, numevents*sizeof(Event_t *));
  augmented[numevents] = event;
  return augmented;
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

void AR_transitions(int n, double * ar_coeffs, int n_arrs, 
		    gsl_matrix ** pp_transition, 
		    gsl_vector ** pp_obs) {
  if (*pp_transition != NULL) {
    gsl_matrix_free(*pp_transition);
  } 
  if (*pp_obs != NULL) {
    gsl_vector_free(*pp_obs);
  }

  if (n_arrs == 0) {
    *pp_transition = NULL;
    *pp_obs = NULL;
    return;
  } 

  *pp_transition = gsl_matrix_alloc(n*n_arrs, n*n_arrs);
  *pp_obs = gsl_vector_alloc(n*n_arrs);
  gsl_matrix_set_zero(*pp_transition);
  gsl_vector_set_zero(*pp_obs);
  for (int arr=0; arr < n_arrs; ++arr) {
    gsl_matrix_set(*pp_transition, arr*n+n-1, arr*n, ar_coeffs[0]);
    for (int i=1; i < n; ++i) {
      gsl_matrix_set(*pp_transition, arr*n+i-1, arr*n+i, 1);
      gsl_matrix_set(*pp_transition, arr*n+n-1, arr*n+i, ar_coeffs[i]);
    }
    gsl_vector_set(*pp_obs, arr*n+n-1, 1);
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


void AR_update(gsl_vector * p_means, gsl_matrix * p_covars, 
	       gsl_vector * p_obs, double obs_perturb_mean, 
		 double obs_perturb_var, int n,
		 double *residual, double *residual_var) {

  double pred_obs_perturb;
  gsl_blas_ddot (p_obs, p_means, &pred_obs_perturb);
  //LogTrace("predicted perturb %lf versus obs %lf",  pred_obs_perturb, obs_perturb_mean);
  *residual = obs_perturb_mean - pred_obs_perturb;
  
  gsl_vector * k = gsl_vector_alloc(p_means->size);
  //  printf("obs: \n");
  //gsl_vector_fprintf(stdout, p_obs, "%lf");
  //printf("covars: \n");
  //gsl_matrix_fprintf(stdout, p_covars, "%lf");
  gsl_blas_dgemv (CblasNoTrans, 1, p_covars, p_obs, 0, k);
  //printf("k: \n");
  //gsl_vector_fprintf(stdout, k, "%lf");
  gsl_blas_ddot (p_obs, k, residual_var);
  //printf("r %lf, rv %lf\n", *residual, *residual_var);

  *residual_var += obs_perturb_var;
  gsl_vector_scale (k, 1/(*residual_var));
  //printf("newk: \n");
  //gsl_vector_fprintf(stdout, k, "%lf");

  gsl_blas_daxpy (*residual, k, p_means);

  gsl_matrix * km = gsl_matrix_alloc(p_means->size, 1);
  gsl_matrix_set_col (km, 0, k);

  gsl_matrix * hm = gsl_matrix_alloc(1, p_means->size);
  gsl_matrix_set_row (hm, 0, p_obs);

  gsl_matrix * I = gsl_matrix_alloc(p_means->size, p_means->size);
  gsl_matrix_set_identity (I);

  gsl_matrix * new_covar = gsl_matrix_alloc(p_means->size, p_means->size);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, -1, km, hm, 1, I);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, I, p_covars, 0, new_covar);
  gsl_matrix_memcpy(p_covars, new_covar);

  gsl_vector_free(k);
  gsl_matrix_free(km);
  gsl_matrix_free(hm);
  gsl_matrix_free(I);
  gsl_matrix_free(new_covar);
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


void abstract_env(SignalPrior_t * prior, const Arrival_t * p_arr, double hz, double ** pp_envelope, long *len) {

  double peak_height = prior->env_height * p_arr->amp;
  long peak_idx = (long) (log(peak_height) / prior->env_onset * hz);
  long end_idx = peak_idx + (long)(log(peak_height)/prior->env_decay * hz);

  if (peak_idx < 0 || end_idx <=0) {
    peak_idx = 0;
    end_idx = 1;
  }

  *len = end_idx;
  double * means = (double *) calloc(*len, sizeof(double));

  if (means == NULL) {
    printf("error allocating memory for means in abstract_env, len %ld\n", *len);
    exit(-1);
  }

  for (int i=0; i < peak_idx; ++i) {
    means[i] = exp(prior->env_onset * (i/hz));
  }
  for (int i=peak_idx; i < end_idx; ++i) {
    means[i] = exp(prior->env_decay * ((end_idx - i)/hz));
  }
  
  *pp_envelope = means;
}

/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + AR(n)
  wiggles + Gaussian iid noise signal model.
 */
double segment_likelihood_AR(SigModel_t * p_sigmodel, ChannelBundle_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals) {

  int siteid = p_segment->siteid;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);

  SignalPrior_t * prior = &p_sigmodel->sig_prior;

  double ll = 0;
  double iidll = 0;

  int n = prior->ar_n;

  ArrivalWaveform_t * st_arrivals = NULL;
  ArrivalWaveform_t * et_arrivals = NULL;

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
  noise.chan=CHAN_BHZ;

  /* populate two linked lists, storing waveform info sorted by
     start_time and end_time respectively */
  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);
      
    if (p_arr->amp == 0 || p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = 0;
    abstract_env(prior, p_arr, p_segment->hz, &w->p_envelope, &w->len);
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
  gsl_vector * p_observation = NULL;

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
      n_active++;
      AR_transitions(n, prior->p_ar_coeffs, n_active, 
		     &p_transition, &p_observation);
      starting_next->active_id = AR_add_arrival(&p_means, &p_covars, n);
      starting_next = starting_next->next_start;
    }

    // clean up any events that have finished
    while (ending_next != NULL && time >= ending_next->end_time) {
      //LogTrace(" deactivating arrivalwaveform w/ et %lf at time %lf", ending_next->end_time, time);
      active_arrivals = remove_active(active_arrivals, ending_next);
      n_active--;
      AR_transitions(n, prior->p_ar_coeffs, n_active, 
		     &p_transition, &p_observation);
      AR_remove_arrival(&p_means, &p_covars, n, ending_next->active_id);
      for (ArrivalWaveform_t * a = active_arrivals; a != NULL; a = a->next_active) {
	if (a->active_id > ending_next->active_id) {
	  //LogTrace(" remove arridx %d, decrementing %d", ending_next->active_id, a->active_id);
	  a->active_id--;
	} else {
	  //LogTrace(" remove arridx %d, not decrementing %d", ending_next->active_id, a->active_id);
	}
      }
      ending_next = ending_next->next_end;
    }


    // compute the predicted envelope for each component
    double env_bhz = prior->p_stations[siteid-1].chan_means[CHAN_BHZ];
    double env_bhe = prior->p_stations[siteid-1].chan_means[CHAN_BHE];
    double env_bhn = prior->p_stations[siteid-1].chan_means[CHAN_BHN];
    for(ArrivalWaveform_t * a = active_arrivals; a != NULL; a = a->next_active) {
      if (a->idx >= a->len) continue;
      env_bhz += a->p_envelope[a->idx] * a->bhz_coeff;
      env_bhn += a->p_envelope[a->idx] * a->bhn_coeff;
      env_bhe += a->p_envelope[a->idx] * a->bhe_coeff;
      a->idx++;
      LogTrace("getting envelope from active id %d st %lf coeffs z %lf e %lf n %lf idx %d env %lf", a->active_id, a->start_time, a->bhz_coeff, a->bhe_coeff, a->bhn_coeff, a->idx, a->p_envelope[a->idx]);
    }
    

    /* here the obs_bhe, etc. are indep observations of the
       perturbations from the predicted envelope. Our model says that
       each observation is sampled indep from a gaussian around the
       true perturbation, with variance given by the channel noise
       model. We combine them into a single observation, given by
       their mean, with correspondingly lower variance. */
    double obs_bhz, obs_bhe, obs_bhn;
    double obs_perturb_mean = 0;
    double obs_perturb_var = 0;
    int obs_perturb_n = 0;
    if (p_segment->p_channels[CHAN_BHZ] != NULL) {
	 obs_bhz = p_segment->p_channels[CHAN_BHZ]->p_data[t] - env_bhz;
	 obs_perturb_mean += obs_bhz;
	 obs_perturb_var += prior->p_stations[siteid-1].chan_vars[CHAN_BHZ];
	 
	 obs_perturb_n++;
	 LogTrace("chan bhz env %lf actual %lf diff %lf", env_bhz, p_segment->p_channels[CHAN_BHZ]->p_data[t], obs_bhz);
    }
    if (p_segment->p_channels[CHAN_BHE] != NULL) {
	 obs_bhe = p_segment->p_channels[CHAN_BHE]->p_data[t] - env_bhe;
	 obs_perturb_mean += obs_bhe;
	 obs_perturb_var += prior->p_stations[siteid-1].chan_vars[CHAN_BHE];
	 obs_perturb_n++;
	 LogTrace("chan bhe env %lf actual %lf diff %lf", env_bhe, p_segment->p_channels[CHAN_BHE]->p_data[t], obs_bhe);
    }
    if (p_segment->p_channels[CHAN_BHN] != NULL) {
	 obs_bhn = p_segment->p_channels[CHAN_BHN]->p_data[t] - env_bhn;
	 obs_perturb_mean += obs_bhn;
	 obs_perturb_var += prior->p_stations[siteid-1].chan_vars[CHAN_BHN];
	 obs_perturb_n++;
	 LogTrace("chan bhn env %lf actual %lf diff %lf", env_bhn, p_segment->p_channels[CHAN_BHN]->p_data[t], obs_bhn);
    }
    obs_perturb_mean /= obs_perturb_n;
    obs_perturb_var /= obs_perturb_n*obs_perturb_n;
    LogTrace(" perturb mean %lf var %lf n %d", obs_perturb_mean, obs_perturb_var, obs_perturb_n);
    
    double residual = obs_perturb_mean;
    double rvar = obs_perturb_var;
    if (n_active > 0) {
      AR_predict(p_means, p_covars, p_transition, prior->ar_noise_sigma2, n);
      AR_update(p_means, p_covars, p_observation, obs_perturb_mean, obs_perturb_var, n, &residual, &rvar);
    } 
    double thisll = 0.5 * log(2*PI * rvar) + 0.5 * residual*residual / rvar;
    double thisiidll =  0.5 * log(2*PI * obs_perturb_var) + 0.5 * obs_perturb_mean*obs_perturb_mean / obs_perturb_var;
    assert(!isnan(thisll) && thisll > -1*DBL_MAX);
    ll -= thisll;
    iidll -= thisiidll;
    LogTrace(" ll minus %lf is %lf (residual %lf var %lf tiidll %lf iidll %lf)", thisll, ll, residual, rvar, thisiidll, iidll);


    /* obs perturb mean is avg distance from envelope 
       residual is difference between that, and predicted perturbation.
       so obs-residual gives predicted perturbation
       whereas residual gives noise, plus perturb model error
 */

    filtered_perturbs.p_data[t] = obs_perturb_mean-residual;
    filtered_waveform.p_data[t] = env_bhz + obs_perturb_mean-residual;
    noise.p_data[t] = residual;

  }

  for(ArrivalWaveform_t * a = st_arrivals; a != NULL; a = a->next_start) {
    free(a->p_envelope);
    free(a);
  }

  if (p_means != NULL) gsl_vector_free(p_means);
  if (p_covars != NULL) gsl_matrix_free(p_covars);
  if (p_transition != NULL) gsl_matrix_free(p_transition);
  if (p_observation != NULL) gsl_vector_free(p_observation);

  save_pdf_plot(p_sigmodel, &filtered_perturbs, "filtered_perturbs");
  save_pdf_plot(p_sigmodel, &filtered_waveform, "filtered_waveform");
  save_pdf_plot(p_sigmodel, &noise, "noise");

  free(filtered_perturbs.p_data);
  free(filtered_waveform.p_data);

  LogTrace ("returning ar ll %lf vs iid ll %lf", ll, iidll);

  return ll;
}

/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + Gaussian
  iid wiggles + Gaussian iid noise signal model.
 */
double segment_likelihood_iid(SigModel_t * p_sigmodel, ChannelBundle_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals) {

  double * p_means, * p_vars;
  long len;
  SignalPrior_t * prior = &p_sigmodel->sig_prior;

  double event_lp = 0;
  for (int chan_num = 0; chan_num < NUM_CHANS; ++chan_num) {
    if (p_segment->p_channels[chan_num] == NULL) continue;
    envelope_means_vars(prior, p_segment->hz, p_segment->start_time, ChannelBundle_EndTime(p_segment), p_sigmodel->p_earth, num_arrivals, pp_arrivals, p_segment->siteid, chan_num, &len, &p_means, &p_vars, 1);
    
    event_lp += indep_Gaussian_LogProb(len, p_segment->p_channels[chan_num]->p_data, p_means, p_vars);

    if(isnan(event_lp) || event_lp < -1 * DBL_MAX) {
      printf(" NAN clen %ld means\n", len);
      print_vector(len, p_means);
      printf(" vars\n");
      print_vector(len, p_vars);
      printf(" data from seg (siteid %d chan %d) starting at idx %ld\n", p_segment->siteid, chan_num, 0l);
      print_vector(len, p_segment->p_channels[chan_num]->p_data);
      exit(-1);
    }
    
    assert (!isnan(event_lp));
    free(p_means);
    free(p_vars);
  }
  return event_lp;
}



double det_likelihood(void * p_sigmodel_v, double env_height, double env_decay, double env_offset) {

  LogDebug("called dl with %lf %lf %lf", env_height, env_decay, env_offset);

  SigModel_t * p_sigmodel = (SigModel_t *) p_sigmodel_v;

  double backup_env_height = p_sigmodel->sig_prior.env_height;
  double backup_env_decay = p_sigmodel->sig_prior.env_decay;
  double backup_env_onset = p_sigmodel->sig_prior.env_onset;

  p_sigmodel->sig_prior.env_height = env_height;
  p_sigmodel->sig_prior.env_height = env_decay;
  p_sigmodel->sig_prior.env_height = env_offset;

  double ll = 0;

  for (int i=0; i < p_sigmodel->numsegments; ++i) {
    
    ChannelBundle_t * p_segment = p_sigmodel->p_segments + i;

    // for each segment, compute a list of arrivals
    int num_arrivals=0;
    int num_alloced = 20;
    Arrival_t ** pp_arrivals = calloc(num_alloced, sizeof(Arrival_t *));
    CHECK_PTR(pp_arrivals);

    for (int d = 0; d < p_sigmodel->numdetections; ++d) {
      Detection_t * p_det = p_sigmodel->p_detections + d;

      if (p_det->site_det != p_segment->siteid-1) continue;
      if (p_det->time_det + MAX_ENVELOPE_LENGTH < p_segment->start_time) continue;
      if (p_det->time_det > ChannelBundle_EndTime(p_segment)) continue;

      if (++num_arrivals > num_alloced) {
	num_alloced *= 2;
	pp_arrivals = realloc(pp_arrivals, num_alloced * sizeof(Arrival_t *));
	CHECK_PTR(pp_arrivals);
      }

      Arrival_t * p_arr = calloc(1, sizeof(Arrival_t));
      *(pp_arrivals+num_arrivals-1) = p_arr;
      p_arr->time = p_det->time_det;
      p_arr->amp = p_det->amp_det;
      p_arr->azi = p_det->azi_det;
      p_arr->slo = p_det->slo_det;
      p_arr->phase = p_det->phase_det;
    }

    double seg_ll = segment_likelihood_AR(p_sigmodel, p_segment, num_arrivals, (const Arrival_t **)pp_arrivals);
    double seg_ll_iid = segment_likelihood_iid(p_sigmodel, p_segment, num_arrivals, (const Arrival_t **)pp_arrivals);
    assert(!isnan(seg_ll) && seg_ll > -1*DBL_MAX);

    LogTrace("segment site %d start %lf contributed ll %lf iidll %lf from %d arrivals", p_segment->siteid, p_segment->start_time, seg_ll, seg_ll_iid,  num_arrivals);

    ll += seg_ll;

    for(int i=0; i < num_arrivals; ++i) {
      free(*(pp_arrivals+i));
    }
    free(pp_arrivals);
  }

  p_sigmodel->sig_prior.env_height = backup_env_height;
  p_sigmodel->sig_prior.env_decay = backup_env_decay;
  p_sigmodel->sig_prior.env_onset = backup_env_onset;

  printf("returning %lf\n", ll);
  return ll;
}



/* Return the score for this event: the ratio of signal likelihoods
   between a world where this event exists, and one where it
   doesn't. */
double SignalPrior_Score_Event_Site(SignalPrior_t * prior, void * p_sigmodel_v, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events) {
  SigModel_t * p_sigmodel = (SigModel_t *) p_sigmodel_v;

  double score = 0;
  ChannelBundle_t * p_segment;

  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);

  for (int i=0; i < p_sigmodel->numsegments; ++i) {

    p_segment = p_sigmodel->p_segments + i;

    if (p_segment->siteid != siteid)  {
      continue;
    }
    //printf("scoring event at siteid %d...\n", siteid);
    /* compute the time period during which the event will affect the station */
    double first_envelope_time, last_envelope_time;
    evt_arrival_times(p_event, siteid, numtimedefphases, &first_envelope_time, &last_envelope_time);
    last_envelope_time += MAX_ENVELOPE_LENGTH;

    /* if this trace doesn't fall within that period, skip it */
    if (p_segment->start_time > last_envelope_time || ChannelBundle_EndTime(p_segment) < first_envelope_time) {
      //printf("     skipping signal segment %d: first env %lf last env %lf seg start %lf seg end %lf\n", i, first_envelope_time, last_envelope_time, p_segment->start_time, ChannelBundle_EndTime(p_segment));
      continue;
    }

    /* we compute scores for the background event set, and for an
       augmented event set which includes the specified event. */
    
    int num_basic_arrivals, num_augmented_arrivals;
    Arrival_t ** pp_basic_arrivals;
    Arrival_t ** pp_augmented_arrivals;
    arrival_list(p_sigmodel->p_earth, siteid, p_segment->start_time, ChannelBundle_EndTime(p_segment), num_other_events, pp_other_events, &num_basic_arrivals, &pp_basic_arrivals);

    const Event_t ** augmented_events = augment_events(num_other_events, pp_other_events, p_event);
    arrival_list(p_sigmodel->p_earth, siteid, p_segment->start_time, ChannelBundle_EndTime(p_segment), num_other_events+1, augmented_events, &num_augmented_arrivals, &pp_augmented_arrivals);

    double event_lp , background_lp;
    if (p_sigmodel->ar_perturbation) {
      event_lp = segment_likelihood_AR(p_sigmodel, p_segment, num_augmented_arrivals, (const Arrival_t **)pp_augmented_arrivals);
      background_lp = segment_likelihood_AR(p_sigmodel, p_segment, num_basic_arrivals, (const Arrival_t **)pp_basic_arrivals);
    } else {
      event_lp = segment_likelihood_iid(p_sigmodel, p_segment, num_augmented_arrivals, (const Arrival_t **)pp_augmented_arrivals);
      background_lp = segment_likelihood_iid(p_sigmodel, p_segment, num_basic_arrivals, (const Arrival_t **)pp_basic_arrivals);
    }
    free(augmented_events);
    free(pp_basic_arrivals);
    free(pp_augmented_arrivals);

    score += (event_lp - background_lp);

    LogTrace("   segment %d contributed score %lf = event_lp %lf - background_lp %lf (perturb %d)", i, event_lp - background_lp, event_lp, background_lp, p_sigmodel->ar_perturbation);

  }

  return score;

}

/* Return the score for this event: the ratio of signal likelihoods
   between a world where this event exists, and one where it
   doesn't. */
double SignalPrior_Score_Event(SignalPrior_t * prior, void * p_sigmodel_v, const Event_t * p_event, int num_other_events, const Event_t ** pp_other_events) {

  SigModel_t * p_sigmodel = (SigModel_t *) p_sigmodel_v;
  int numsites = EarthModel_NumSites(p_sigmodel->p_earth);

  double score = 0;
  for (int siteid = 1; siteid <= numsites; ++siteid) {
    score += SignalPrior_Score_Event_Site(prior, p_sigmodel_v, p_event, 
					  siteid, num_other_events, 
					  pp_other_events);
  }

  return score;
}

void SignalPrior_UnInit(SignalPrior_t * prior) {
  free(prior->p_stations);
  free(prior->p_ar_coeffs);
}
