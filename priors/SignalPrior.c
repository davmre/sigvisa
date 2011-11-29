#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "../sigvisa.h"

#define MAX_ENV_LENGTH 100

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
      (prior->p_stations + siteid)->chan_means[chan_num] = mean;
      (prior->p_stations + siteid)->chan_vars[chan_num] = var;
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
  long result = lround(delta_t * hz);

  return result;
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
  double iangle = asin(0.05215*p_arr->slo);
  LogTrace("iangle %lf slowness %lf", iangle, p_arr->slo);
  switch (chan_num) {
  case CHAN_BHE:
    component_coeff = SPHERE2X(p_arr->azi, iangle); break;
  case CHAN_BHN:
    component_coeff = SPHERE2Y(p_arr->azi, iangle); break;
  case CHAN_BHZ:
    component_coeff = SPHERE2Z(p_arr->azi, iangle); break;
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

void phase_env_triangle(SignalPrior_t * prior, 
	       const Arrival_t * p_arr, 
	       double hz,
	       int chan_num,
	       double ** p_envelope,
	       int * t) {

  double newmean = prior->env_height * p_arr->amp;

  double component_coeff = 0;
  switch (chan_num) {
  case CHAN_BHE:
    component_coeff = SPHERE2X(p_arr->azi, p_arr->slo); break;
  case CHAN_BHN:
    component_coeff = SPHERE2Y(p_arr->azi, p_arr->slo); break;
  case CHAN_BHZ:
    component_coeff = SPHERE2Z(p_arr->azi, p_arr->slo); break;
  }

  //  printf("generating event signal with arrival azi %lf and slo %lf\n", p_arr->azi, p_arr->slo);
  //  printf("channel is %d and ratio is %lf\n", chan_num, component_coeff);

  double step = (prior->env_decay / hz);
  unsigned long len = ceil( newmean / step );

  /*if (len >= 30 * hz) {
    printf("event lasting more than 30 seconds! env_decay = %lf, hz = %lf, step = %lf, amp = %lf, env_height = %lf, newmean = %lf, len = %ld\n", prior->env_decay, hz, step, p_arr->amp, prior->env_height, newmean, len);
    step = newmean /(30.0 * hz);
    len = ceil( newmean / step );
    printf("resetting step to %lf , len to %ld\n", step, len);
    }*/

  double * means = (double *) calloc(len, sizeof(double));

  if (means == NULL) {
    printf("error allocating memory for means in phase_env, len %ld = %lf / %lf\n", len, newmean, step);
    exit(-1);
  }

  for (int i=0; i < len; ++i) {
    means[i] = newmean * component_coeff;
    newmean -= step;
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
			 int numevents, 
			 const Event_t ** pp_events, 
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

  double noise_mean = (prior->p_stations + siteid)->chan_means[chan_num];
  double noise_var = (prior->p_stations + siteid)->chan_vars[chan_num];
  for (int i=0; i < *p_len; ++i) {
    p_means[i] = noise_mean;
    if (pp_vars != NULL) p_vars[i] = noise_var;
  }

  for (int i=0; i < numevents; ++i) {
    
    const Event_t * p_event = pp_events[i];

    for (int phaseid = 0; phaseid < MAX_PHASE(p_earth->numtimedefphases); ++phaseid) {    

      const Arrival_t * p_arr = p_event->p_arrivals + (siteid-1)*p_earth->numtimedefphases + phaseid;

      //    printf("event %d at siteid %d, ratios n/z %lf e/z %lf\n", i, siteid, SPHERE2Y(p_arr->azi, p_arr->slo)/SPHERE2Z(p_arr->azi, p_arr->slo), SPHERE2X(p_arr->azi, p_arr->slo)/SPHERE2Z(p_arr->azi, p_arr->slo)   );

      double arrtime = p_arr->time;
      if (arrtime < 0) continue;
    
      long idx = time2idx(arrtime, start_time, hz);

      // skip events which don't arrive during the current time period
      if (idx < 0 - MAX_ENV_LENGTH * hz || idx >= *p_len) {
	//printf("skipping event arrival at time %lf vs start_time %lf idx %ld\n", arrtime, start_time, idx);
	continue;
      }

      double * p_envelope;
      int env_len;
      phase_env_doubleexp(prior, p_arr, hz, chan_num, &p_envelope, &env_len);
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

  for (int i=0; i < 3; ++i) {
    
    int chan_num = chan_nums[i];
    
    p_segment->p_channels[chan_num] = alloc_signal(p_segment);
    p_segment->p_channels[chan_num]->chan = chan_num;
    // printf("tae calling envelope_means_vars(%lf, %lf, %lf, %d, %d, %d)\n", p_segment->hz, p_segment->start_time, end_time, numevents, p_segment->siteid, chan_num);
    envelope_means_vars(prior, 
			p_segment->hz, p_segment->start_time, end_time,
			p_earth, numevents, pp_events, 
			p_segment->siteid,chan_num, 
			&(p_segment->p_channels[chan_num]->len),
			&(p_segment->p_channels[chan_num]->p_data),
			NULL,
			1);
    assert(p_segment->p_channels[chan_num]->len == p_segment->len);

    //printf("segment siteid is %d\n", p_segment->siteid);

    if (p_wave_segment != NULL) {
      p_wave_segment->p_channels[chan_num] = alloc_signal(p_wave_segment);
      p_wave_segment->p_channels[chan_num]->chan = chan_num;
      envelope_means_vars(prior, 
			  p_segment->hz, p_segment->start_time, end_time,
			  p_earth, numevents, pp_events, 
			  p_segment->siteid,chan_num, 
			  &(p_wave_segment->p_channels[chan_num]->len),
			  &(p_wave_segment->p_channels[chan_num]->p_data),
			  NULL,
			  0);
    }



    //printf("generated signal of length %ld:\n", p_segment->p_channels[chan_num]->len);
    //print_vector(p_segment->p_channels[chan_num]->len, p_segment->p_channels[chan_num]->p_data);
  }

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
    if (phase_arr_time < 0) continue;
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

double **ptrvector(long n)  {
  double **v;
  v=(double **)malloc((size_t) (n*sizeof(double)));
  if (!v)   {
    printf("In **ptrvector. Allocation of memory for double array failed.");
    exit(0);  }
  return v;
}
void free_Carrayptrs(double **v)  {
  free((char*) v);
}
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
     int n;
     n=arrayin->dimensions[0];
     return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
      double **c, *a;
      int i,n,m;
      
      n=arrayin->dimensions[0];
      m=arrayin->dimensions[1];
      c=ptrvector(n);
      a=(double *) arrayin->data;  /* pointer to arrayin data as double */
      for ( i=0; i<n; i++)  {
          c[i]=a+i*m;  }
      return c;
}

int AR_add_arrival(PyArrayObject ** pp_means, PyArrayObject ** pp_covars, int n) {

  int arridx = 0;

  if (*pp_means == NULL || *pp_covars == NULL) {

    npy_intp mean_dims[1];
    mean_dims[0] = n;
    *pp_means = PyArray_GETCONTIGUOUS(PyArray_ZEROS(1, mean_dims, NPY_DOUBLE, 0));

    npy_intp covar_dims[2];
    covar_dims[0] = n;
    covar_dims[1] = n;
    *pp_covars = PyArray_GETCONTIGUOUS(PyArray_ZEROS(2, covar_dims, NPY_DOUBLE, 0));
    double **covars = pymatrix_to_Carrayptrs(*pp_covars);
    for (int i=0; i < n; ++i) {
      covars[i][i] = 1;
    }
    free_Carrayptrs(covars);
  } else {

    int ndim = PyArray_NDIM(*pp_means);
    assert(ndim==1);
    PyArray_Dims newdims;
    newdims.len=ndim;
    newdims.ptr = calloc(ndim, sizeof(npy_intp));
    CHECK_ALLOC(newdims.ptr);
    npy_intp *mean_dims = PyArray_DIMS(*pp_means);
    arridx = mean_dims[0]/n;
    newdims.ptr[0] = mean_dims[0]+n;
    *pp_means=(PyArrayObject*)PyArray_Resize(*pp_means, &newdims, 1, 0);
    free(newdims.ptr);

    ndim = PyArray_NDIM(*pp_covars);
    assert(ndim==2);
    newdims.len=ndim;
    newdims.ptr = calloc(ndim, sizeof(npy_intp));
    CHECK_ALLOC(newdims.ptr);
    npy_intp *covar_dims = PyArray_DIMS(*pp_covars);
    assert(covar_dims[0] == covar_dims[1]);
    newdims.ptr[0] = covar_dims[0]+n;
    newdims.ptr[1] = covar_dims[1]+n;
    *pp_covars=(PyArrayObject*)PyArray_Resize(*pp_covars, &newdims, 1, 0);
    free(newdims.ptr);
    double **covars = pymatrix_to_Carrayptrs(*pp_covars);
    for (int i=covar_dims[0]-n; i < covar_dims[0]; ++i) {
      covars[i][i] = 1;
    }
    free_Carrayptrs(covars);
  }
  return arridx;
}

void AR_remove_arrival(PyArrayObject ** pp_means, PyArrayObject ** pp_covars, int n, int arridx) {
  if (*pp_means == NULL || *pp_covars == NULL) {
    return;
  } 

  int ndim = PyArray_NDIM(*pp_means);
  assert(ndim==1);
  npy_intp *mean_dims = PyArray_DIMS(*pp_means);
  if (mean_dims[0] == n) {
    Py_DECREF(*pp_means);
    Py_DECREF(*pp_covars);
    *pp_means = NULL;
    *pp_covars = NULL;
  } else {

    PyArrayObject * p_means = PyArray_GETCONTIGUOUS(*pp_means);
    PyArrayObject * p_covars = PyArray_GETCONTIGUOUS(*pp_covars);

    double * means = pyvector_to_Carrayptrs(p_means);
    for(int i=(arridx+1)*n; i < mean_dims[0]; ++i) {
      means[i-n] = means[i]; 
    }
    PyArray_Dims newdims;
    newdims.len=ndim;
    newdims.ptr = calloc(ndim, sizeof(npy_intp));
    CHECK_ALLOC(newdims.ptr);
    newdims.ptr[0] = mean_dims[0]-n;
    *pp_means=(PyArrayObject*)PyArray_Resize(p_means, &newdims, 1, 0);
    free(newdims.ptr);    
    free(means);

    ndim = PyArray_NDIM(p_covars);
    assert(ndim==2);
    npy_intp *covar_dims = PyArray_DIMS(p_covars);
    assert(covar_dims[0] == covar_dims[1]);
    double **covars = pymatrix_to_Carrayptrs(p_covars);
    // move everything up by n rows
    for (int i=(arridx+1)*n; i < covar_dims[0]; ++i) {
      for(int j=0; j < covar_dims[1]; ++j) {
	covars[i-n][j] = covars[i][j];
      }
    }

    // move everything in by n columns
    for (int i=0; i < covar_dims[0]; ++i) {
      for(int j=(arridx+1)*n; j < covar_dims[1]; ++j) {
	covars[i][j-n] = covars[i][j];
      }
    }

    newdims.len=ndim;
    newdims.ptr = calloc(ndim, sizeof(npy_intp));
    CHECK_ALLOC(newdims.ptr);
    newdims.ptr[0] = covar_dims[0]-n;
    newdims.ptr[1] = covar_dims[1]-n;
    *pp_covars=(PyArrayObject*)PyArray_Resize(p_covars, &newdims, 1, 0);
    free(newdims.ptr);
    free_Carrayptrs(covars);
  }
}



void AR_predict(PyArrayObject ** pp_means, PyArrayObject ** pp_covars, PyArrayObject * p_transition, PyArrayObject * p_transition_T, double noise_sigma2, int n) {
  PyArrayObject * p_new_means = PyArray_MatrixProduct((PyObject*)p_transition, (PyObject*)*pp_means);
  PyArrayObject * p_new_covars = PyArray_MatrixProduct((PyObject*)p_transition, PyArray_MatrixProduct((PyObject*)*pp_covars, (PyObject*)p_transition_T));
  
  int ndim = PyArray_NDIM(*pp_covars);
  assert(ndim==2);
  npy_intp *covar_dims = PyArray_DIMS(*pp_covars);
  assert(covar_dims[0]==covar_dims[1]);
  assert(covar_dims[0]%n==0);
  npy_intp num_arrs = covar_dims[0]/n;

  double ** ptr;
  PyArray_AsCArray((PyObject **) pp_covars, (void *)&ptr, covar_dims, ndim, PyArray_DescrFromType(NPY_DOUBLE));
  for(int i=0; i < num_arrs; ++i) {
    int idx = i*n-1;
    ptr[idx][idx] = ptr[idx][idx]+noise_sigma2;
  }


  Py_DECREF(*pp_means);
  Py_DECREF(*pp_covars);
  *pp_means = p_new_means;
  *pp_covars = p_new_covars;
}

/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + AR(n)
  wiggles + Gaussian iid noise signal model.
 */
double segment_likelihood_AR(SigModel_t * p_sigmodel, SignalPrior_t * prior, ChannelBundle_t * p_segment, int num_events, const Event_t ** pp_events) {

  double * p_means, * p_vars;
  long len;

  

  double event_lp = 0;
  for (int chan_num = 0; chan_num < NUM_CHANS; ++chan_num) {
    if (p_segment->p_channels[chan_num] == NULL) continue;
    envelope_means_vars(prior, p_segment->hz, p_segment->start_time, ChannelBundle_EndTime(p_segment), p_sigmodel->p_earth, num_events, pp_events, p_segment->siteid, chan_num, &len, &p_means, &p_vars, 1);
    
    event_lp += indep_Gaussian_LogProb(len, p_segment->p_channels[chan_num]->p_data, p_means, p_vars);

    if(isnan(event_lp) || event_lp < -1 * DBL_MAX) {
      printf(" NAN clen %ld means\n", len);
      print_vector(len, p_means);
      printf(" vars\n");
      print_vector(len, p_vars);
      printf(" data from seg  (siteid %d chan %d) starting at idx %ld\n", p_segment->siteid, chan_num, 0l);
      print_vector(len, p_segment->p_channels[chan_num]->p_data);
      exit(-1);
    }
    
    assert (!isnan(event_lp));
    free(p_means);
    free(p_vars);
  }
  return event_lp;
}

/*
  Return the likelihood of the given signal segment (three channels at
  some station over some time period), under the envelope + Gaussian
  iid wiggles + Gaussian iid noise signal model.
 */
double segment_likelihood_iid(SigModel_t * p_sigmodel, SignalPrior_t * prior, ChannelBundle_t * p_segment, int num_events, const Event_t ** pp_events) {

  double * p_means, * p_vars;
  long len;

  double event_lp = 0;
  for (int chan_num = 0; chan_num < NUM_CHANS; ++chan_num) {
    if (p_segment->p_channels[chan_num] == NULL) continue;
    envelope_means_vars(prior, p_segment->hz, p_segment->start_time, ChannelBundle_EndTime(p_segment), p_sigmodel->p_earth, num_events, pp_events, p_segment->siteid, chan_num, &len, &p_means, &p_vars, 1);
    
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
    const Event_t ** augmented_events = augment_events(num_other_events, pp_other_events, p_event);
    double event_lp = segment_likelihood_iid(p_sigmodel, prior, p_segment, num_other_events+1, augmented_events);
    free(augmented_events);

    double background_lp = segment_likelihood_iid(p_sigmodel, prior, p_segment, num_other_events, pp_other_events);
    score += (event_lp - background_lp);

    //printf("   segment %d contributed score %lf = event_lp %lf - background_lp %lf\n", i, event_lp - background_lp, event_lp, background_lp);

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
    score += SignalPrior_Score_Event_Site(prior, p_sigmodel_v, p_event, siteid, num_other_events, pp_other_events);
  }

  return score;
}

void SignalPrior_UnInit(SignalPrior_t * prior) {
  free(prior->p_stations);
}
