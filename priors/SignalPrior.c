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

  if (2 != fscanf(fp, "%lf %lf\n", &prior->env_height, &prior->env_decay))
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
      printf(" set %d, %d, vars to %lf\n", i, j, (prior->p_stations + i)->chan_vars[j]);
    }
  }

  int numentries;
  if (1 != fscanf(fp, "%d\n", &numentries))
  {
    fprintf(stderr, "error reading num entries from %s\n", filename);
    exit(1);
  }

  for (int i=0; i < numentries; ++i) {

    int siteid;
    int num_chans;
    if (2 != fscanf(fp, "%d %d", &siteid, &num_chans)) {
      fprintf(stderr, "error reading siteid and num_chans from station line %d of %s\n", i, filename);
      exit(1);
    }

    for (int j=0; j < num_chans; ++j) {
      int chan_num;
      double mean, var;
      if (3 != fscanf(fp, " %d %lf %lf", &chan_num, &mean, &var))
	{
	  fprintf(stderr, "error reading mean and variance for site %d from %s\n", i, filename);
	  exit(1);
	}
    
      fprintf(stdout, "%d: loaded %d %d %lf %lf\n", i, siteid, chan_num, mean, var);
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

  //fprintf(stdout, "time2idx: %lf %lf %lf = %ld\n", t, start_time, hz, result);
  return result;
}

double indep_Gaussian_LogProb(int n, double * x, double * means, double * vars) {

  double lp = 0;

  for (int i=0; i < n; ++i) {
    assert(vars[i] >= 0);
    if (isnan(x[i])) continue;
    lp -= 0.5 * log(2*PI * vars[i]) + 0.5 * (x[i] - means[i])*(x[i] - means[i]) / vars[i];
  }

  return lp;
}


void phase_env(SignalPrior_t * prior, 
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

      const Arrival_t * p_arr = p_event->p_arrivals + siteid*p_earth->numtimedefphases + phaseid;

      //    printf("event %d at siteid %d, ratios n/z %lf e/z %lf\n", i, siteid, SPHERE2Y(p_arr->azi, p_arr->slo)/SPHERE2Z(p_arr->azi, p_arr->slo), SPHERE2X(p_arr->azi, p_arr->slo)/SPHERE2Z(p_arr->azi, p_arr->slo)   );

      double arrtime = p_arr->time;
      if (arrtime < 0) continue;
    
      long idx = time2idx(arrtime, start_time, hz);

      // skip events which don't arrive during the current time period
      if (idx < 0 - MAX_ENV_LENGTH * hz || idx >= *p_len) {
	continue;
      }

      double * p_envelope;
      int env_len;
      phase_env(prior, p_arr, hz, chan_num, &p_envelope, &env_len);
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
  for (int i=0; i < MAX_PHASE(numtimedefphases); ++i) {
    double phase_arr_time = (p_event->p_arrivals + siteid*numtimedefphases + i)->time;
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

    /* compute the time period during which the event will affect the station */
    double first_envelope_time, last_envelope_time;
    evt_arrival_times(p_event, siteid, numtimedefphases, &first_envelope_time, &last_envelope_time);
    last_envelope_time += MAX_ENVELOPE_LENGTH;

    /* if this trace doesn't fall within that period, skip it */
    if (p_segment->start_time > last_envelope_time || ChannelBundle_EndTime(p_segment) < first_envelope_time) {
      //printf("     skipping signal segment %d: first env %lf last env %lf seg start %lf seg end %lf\n", i, first_envelope_time, last_envelope_time, p_segment->start_time, ChannelBundle_EndTime(p_segment));
      continue;
    }

    /* index within this trace at which the event arrives */
    long env_start_idx = time2idx(first_envelope_time, p_segment->start_time, p_segment->hz);
    
    /* we compute scores for the background event set, and for an
       augmented event set which includes the specified event. */
    const Event_t ** augmented_events = augment_events(num_other_events, pp_other_events, p_event);
    
    double * p_means, * p_vars;
    long len;

    /* score augmented event set */
    double event_lp = 0;

    for (int chan_num = 0; chan_num < NUM_CHANS; ++chan_num) {
      if (p_segment->p_channels[chan_num] == NULL) continue;
      envelope_means_vars(prior, p_segment->hz, first_envelope_time, last_envelope_time, p_sigmodel->p_earth, num_other_events+1, augmented_events, siteid, chan_num, &len, &p_means, &p_vars, 1);
      long compare_len = MIN(len, p_segment->len - env_start_idx);
    

      event_lp += indep_Gaussian_LogProb(compare_len, p_segment->p_channels[chan_num]->p_data + env_start_idx, p_means, p_vars);

      if(isnan(event_lp)) {
	printf(" NAN clen %ld means\n", compare_len);
	print_vector(compare_len, p_means);
	printf(" vars\n");
	print_vector(compare_len, p_vars);
	printf(" data\n");
	print_vector(compare_len, p_segment->p_channels[chan_num]->p_data + env_start_idx);
	exit(-1);
      }

      assert (!isnan(event_lp));
      free(p_means);
      free(p_vars);
    }
    free(augmented_events);
    
    /* score background event set */
    double background_lp = 0;
    for (int chan_num = 0; chan_num < NUM_CHANS; ++chan_num) {
      if (p_segment->p_channels[chan_num] == NULL) continue;
      envelope_means_vars(prior, p_segment->hz, first_envelope_time, last_envelope_time, p_sigmodel->p_earth, num_other_events, pp_other_events, siteid, chan_num, &len, &p_means, &p_vars, 1);
      long compare_len = MIN(len, p_segment->len - env_start_idx);
      background_lp +=  indep_Gaussian_LogProb(compare_len, p_segment->p_channels[chan_num]->p_data + env_start_idx, p_means, p_vars);
      assert (!isnan(background_lp));
      free(p_means);
      free(p_vars);
    }

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
  for (int siteid = 0; siteid < numsites; ++siteid) {
    score += SignalPrior_Score_Event_Site(prior, p_sigmodel_v, p_event, siteid, num_other_events, pp_other_events);
  }

  return score;
}

void SignalPrior_UnInit(SignalPrior_t * prior) {
  free(prior->p_stations);
}
