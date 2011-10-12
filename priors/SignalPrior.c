#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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
  prior->p_stations = (double *) calloc(numsites, sizeof(StationNoiseModel_t));
  int numentries;
  if (1 != fscanf(fp, "%d\n", &numentries))
  {
    fprintf(stderr, "error reading num entries from %s\n", filename);
    exit(1);
  }

  for (int i=0; i < numentries; ++i) {

    int siteid;
    int num_chans;
    if (1 != fscanf(fp, "%d %d", &siteid, &num_chans)) {
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
      prior->p_stations[siteid]->chan_means[chan_num] = mean;
      prior->p_stations[siteid]->chan_vars[chan_num] = var;
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
    lp -= 0.5 * log(2*PI * vars[i]) + 0.5 * (x[i] - means[i])*(x[i] - means[i]) / vars[i];
  }

  return lp;
}


void phase_env(SignalPrior_t * prior, 
	       EarthModel_t * p_earth, 
	       Event_t * event, 
	       double hz,
	       int siteid,
	       int phaseid,
	       int chan_num,
	       double ** p_envelope,
	       int * t) {

  // TODO: make use of chanid

  double distance_deg = EarthModel_Delta(p_earth, event->evlon, event->evlat, siteid);
  double distance_mi = 12000 * (distance_deg/180);

  double newmean = prior->env_height / distance_mi * exp(event->evmag);

  double step = (prior->env_decay / hz);
  int len = ceil( newmean / step );
  double * means = (double *) calloc(len, sizeof(double));
  for (int i=0; i < len; ++i) {
    means[i] = newmean;
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
			 Event_t ** pp_events, 
			 int siteid,
			 int chan_num, 
			 long * p_len,
			 double ** pp_means,
			 double ** pp_vars) {
 

  p_len = time2idx(end_time, start_time, hz);

  (*pp_means) = (double *) calloc(p_len, sizeof(double));
  double *p_means = (*pp_means);
  (*pp_vars) = (double *) calloc(p_len, sizeof(double));
  double *p_vars = (*pp_vars);

  double noise_mean = prior->p_stations[siteid]->chan_means[chan_num];
  double noise_var = prior->p_stations[siteid]->chan_means[chan_num];
  for (int i=0; i < p_len; ++i) {
    p_means[i] = noise_mean;
    p_vars[i] = noise_var;
  }

  for (int i=0; i < numevents; ++i) {

    Event_t * p_event = pp_events[i];

    // for (int phaseid = 0; phaseid < p_earth->num_phases; ++phaseid) {    
    int phaseid = 0; /* TODO: work with multiple phases */

    double arrtime = p_event->p_arrivals[siteid*p_earth->numtimedefphases + phaseid];
    if (arrtime < 0) continue;
    
    long idx = time2idx(arrtime, start_time, hz);
    if (idx < 0 - MAX_ENV_LENGTH * hz || idx >= p_len) {
      //      fprintf(stdout, "   skipping event w/ arrtime %lf (from st %lf) and base_t = %ld ( vs len %d) ...\n", arrtime, (*(events+i)).evtime, idx, p_signal->len);      
      continue;
    } else {
      //      print_event(events+i);
      // fprintf(stdout, "  event w/ arrtime %lf (from st %lf) and base_t = %ld\n", arrtime, (*(events+i)).evtime, idx);     
    }

    double * p_envelope;
    int env_len;
    phase_env(prior, p_earth, p_event, hz, siteid, phaseid, chan_num, &p_envelope, &env_len);
    
    add_signals(p_means, p_len, p_envelope, env_len, idx);
    vector_times_scalar_inplace(len, p_envelope, 0.5);
    add_signals(p_vars, p_len, p_envelope, env_len, idx);
      
    free(p_envelope);
    //}
  }

  
}



double vector_sum(int n, double * vector) {
  double result = 0;
  for (int i=0; i < n; ++i) {
    result += *(vector++);
  }
  return result;
}

void evt_arrival_times(Event_t * p_event, int siteid, int numtimedefphases, double * first_arrival, double *last_arrival) {
  *first_arrival = DBL_MAX;
  *last_arrival = DBL_MIN;
  for (int i=0; i < numtimedefphases; ++i) {
    double phase_arr_time = p_event->p_arrivals[siteid*numtimedefphases + i]->time;
    if (phase_arr_time < *first_arrival) {
      *first_arrival = phase_arr_time;
    }
    if (phase_arr_time > *last_arrival) {
      *last_arrival = phase_arr_time;
    }
  }
}

Event_t ** augment_events(int numevents, Event_t ** events, Event_t *event) {
  Event_t ** augmented =  (Event_t **)malloc((numevents+1)* sizeof(Event *));
  CHECK_PTR(augmented);
  memcpy(augmented, events, numevents*sizeof(Event *));
  augmented[numevents] = event;
  return augmented;
}

/* Return the score for this event: the ratio of signal likelihoods
   between a world where this event exists, and one where it
   doesn't. */
double SignalPrior_Score_Event(SignalPrior_t * prior, SigModel * p_sigmodel, Event_t * event, int num_other_events, Events ** pp_other_events) {

  double score = 0;
  ChannelBundle_t * p_segment = p_sigmodel->p_segments;
  int numtimedefphases = p_sigmodel->p_earth->numtimedefphases;

  for (int i=0; i < p_sigmodel->numsegments; ++i) {

    /* compute the time period during which the event will affect the station */
    double first_envelope_time, last_envelope_time;
    evt_arrival_times(p_event, p_segment->siteid, numtimedefphases, &first_envelope_time, &last_envelope_time);
    last_envelope_time += MAX_ENVELOPE_LENGTH;

    /* if this trace doesn't fall within that period, skip it */
    if (p_signal->start_time > last_envelope_time || p_signal->end_time < first_envelope_time) {
      continue;
    }

    /* index within this trace at which the event arrives */
    env_start_idx = time2idx(first_envelope_time, p_segment->start_time, p_segment->hz);
    
    /* we compute scores for the background event set, and for an
       augmented event set which includes the specified event. */
    Event_t ** augmented_events = augment_events(pp_other_events, event, num_other_events);
    
    double * p_means, * p_vars;
    long len;




    /* score augmented event set */
    double event_lp = 0;
    for (int chan_num = 0; chan_num < NUM_CHANS; ++chan_num) {
      if (p_segment->p_channels[chan_num] == NULL) continue;

      envelope_means_vars(prior, p_segment->hz, first_envelope_time, last_envelope_time, p_earth, numevents+1, augmented_events, p_segment->siteid, chan_num, &len, &p_means, &p_vars);
      long compare_len = MIN(len, p_segment->len - env_start_idx);
      event_lp +=  indep_Gaussian_LogProb(compare_len, p_segment->p_channels[chan_num]->p_data + env_start_idx, p_means, p_vars);
      free(p_means);
      free(p_vars);
    }
    free(augmented_events);
    
    /* score background event set */
    double background_lp = 0;
    for (int chan_num = 0; chan_num < NUM_CHANS; ++chan_num) {
      if (p_segment->p_channels[chan_num] == NULL) continue;
      envelope_means_vars(prior, p_segment->hz, first_envelope_time, last_envelope_time, p_earth, numevents, p_world->pp_prop_events, p_segment->siteid, chan_num, &len, &p_means, &p_vars);
    background_lp +=  indep_Gaussian_LogProb(compare_len, p_segment->p_channels[chan_num]->p_data + env_start_idx, p_means, p_vars);
    free(p_means);
    free(p_vars);
    }
    score += (event_lp - background_lp);

    p_signal++;
  }

  return score;

}

void SignalPrior_UnInit(SignalPrior_t * prior) {
  free(prior->p_stations);
  free(prior->p_stations);
}
