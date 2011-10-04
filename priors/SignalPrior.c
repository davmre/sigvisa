#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../sigvisa.h"

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
  prior->p_station_noise_means = (double *) calloc(prior->numsites,
                                              sizeof(double));
  prior->p_station_noise_vars = (double *) calloc(prior->numsites,
                                              sizeof(double));
  int numentries;
  if (1 != fscanf(fp, "%d\n", &numentries))
  {
    fprintf(stderr, "error reading num entries from %s\n", filename);
    exit(1);
  }

  for (int i=0; i < numentries; ++i) {
    int siteid;
    double mean, var;
    if (3 != fscanf(fp, "%d %lf %lf\n", &siteid, &mean, &var))
      {
	fprintf(stderr, "error reading mean and variance for site %d from %s\n", i, filename);
	exit(1);
      }
    
    fprintf(stdout, "%d: loaded %d %lf %lf\n", i, siteid, mean, var);
    prior->p_station_noise_means[siteid] = mean;
    prior->p_station_noise_vars[siteid] = var;
  }

  fclose(fp);
  fflush(stdout);
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
	       double ** p_envelope,
	       int * t) {
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
			Signal_t * p_signal, 
			EarthModel_t * p_earth, 
			int numevents, 
			Event_t * events, 
			PyArrayObject * arrtimes,
			 int siteid,
			 double ** p_p_means,
			 double ** p_p_vars) {
  
  (*p_p_means) = (double *) calloc(p_signal->len, sizeof(double));
  double *p_means = (*p_p_means);
  (*p_p_vars) = (double *) calloc(p_signal->len, sizeof(double));
  double *p_vars = (*p_p_vars);

  double noise_mean = prior->p_station_noise_means[p_signal->siteid];
  double noise_var = prior->p_station_noise_vars[p_signal->siteid];
  for (int i=0; i < p_signal->len; ++i) {
    p_means[i] = noise_mean;
    p_vars[i] = noise_var;
  }

  for (int i=0; i < numevents; ++i) {
    // for (int phaseid = 0; phaseid < p_earth->num_phases; ++phaseid) {
    
    int phaseid = 0; /* TODO: work with multiple phases */
    double arrtime = ARRAY3(arrtimes, siteid, i, phaseid);
    if (arrtime < 0) continue;

    double * p_envelope;
    int len;
    phase_env(prior, p_earth, events + i, p_signal->hz, siteid, phaseid, &p_envelope, &len);
    
    add_signals(p_means, p_signal->len, p_envelope, len, lround(time2idx(arrtime, p_signal->start_time, p_signal->hz)));
    add_signals(p_vars, p_signal->len, p_envelope, len, lround(time2idx(arrtime, p_signal->start_time, p_signal->hz)));
      
    free(p_envelope);
    //}
  }
  
}

double SignalPrior_LogProb(SignalPrior_t * prior, int numsignals, Signal_t * p_signals, EarthModel_t * p_earth, int numevents, Event_t * events, PyArrayObject * arrtimes) {
  
  double lp = 0;
  Signal_t * p_signal = p_signals;
  for (int i=0; i < numsignals; ++i) {
    print_signal(p_signal);

    double * p_means, * p_vars;
    envelope_means_vars(prior, p_signal, p_earth, numevents, events, arrtimes, p_signal->siteid, &p_means, &p_vars);

    
    lp += indep_Gaussian_LogProb(p_signal->len, p_signal->p_data, p_means, p_vars);
    fprintf(stdout, "lp is %lf\n", lp);

    free(p_means);
    free(p_vars);

    p_signal++;
  }

  return lp;

}

void SignalPrior_UnInit(SignalPrior_t * prior) {
  free(prior->p_station_noise_means);
  free(prior->p_station_noise_vars);
}
