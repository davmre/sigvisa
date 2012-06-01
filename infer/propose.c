#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <pthread.h>

#include "../sigvisa.h"
#include "quickselect.h"

#define BUCKET(lonidx, latidx, timeidx) \
p_bucket_score[(lonidx)*(numlat)*(numtime) + (latidx)*(numtime) + (timeidx)]

#define BEGIN_LON_LOOP \
for (lonidx = 0; lonidx < numlon; lonidx ++)\
{\
  lon = -180 + lonidx * degree_step

#define END_LON_LOOP } do {} while(0)

#define BEGIN_LAT_LOOP \
for (latidx = 0; latidx < numlat; latidx ++)\
{\
  lat = RAD2DEG * asin(-1.0 + latidx * z_step)

#define END_LAT_LOOP } do {} while(0)

#define BEGIN_TIME_LOOP \
for (timeidx = 0; timeidx < numtime; timeidx ++)\
{\
  time = time_low + timeidx * time_step

#define END_TIME_LOOP } do {} while(0)



/* COPIED FROM INFER.C */
#define FIXUP_EVLON(p_event)                    \
  do {                                          \
    if ((p_event)->evlon < -180)                \
      (p_event)->evlon += 360;                  \
    else if ((p_event)->evlon >= 180)           \
      (p_event)->evlon -= 360;                  \
  }  while (0)

/* TODO: if latitude exceeds the poles then we should change the longitude
 * to walk across the pole */
#define FIXUP_EVLAT(p_event)                            \
  do {                                                  \
    if ((p_event)->evlat < -90)                         \
      (p_event)->evlat = -180 - (p_event)->evlat;       \
    else if ((p_event)->evlat > 90)                     \
      (p_event)->evlat = 180 - (p_event)->evlat;        \
  } while(0)


struct thread_data {
  int det_low;
  int det_high;
  int *p_skip_inv;
  int *p_skip_det;
  
  int num_step;
  double degree_step;
  double time_low;
  double time_high;
  
  SigModel_t *p_sigmodel;
  Event_t *p_inv_events;
  Event_t *p_event;
  EarthModel_t *p_earth;
  
  int num_other_events;
  const Event_t ** pp_other_events;
  PyObject * log_segment_cb;

  int N;
  double scale;
  Event_t *p_curr_event;
  Event_t *p_best_event;

  int tid;
  int numthreads;
};


void initialize_mean_arrivals(SigModel_t * p_sigmodel,
			      Event_t * p_event) {


  EarthModel_t * p_earth = p_sigmodel->p_earth;

  int numsites = EarthModel_NumSites(p_earth);
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  

  for (int siteid = 1; siteid <= numsites; ++siteid) {

    
    for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {
      if (!USE_PHASE(phase)) continue;
      double pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
						   p_event->evlat, p_event->evdepth,
						   p_event->evtime, phase,
						   siteid-1);


      /* check if the site is in the shadow zone for the event - phase */
      if (pred_arrtime < 0) {
	continue;
      }
      


    /* check if the site is up */
    //if (!SigModel_IsSiteUp(p_sigmodel, siteid, pred_arrtime))
    //continue;
    /*if (!have_signal(p_sigmodel, siteid, pred_arrtime - 5, pred_arrtime+MAX_ENVELOPE_LENGTH)) {
      continue;
      }*/


      Arrival_t * p_arr = p_event->p_arrivals + 
	(siteid-1)*numtimedefphases + phase;

      p_arr->phase = phase;
      p_arr->siteid = siteid;
      
      p_arr->time = EarthModel_ArrivalTime(p_earth, 
					   p_event->evlon, 
					   p_event->evlat, 
					   p_event->evdepth, 
					   p_event->evtime, 
					   phase, siteid-1);
      
      p_arr->amp = ArrivalAmplitudePrior_Point(&p_sigmodel->arr_amp_prior,
					       p_event->evmag, 
					       p_event->evdepth, 
					       p_arr->time - p_event->evtime, 
					       siteid-1, phase);

      p_arr->azi = EarthModel_ArrivalAzimuth(p_earth, 
					     p_event->evlon, 
					     p_event->evlat, 
					     siteid-1);
      p_arr->slo = EarthModel_ArrivalSlowness(p_earth, 
					      p_event->evlon, 
					      p_event->evlat, 
					      p_event->evdepth, 
					      phase, siteid-1); 

      LogTrace("got arrival time %lf amp %lf azi %lf slo %lf for evtime %lf phase %d siteid %d", p_arr->time,  p_arr->amp,  p_arr->azi,  p_arr->slo, p_event->evtime, phase, siteid);      
    }
  }
}

/* find the best set of detection from the available ones to build the
 * event and find its best possible score */
double optimize_arrivals_sta_grid(SigModel_t * p_sigmodel,
				  Event_t * p_event,
				  int siteid,
				  int num_other_events,
				  const Event_t ** pp_other_events)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  int numarrivals;
  
  clock_t start, local_start, end;
  double dif;
  start = local_start = clock();

  p_earth = p_sigmodel->p_earth;

  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  numarrivals = numtimedefphases;  

  /* for each site and phase, estimate arrival info from the relevant priors */

  Arrival_t * sta_arrivals = p_event->p_arrivals + (siteid-1)*numtimedefphases;

  Arrival_t * base = calloc(numarrivals, sizeof(Arrival_t)); 
  memcpy(base, sta_arrivals, numarrivals * sizeof(Arrival_t));

  LogTrace("    optimizing arrivals at site %d, given %d other events, for event %s ", siteid, num_other_events, event_str(p_event));


  //score_event_sig(p_sigmodel, p_event, num_other_events, pp_other_events);
  //printf("    naive score is %lf\n", p_event->evscore);
  

  Arrival_t * best = calloc(numarrivals, sizeof(Arrival_t));
  memcpy(best, base, numarrivals * sizeof(Arrival_t));

  double best_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
  LogTrace("    naive sta score is %lf", best_score);
  /* then try a grid search over arrival info */
  
  const double time_step = .5;
  const double num_time_steps = 10;
  const double amp_step = 0.3;
  const double num_amp_steps = 5;
  //const double azi_step = 8;
  //const double num_azi_steps = 10;
  //const double slo_step = 1;
  //const double num_slo_steps = 2;

  for (int i=0; i < 10; ++i) {

    for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {
      if (!USE_PHASE(phase)) continue;
	if (!have_signal(p_sigmodel, siteid, (base+phase)->time - 5, (base+phase)->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}

	for (double time = (base+phase)->time - time_step * num_time_steps; time < (base+phase)->time + time_step * num_time_steps; time += time_step) { 

	  double min_amp = MAX((base+phase)->amp-amp_step*num_amp_steps, 0.05);
	double max_amp = MAX((base+phase)->amp+amp_step*num_amp_steps, 0.1);

	for (double amp = min_amp; amp <= max_amp; amp += amp_step) { 

	double min_azi = 0;
	double max_azi = 360;
	double azi_step = 36;

	for (double azi = min_azi; azi <= max_azi; azi += azi_step) { 

	double min_slo = 0;
	double max_slo;
	iangle_to_slowness(90, phase, &max_slo);
	double slo_step = 5;

	for (double slo = min_slo; slo <= max_slo; slo += slo_step) { 


	  Arrival_t * p_arr = sta_arrivals + phase;
	  p_arr->time = time;	      
	  p_arr->amp = amp;	      
	  p_arr->azi = azi;	      
	  p_arr->slo = slo;	      
	  

	  double ev_sta_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
	  
	  LogTrace("phase %d time %lf amp %lf azi %lf slo %lf score %lf", phase, time, amp, azi, slo, ev_sta_score);

	  if (ev_sta_score > best_score) {
	    LogTrace("new arrival time %lf at %d for phase %d has score %lf better than best %lf", time, siteid, phase, ev_sta_score, best_score);
	    best_score = ev_sta_score;
	    memcpy(best, sta_arrivals, numarrivals * sizeof(Arrival_t));
	  }
	}

	}
	}
	}

	// use the best arrival time for this phase/station while searching at the next
	memcpy(sta_arrivals, best, numarrivals * sizeof(Arrival_t));
    
   
    }
    memcpy(base, best, numarrivals * sizeof(Arrival_t));
  }

  

  end = clock();
  dif = (end - start) / (double)CLOCKS_PER_SEC;

  
  double final_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
  //score_event_sig(p_sigmodel, p_event, num_other_events, pp_other_events);
  //printf("    final score is %lf\n", p_event->evscore);
  

  if (isnan(final_score)) {
    LogError("score is nan! %lf %lf\n", final_score, best_score);
    exit(-1);
  }

  assert(fabs(final_score - best_score) < 0.01 );

  free(base);
  free(best);

  return final_score;
}

/* find the best set of detection from the available ones to build the
 * event and find its best possible score */
double optimize_arrivals_sta(SigModel_t * p_sigmodel,
			     Event_t * p_event,
			     int siteid,
			     int num_other_events,
			     const Event_t ** pp_other_events)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  int numarrivals;
  
  clock_t start, local_start, end;
  double dif;
  start = local_start = clock();

  p_earth = p_sigmodel->p_earth;

  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  numarrivals = numtimedefphases;  

  /* for each site and phase, estimate arrival info from the relevant priors */

  Arrival_t * sta_arrivals = p_event->p_arrivals + (siteid-1)*numtimedefphases;

  Arrival_t * base = calloc(numarrivals, sizeof(Arrival_t)); 
  memcpy(base, sta_arrivals, numarrivals * sizeof(Arrival_t));

  LogTrace("    optimizing arrivals at site %d, given %d other events, for event %s ", siteid, num_other_events, event_str(p_event));


  //score_event_sig(p_sigmodel, p_event, num_other_events, pp_other_events);
  //printf("    naive score is %lf\n", p_event->evscore);
  

  Arrival_t * best = calloc(numarrivals, sizeof(Arrival_t));
  memcpy(best, base, numarrivals * sizeof(Arrival_t));

  double best_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
  LogTrace("    naive sta score is %lf", best_score);
  /* then try a grid search over arrival info */
  
  const double time_step = 1;
  const double num_time_steps = 8;
  const double amp_step = 1.5;
  const double num_amp_steps = 7;
  //const double azi_step = 8;
  //const double num_azi_steps = 10;
  //const double slo_step = 1;
  //const double num_slo_steps = 2;

  for (int i=0; i < 4; ++i) {

    for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {
      if (!USE_PHASE(phase)) continue;
	if (!have_signal(p_sigmodel, siteid, (base+phase)->time - 5, (base+phase)->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}

	for (double time = (base+phase)->time - time_step * num_time_steps; time < (base+phase)->time + time_step * num_time_steps; time += time_step) { 
	  
	  Arrival_t * p_arr = sta_arrivals + phase;
	  p_arr->time = time;
	      
	  double ev_sta_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
	  
	  if (ev_sta_score > best_score) {
	    LogTrace("new arrival time %lf at %d for phase %d has score %lf better than best %lf", time, siteid, phase, ev_sta_score, best_score);
	    best_score = ev_sta_score;
	    memcpy(best, sta_arrivals, numarrivals * sizeof(Arrival_t));
	  }
	}

	// use the best arrival time for this phase/station while searching at the next
	memcpy(sta_arrivals, best, numarrivals * sizeof(Arrival_t));
    
   
    }

    for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {
      if (!USE_PHASE(phase)) continue;
	//	printf("starting w/ arrival amp %lf at %d for phase %d\n", (base+idx)->amp, site, phase);

	if (!have_signal(p_sigmodel, siteid, (base+phase)->time - 5, (base+phase)->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}

	

	double min_amp = MAX((base+phase)->amp*pow(amp_step, -1*num_amp_steps), 0.05);
	double max_amp = MAX((base+phase)->amp*pow(amp_step, num_amp_steps), 0.1);

	for (double amp = min_amp; amp <= max_amp; amp *= amp_step) { 
	  Arrival_t * p_arr = sta_arrivals + phase;
	  p_arr->amp = amp;

	  double ev_sta_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
	  
	  if (ev_sta_score > best_score) {
	    best_score = ev_sta_score;
	    memcpy(best, sta_arrivals, numarrivals * sizeof(Arrival_t));
	  }
  
	}

	// use the best arrival time for this phase/station while searching at the next
	memcpy(sta_arrivals, best, numarrivals * sizeof(Arrival_t));
    
    }
 
for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {
  if (!USE_PHASE(phase)) continue;
	if (!have_signal(p_sigmodel, siteid, (base+phase)->time - 5, (base+phase)->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}
	
	//double min_azi = (base+phase)->azi-azi_step*num_azi_steps;
	//double max_azi = (base+phase)->azi+azi_step*num_azi_steps;
	double min_azi = 0;
	double max_azi = 360;
	double azi_step = 36;

	for (double azi = min_azi; azi <= max_azi; azi += azi_step) { 
	  Arrival_t * p_arr = sta_arrivals + phase;
	  p_arr->azi = azi;

	  double ev_sta_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
	  
	  if (ev_sta_score > best_score) {
	    best_score = ev_sta_score;
	    memcpy(best, sta_arrivals, numarrivals * sizeof(Arrival_t));
	  }
  
	}

	// use the best arrival time for this phase/station while searching at the next
	memcpy(sta_arrivals, best, numarrivals * sizeof(Arrival_t));
    
    }

for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {
  if (!USE_PHASE(phase)) continue;
	//	printf("starting w/ arrival amp %lf at %d for phase %d\n", (base+idx)->amp, site, phase);

	if (!have_signal(p_sigmodel, siteid, (base+phase)->time - 5, (base+phase)->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}
	
	//double min_slo = (base+phase)->slo-slo_step*num_slo_steps;
	//double max_slo = (base+phase)->slo+slo_step*num_slo_steps;
	double min_slo = 0;
	double max_slo;
	iangle_to_slowness(90, phase, &max_slo);
	double slo_step = 5;

	for (double slo = min_slo; slo <= max_slo; slo += slo_step) { 
	  Arrival_t * p_arr = sta_arrivals + phase;
	  p_arr->slo = slo;

	  double ev_sta_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
	  
	  if (ev_sta_score > best_score) {
	    best_score = ev_sta_score;
	    memcpy(best, sta_arrivals, numarrivals * sizeof(Arrival_t));
	  }
  
	}

	// use the best arrival time for this phase/station while searching at the next
	memcpy(sta_arrivals, best, numarrivals * sizeof(Arrival_t));
    
    }

    // TODO: re-enable optimizing slowness and azimuth once I've actually told the signal model to care about them
    /*
    for (int site = 0; site < numsites; ++site) {
      for (int phase = 0; phase < numtimedefphases; ++phase) {
	
	int idx = site*numtimedefphases+phase;	
	

	............
    */

    memcpy(base, best, numarrivals * sizeof(Arrival_t));
  }

  

  end = clock();
  dif = (end - start) / (double)CLOCKS_PER_SEC;

  
  double final_score = score_event_sta_sig(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);
  //score_event_sig(p_sigmodel, p_event, num_other_events, pp_other_events);
  //printf("    final score is %lf\n", p_event->evscore);
  

  if (isnan(final_score)) {
    LogError("score is nan! %lf %lf\n", final_score, best_score);
    exit(-1);
  }

  assert(fabs(final_score - best_score) < 0.01 );

  free(base);
  free(best);

  return final_score;
}


/* find the best set of detection from the available ones to build the
 * event and find its best possible score */
void optimize_arrivals(SigModel_t * p_sigmodel,
		       Event_t * p_event,
		       int num_other_events,
		       const Event_t ** pp_other_events)
{
  EarthModel_t * p_earth;
  int numsites;

  p_earth = p_sigmodel->p_earth;
  numsites = EarthModel_NumSites(p_earth);

  /*  if (p_event->evscore != 0) {
    score_event_sig(p_sigmodel, p_event, num_other_events, pp_other_events);
    printf("pre-optimize score is %lf\n", p_event->evscore);
    }*/

  initialize_mean_arrivals(p_sigmodel, p_event);
  /*score_event_sig(p_sigmodel, p_event, num_other_events, pp_other_events);
  printf("mean-arrivals score is %lf\n", p_event->evscore);
  */
  double evscore = 0;

  for (int siteid = 1; siteid <= numsites; ++siteid) {

    double pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
					    p_event->evlat, p_event->evdepth,
					    p_event->evtime, 0,
					    siteid-1);
    /* check if the site is in the shadow zone for the event - phase */
    if (pred_arrtime < 0)
      continue;
    
    /* check if the site is up */
    //if (!SigModel_IsSiteUp(p_sigmodel, siteid, pred_arrtime))
    //continue;
    if (!have_signal(p_sigmodel, siteid, pred_arrtime - 5, pred_arrtime+MAX_ENVELOPE_LENGTH)) {
      continue;
    }

    double site_score = optimize_arrivals_sta(p_sigmodel, p_event, siteid, num_other_events, pp_other_events);

    assert(!isnan(site_score));

    evscore += site_score;
  }

  evscore += score_event_evprior(p_sigmodel, p_event);
 
  //score_event_sig(p_sigmodel, p_event, num_other_events, pp_other_events);
  p_event->evscore = evscore;
  LogTrace("post-optimize score is %lf", p_event->evscore);
  

}


void *propose_best_event_helper(void *args)
{
  struct thread_data *params = (struct thread_data *) args;
  
  int det_low = params->det_low;
  int det_high = params->det_high;
  int *p_skip_det = params->p_skip_det;
  
  double time_low = params->time_low;
  double time_high = params->time_high;

  SigModel_t * p_sigmodel = params->p_sigmodel;

  Event_t *p_event = params->p_event;
  Event_t *p_curr_event = params->p_curr_event;

  double scale = params->scale;

  // srand(time(NULL) + params->tid);

  for (int i = 0; i < params->N; i += 1)
  {
    /* perturb the current event in all dimensions */
    p_event->evlon = p_curr_event->evlon + RAND_UNIFORM(-scale, scale) * 5.0;
    FIXUP_EVLON(p_event);
    p_event->evlat = p_curr_event->evlat + RAND_UNIFORM(-scale, scale) * 5.0;
    FIXUP_EVLAT(p_event);
    p_event->evdepth = RAND_UNIFORM(MAX(MIN_DEPTH, 
                                        p_curr_event->evdepth-100*scale), 
                                    MIN(MAX_DEPTH,
                                        p_curr_event->evdepth+100*scale));
    p_event->evtime = RAND_UNIFORM(MAX(p_curr_event->evtime - scale * 50, 
                                       time_low),
                                   MIN(p_curr_event->evtime + scale * 50,
                                       time_high));
    
    p_event->evmag = RAND_UNIFORM(MAX(MIN_MAGNITUDE, 
                                      p_curr_event->evmag - scale), 
                                  MIN(MAX_MAGNITUDE,
                                      p_curr_event->evmag + scale));

    optimize_arrivals(p_sigmodel, p_event,  params->num_other_events, params->pp_other_events);

    /* maintain the overall best event */
    if (p_event->evscore > params->p_best_event->evscore) {
      LogTrace(" new score %lf is better than old best score %lf", p_event->evscore, params->p_best_event->evscore);
      COPY_EVENT(p_sigmodel, params->p_best_event, p_event);
    }

    if (p_event->evscore > p_curr_event->evscore)
      COPY_EVENT(p_sigmodel, p_curr_event, p_event);

#ifdef SIM_ANNEAL    
    else
    {
      double temp = 20.0 / log(params->N+2);
      
      if (RAND_DOUBLE < exp((p_event->evscore - p_curr_event->evscore) / temp))
        COPY_EVENT(p_sigmodel, p_event, p_curr_event);
    }
#endif

  }
  return NULL;
}

static void propose_best_event(SigModel_t * p_sigmodel,
                               Event_t * p_event,
			       int num_other_events, const Event_t ** pp_other_events,
                               int det_low, int det_high,
                               int * p_skip_det,
                               double time_low, double time_high,
                               double scale, int numthreads)
{
  Event_t * p_best_event;
  Event_t * p_curr_event;
  
  // TODO: make it feasible to set this back to 1000
  const int N = 10;
  
  p_best_event = ALLOC_EVENT(p_sigmodel);
  p_curr_event = ALLOC_EVENT(p_sigmodel);

  /* the initial event is the starting event as well as the initial best */
  COPY_EVENT(p_sigmodel, p_best_event, p_event);
  COPY_EVENT(p_sigmodel, p_curr_event, p_event);



  /* Allocate space for threads and args, initialize starting values */
  pthread_t * threads;
  struct thread_data * thread_args;
  threads = (pthread_t *) calloc(numthreads, sizeof(*threads));
  thread_args = (struct thread_data *) calloc(numthreads,
                                              sizeof(*thread_args));

  for (int i = 0; i < numthreads; i++)
  {
    thread_args[i].det_low = det_low;
    thread_args[i].det_high = det_high;
    thread_args[i].p_skip_det = p_skip_det;
    thread_args[i].time_low = time_low;
    thread_args[i].time_high = time_high;

    thread_args[i].p_sigmodel = p_sigmodel;
    thread_args[i].p_event = ALLOC_EVENT(p_sigmodel);
    COPY_EVENT(p_sigmodel, thread_args[i].p_event, p_event);
    thread_args[i].p_curr_event = ALLOC_EVENT(p_sigmodel);
    COPY_EVENT(p_sigmodel, thread_args[i].p_curr_event, p_curr_event);
    thread_args[i].p_best_event = ALLOC_EVENT(p_sigmodel);
    COPY_EVENT(p_sigmodel, thread_args[i].p_best_event, p_best_event);

    thread_args[i].num_other_events = num_other_events;
    thread_args[i].pp_other_events = pp_other_events;
    
    thread_args[i].N = N/numthreads+1;
    thread_args[i].scale = scale;
    thread_args[i].tid = i;
    pthread_create(&threads[i], NULL, propose_best_event_helper,
                   (void *) &thread_args[i]);
  }

  /* Wait for all threads to finish */
  for (int i = 0; i < numthreads; i++)
  {
    pthread_join(threads[i], NULL);
  }

  /* Get best event from all threads */
  for (int i = 0; i < numthreads; i++)
  {
    if (thread_args[i].p_best_event->evscore > p_best_event->evscore) {
      COPY_EVENT(p_sigmodel, p_best_event, thread_args[i].p_best_event);
    }
    free_event(thread_args[i].p_event);
    free_event(thread_args[i].p_curr_event);
    free_event(thread_args[i].p_best_event);
  }

  /* return the overall best event */
  COPY_EVENT(p_sigmodel, p_event, p_best_event);

  free_event(p_curr_event);
  free_event(p_best_event);
  free(threads);
  free(thread_args);
}

void *propose_invert_step_helper(void *args)
{
  struct thread_data *params = (struct thread_data *) args;

  int det_low = params->det_low;
  int det_high = params->det_high;
  int *p_skip_inv = params->p_skip_inv;
  int *p_skip_det = params->p_skip_det;

  int num_step = params->num_step;
  double degree_step = params->degree_step;
  double time_low = params->time_low;
  double time_high = params->time_high;
  int numthreads = params->numthreads;

  int num_other_events = params->num_other_events;
  const Event_t ** pp_other_events = params->pp_other_events;
  PyObject * log_segment_cb = params->log_segment_cb;

  SigModel_t * p_sigmodel = params->p_sigmodel;

  Event_t *p_inv_events = params->p_inv_events;
  Event_t *p_event = params->p_event;
  EarthModel_t * p_earth = params->p_earth;

  int tid = params->tid;

  int inv_detnum, lonidx, latidx;
  double lon, lat, mag;



  /* Loop increments by NUM_THREADS to ensure that each thread gets a
   * different datum */
  for (inv_detnum = det_low+tid; inv_detnum < det_high; 
       inv_detnum += numthreads)
  {
    Detection_t * p_inv_det;
    int det_off = inv_detnum - det_low;

    Event_t * p_best_nonzero_event = ALLOC_EVENT(p_sigmodel);
    p_best_nonzero_event->evscore = -1 * DBL_MAX;

    if (p_skip_inv[det_off])
      continue;

    p_inv_det = p_sigmodel->p_detections + inv_detnum;

    /* save the longitude and latitude */
    lon = (p_inv_events + det_off)->evlon;
    lat = (p_inv_events + det_off)->evlat;

    /* fix the depth to 0 */
    p_event->evdepth = 0;

    LogDebug("starting to invert %s", event_str(p_inv_events + det_off));

    clock_t start = clock();

    for (mag=3; mag <4.1; mag+=1)
    {
      p_event->evmag = mag;

      for (lonidx=-num_step; lonidx<=num_step; lonidx++)
      {
        p_event->evlon = lon + lonidx * degree_step;
        FIXUP_EVLON(p_event);

        for (latidx=-num_step; latidx<=num_step; latidx ++)
        {
          double trvtime;
          p_event->evlat = lat + latidx * degree_step;
          FIXUP_EVLAT(p_event);

	  for (int phase = 0; phase < EarthModel_NumTimeDefPhases(p_sigmodel->p_earth); ++phase) {
	    if(!USE_PHASE(phase)) continue;
	    

	    trvtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                           p_event->evlat, p_event->evdepth,0,
                                           phase, p_inv_det->site_det);
          if (trvtime < 0)
            continue;

          p_event->evtime = p_inv_det->time_det - trvtime;

          if ((p_event->evtime < time_low) || (p_event->evtime > time_high))
          {
	    LogDebug("continuing time %lf dtime %lf ttime %lf low %lf high %lf", p_event->evtime, p_inv_det->time_det, trvtime, time_low, time_high);
            continue;
          }


	  LogDebug("optimizing arrivals for %s", event_str(p_event));
	  optimize_arrivals(p_sigmodel, p_event, num_other_events, pp_other_events);

	  //Compare to the score in p_best_events with index tid (for this thread)
          if (p_event->evscore > params->p_best_event->evscore)
          {

	    COPY_EVENT(p_sigmodel, params->p_best_event, p_event);

            LogDebug("CURR BEST: %s", event_str(params->p_best_event));
          } else {
	    LogDebug("score %lf not better than %lf", p_event->evscore, params->p_best_event->evscore);
	  }

	  if (p_event->evscore > p_best_nonzero_event->evscore)
          {

	    COPY_EVENT(p_sigmodel, p_best_nonzero_event, p_event);
          }
        }
      }
      }
      
      

    }
    /*
        if (p_sigmodel != NULL && log_segment_cb != NULL) {
	printf("writing signal logs\n");
	char txt[50];
	printf("best event ");
	print_event(p_best_nonzero_event);
	snprintf(txt, 50, "inverted %d score %lf", inv_detnum, p_best_nonzero_event->evscore);
	log_segments_events(p_sigmodel, log_segment_cb, 1, (const Event_t **) &p_best_nonzero_event, DBL_MAX, Py_BuildValue("s", txt));
	}*/

    free(p_best_nonzero_event);

    clock_t end = clock();
    LogInfo("finished inverting detection %d, best score %lf. time elapsed : %lf seconds ", inv_detnum, params->p_best_event->evscore, (end-start) / (double)CLOCKS_PER_SEC);

  }

  
  return NULL;
}

/* propose events by inverting detections and keeping the best
 * inverted detections 
 * this version does a fixed number of steps per detection */
/* can be very slow when there are a large number of detections, for example:
  python -m utils.check_propose -k 1299822400 -l tohoku -d 2.5 -w 1800  -r 1.2
*/
int propose_invert_step(SigModel_t * p_sigmodel,
			Event_t **pp_events,
                        double time_low, double time_high, int det_low,
                        int det_high, double degree_step, int num_step,
                        int numthreads, 
			int runid, PyObject * log_segment_cb)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  
  int detnum, inv_detnum;

  int numevents;

  Event_t * p_best_event;
  int * p_skip_det;                          /* numdetections */
  
  Event_t * p_inv_events;
  int * p_skip_inv;

  p_earth = p_sigmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_skip_det = (int *) calloc(p_sigmodel->numdetections, sizeof(*p_skip_det));
  
  if (!p_skip_det)
  {
    return -1;
  }

  p_inv_events = (Event_t *)calloc(det_high - det_low, sizeof(*p_inv_events));
  if (!p_inv_events)
  {
    return -1;
  }
  
  /* initialize to 0 to indicate that the detection is not to be skipped */
  p_skip_inv = (int *)calloc(det_high - det_low, sizeof(*p_skip_inv));
  if (!p_skip_inv)
  {
    return -1;
  }

  /* first invert all the detections */
  for (inv_detnum = det_low; inv_detnum < det_high; inv_detnum ++)
  {
    Detection_t * p_inv_det;
    int inv_status;
    int det_off = inv_detnum - det_low;
    Event_t * p_event;
    
    
    p_inv_det = p_sigmodel->p_detections + inv_detnum;

    p_event = p_inv_events+det_off;
    
    inv_status = invert_detection(p_earth, p_inv_det, p_event,
                                  0 /* don't perturb */);

    LogInfo("inverted detection to %s", event_str(p_event));

    /* if the detection can't be inverted or the inverted event time is
     * outside the bounds of the desired window then skip it */
    if ((0 != inv_status) || (p_event->evtime < time_low)
        || (p_event->evtime > time_high))
    {
      p_skip_inv[det_off] = 1;
    }
  }
  

  
  numevents = 0;

  /* Allocate space for threads and args, initialize starting values */
    pthread_t * threads;
    struct thread_data * thread_args;
    threads = (pthread_t *) calloc(numthreads, sizeof(*threads));
    thread_args = (struct thread_data *) calloc(numthreads,
                                                sizeof(*thread_args));
    for (int i = 0; i < numthreads; i++)
      {
	thread_args[i].det_low = det_low;
	thread_args[i].det_high = det_high;
	thread_args[i].p_skip_inv = p_skip_inv;
	thread_args[i].p_skip_det = p_skip_det;
	thread_args[i].num_step = num_step;
	thread_args[i].degree_step = degree_step;
	thread_args[i].time_low = time_low;
	thread_args[i].time_high = time_high;
	thread_args[i].numthreads = numthreads;
	thread_args[i].p_sigmodel = p_sigmodel;
	thread_args[i].num_other_events = numevents;
	thread_args[i].pp_other_events = (const Event_t **)pp_events;
	thread_args[i].log_segment_cb = log_segment_cb;
	thread_args[i].p_inv_events = p_inv_events;
	thread_args[i].p_earth = p_earth;
	thread_args[i].tid = i;
      }

  do
  {

    for (int i = 0; i < numthreads; i++) {
      thread_args[i].num_other_events = numevents;
    }

    int siteid;
    int phase;

    LogInfo("spawning %d proposal threads", numthreads);

    /* Initialize what will be the overall best event */
    p_best_event = ALLOC_EVENT(p_sigmodel);

    p_best_event->evscore = 0;

    /* Create threads and have them run the proposal loop */
    for (int i = 0; i < numthreads; i++)
    {
      thread_args[i].p_event = ALLOC_EVENT(p_sigmodel);
      thread_args[i].p_best_event = ALLOC_EVENT(p_sigmodel);
      thread_args[i].p_best_event->evscore = 0;
      pthread_create(&threads[i], NULL, propose_invert_step_helper,
                     (void *) &thread_args[i]);
    }

    /* Wait for all threads to finish */
    for (int i = 0; i < numthreads; i++)
    {
      pthread_join(threads[i], NULL);
    }

    LogInfo("all proposal threads have finished!");
	
    /* Get the best event from all threads combined */
    for (int i = 0; i < numthreads; ++i)
    {
      if (thread_args[i].p_best_event->evscore > p_best_event->evscore)
      {
	COPY_EVENT(p_sigmodel, p_best_event, thread_args[i].p_best_event);

      } else {
	LogTrace("thread %d score %lf is not better than best %lf", i, thread_args[i].p_best_event->evscore, p_best_event->evscore);
      }
        free_event(thread_args[i].p_best_event);
        free_event(thread_args[i].p_event);
    }
    
    

    LogDebug("best proposed event is %s", event_str(p_best_event));

    /* finished inverting all detections and trying events in a ball around
     * them now let's see if we got something good */
    if (0 == p_best_event->evscore)
    {
      free_event(p_best_event);
      break;
    }
    
    /* now, improve this event to take advantage of its detections */

    LogDebug("now improving this event ...");
      
    clock_t improve_start = clock();

    propose_best_event(p_sigmodel, p_best_event, 
		       numevents, (const Event_t **)pp_events, 
		       det_low, det_high,
			 p_skip_det, time_low, time_high, 1, numthreads);
    
    LogDebug("improvement round 1 done, time %lf, resulting best event %s", (clock() - improve_start) / (double)CLOCKS_PER_SEC, event_str(p_best_event));
    improve_start = clock();

    propose_best_event(p_sigmodel, p_best_event, 
		       numevents, (const Event_t **)pp_events,  
		       det_low, det_high,
			 p_skip_det, time_low, time_high, .1, numthreads);

    LogDebug("improvement round 2 done, time %lf, resulting best event %s", (clock() - improve_start) / (double)CLOCKS_PER_SEC, event_str(p_best_event));

      //printf("now optimizing arrivals...\n");
      optimize_arrivals(p_sigmodel, p_best_event, numevents, 
			(const Event_t **)pp_events);
    
    
    LogInfo("BEST: %s", event_str(p_best_event));
    
    
    LogDebug("adding best event, and the cycle repeats!\n");
    /* add the best event to the list of events */
    pp_events[numevents ++] = p_best_event;

    
    
  } while (1);
  
  free(thread_args);
  free(threads);

  free(p_skip_inv);
  free(p_inv_events);
  free(p_skip_det);
  
  return numevents;
}

