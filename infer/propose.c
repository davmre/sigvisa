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
  
  NetModel_t *p_netmodel;
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

  Arrival_t * sta_arrivals = p_event->p_arrivals + siteid*numtimedefphases;

  Arrival_t * base = calloc(numarrivals, sizeof(Arrival_t)); 

  //printf("optimizing arrivals, given %d other events, for event ", num_other_events);
  //print_event(p_event);

  for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {

    (base+phase)->time = EarthModel_ArrivalTime(p_earth, 
					      p_event->evlon, 
					      p_event->evlat, 
					      p_event->evdepth, 
					      p_event->evtime, 
					      phase, siteid);

    (base+phase)->amp = ArrivalAmplitudePrior_Point(&p_sigmodel->arr_amp_prior,    p_event->evmag, 
						    p_event->evdepth, 
						    (base+phase)->time - p_event->evtime, 
						    siteid, phase);
      (base+phase)->azi = EarthModel_ArrivalAzimuth(p_earth, 
						  p_event->evlon, 
						  p_event->evlat, 
						  siteid);
      (base+phase)->slo = EarthModel_ArrivalSlowness(p_earth, 
						     p_event->evlon, 
						     p_event->evlat, 
						     p_event->evdepth, 
						     phase, siteid); 
  }
    

  end = clock();
  dif = (end - local_start) / (double)CLOCKS_PER_SEC;
  local_start = clock();
  //printf("init sites took %lf seconds.\n", dif);


  memcpy(sta_arrivals, base, numarrivals * sizeof(Arrival_t));

  Arrival_t * best = malloc(numarrivals * sizeof(Arrival_t));
  memcpy(best, base, numarrivals * sizeof(Arrival_t));


  /* then try a grid search over arrival info */
  
  const double time_step = 2.5;
  const double num_time_steps = 8;
  const double amp_step = 0.3;
  const double num_amp_steps = 2;
  //const double azi_step = 5;
  //const double num_azi_steps = 2;
  //const double slo_step = 2;
  //const double num_slo_steps = 2;

  double best_score = -DBL_MAX;

  for (int i=0; i < 3; ++i) {

    for (int phase = 0; phase < MAX_PHASE(numtimedefphases); ++phase) {

	if (!have_signal(p_sigmodel, siteid, (base+phase)->time - 5, (base+phase)->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}

	for (double time = (base+phase)->time - time_step * num_time_steps; time < (base+phase)->time + time_step * num_time_steps; time += time_step) { 
	  
	  Arrival_t * p_arr = sta_arrivals + phase;
	  p_arr->time = time;
	      
	  //printf("scoring arrival time at %d for phase %d is %lf\n", siteid, phase, time);

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
	
	//	printf("starting w/ arrival amp %lf at %d for phase %d\n", (base+idx)->amp, site, phase);

	if (!have_signal(p_sigmodel, siteid, (base+phase)->time - 5, (base+phase)->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}
	
	double min_amp = MAX((base+phase)->amp-amp_step*num_amp_steps, 0.05);
	double max_amp = MAX((base+phase)->amp+amp_step*num_amp_steps, 0.1);

	for (double amp = min_amp; amp <= max_amp; amp += amp_step) { 
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

  if (isnan(final_score)) {
    printf("score is nan! %lf %lf\n", final_score, best_score);
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

  double evscore = 0;

  for (int siteid = 0; siteid < numsites; ++siteid) {

    double pred_arrtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
					    p_event->evlat, p_event->evdepth,
					    p_event->evtime, 0,
					    siteid);
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
    //if (site_score) printf("site %d contributed score %lf\n", siteid, site_score);
  }

  double numsc, locsc, magsc;
  
  numsc = NumEventPrior_LogTimeRate(&p_sigmodel->num_event_prior);
  
  locsc = EventLocationPrior_LogProb(&p_sigmodel->event_location_prior,
				     p_event->evlon, p_event->evlat,
				     p_event->evdepth);
  
  magsc = EventMagPrior_LogProb(&p_sigmodel->event_mag_prior,
				p_event->evmag, 0);
  
  evscore += numsc + locsc + magsc;

  p_event->evscore = evscore;

}


/* find the best set of detection from the available ones to build the
 * event and find its best possible score */
static void propose_best_detections(NetModel_t * p_netmodel,
                                    Event_t * p_event,
                                    int det_low, int det_high,
                                    int * p_skip_det,
                                    int P_phase_only)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  int i;
  int detnum;
  int * p_event_best_detids;
  double * p_event_best_score;
  
  p_earth = p_netmodel->p_earth;

  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_event_best_detids = (int *) malloc(numsites * numtimedefphases 
                                       * sizeof (*p_event_best_score));
  p_event_best_score = (double *) malloc(numsites * numtimedefphases 
                                         * sizeof (*p_event_best_score));  

  /* find the best set of available detections for this event */
  for (i=0; i < numsites * numtimedefphases; i++)
  {
    p_event_best_detids[i] = -1;
    p_event_best_score[i] = 0;
  }


  /* for each detection find the best phase at the event */
  for (detnum = det_low; detnum < det_high; detnum ++)
  {
    Detection_t * p_det;
    int best_phase;
    double best_phase_score;
    int siteid;
    int phase;
    double distance, pred_az;
    
    if (p_skip_det[detnum])
      continue;
    
    p_det = p_netmodel->p_detections + detnum;

    siteid = p_det->site_det;
  
    best_phase = -1;
    best_phase_score = 0;

    distance = EarthModel_Delta(p_earth, p_event->evlon,
                                p_event->evlat, siteid);

    pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                        p_event->evlat, siteid);
  
    /* find the best phase for this detection if it is to match the event */
    for (phase=0; phase < numtimedefphases; phase++)
    {
      int poss;
      double detscore;

      /* we assume P phase is phasenum 0 */
      if (P_phase_only && (phase > 0))
        continue;
      
      p_event->p_num_dets[siteid * numtimedefphases + phase] = 1;
      p_event->p_all_detids[(siteid * numtimedefphases + phase)*MAX_PHASE_DET]
        = detnum;
      
      poss = score_event_site_phase(p_netmodel, p_event, siteid, phase,
                                    distance, pred_az, &detscore);

      if (poss && (detscore > 0)
          && ((-1 == best_phase) || (detscore > best_phase_score)))
      {
        best_phase = phase;
        best_phase_score = detscore;
      }
    }

    /* check if this detection-phase is the best detection for this phase at
     * the site */
    if ((-1 != best_phase)
        && ((-1 == p_event_best_detids[siteid * numtimedefphases 
                                       + best_phase])
            || (best_phase_score 
                > p_event_best_score[siteid * numtimedefphases 
                                     + best_phase])))
    {
      p_event_best_detids[siteid * numtimedefphases + best_phase] 
        = detnum;
      p_event_best_score[siteid * numtimedefphases + best_phase] 
        = best_phase_score;
    }
  }
  
  /* score the best such event */
  for (i=0; i < numsites * numtimedefphases; i++)
  {
    if (-1 == p_event_best_detids[i])
      p_event->p_num_dets[i] = 0;
    else
    {
      p_event->p_num_dets[i] = 1;
      p_event->p_all_detids[i * MAX_PHASE_DET] = p_event_best_detids[i];
    }
  }
  p_event->evscore = score_event(p_netmodel, p_event);

  free(p_event_best_detids);
  free(p_event_best_score);
}

void *propose_best_event_helper(void *args)
{
  struct thread_data *params = (struct thread_data *) args;
  
  int det_low = params->det_low;
  int det_high = params->det_high;
  int *p_skip_det = params->p_skip_det;
  
  double time_low = params->time_low;
  double time_high = params->time_high;

  NetModel_t * p_netmodel = params->p_netmodel;
  SigModel_t * p_sigmodel = params->p_sigmodel;
  assert(p_netmodel == NULL || p_sigmodel == NULL);

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

    if (p_netmodel != NULL) 
      propose_best_detections(p_netmodel, p_event, det_low, det_high,
			      p_skip_det, 0);
    else
      optimize_arrivals(p_sigmodel, p_event,  params->num_other_events, params->pp_other_events);

    /* maintain the overall best event */
    if (p_event->evscore > params->p_best_event->evscore) {
      printf(" new score %lf is better than old best score %lf\n", p_event->evscore, params->p_best_event->evscore);
      COPY_EVENT(p_netmodel, p_sigmodel, params->p_best_event, p_event);
    }

    if (p_event->evscore > p_curr_event->evscore)
      COPY_EVENT(p_netmodel, p_sigmodel, p_curr_event, p_event);

#ifdef SIM_ANNEAL    
    else
    {
      double temp = 20.0 / log(params->N+2);
      
      if (RAND_DOUBLE < exp((p_event->evscore - p_curr_event->evscore) / temp))
        COPY_EVENT(p_netmodel, p_sigmodel, p_event, p_curr_event);
    }
#endif

  }
  return NULL;
}

static void propose_best_event(NetModel_t * p_netmodel,
			       SigModel_t * p_sigmodel,
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
  
  assert(p_netmodel == NULL || p_sigmodel == NULL);

  p_best_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
  p_curr_event = ALLOC_EVENT(p_netmodel, p_sigmodel);

  /* the initial event is the starting event as well as the initial best */
  COPY_EVENT(p_netmodel, p_sigmodel, p_best_event, p_event);
  COPY_EVENT(p_netmodel, p_sigmodel, p_curr_event, p_event);



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

    thread_args[i].p_netmodel = p_netmodel;
    thread_args[i].p_sigmodel = p_sigmodel;
    thread_args[i].p_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
    COPY_EVENT(p_netmodel, p_sigmodel, thread_args[i].p_event, p_event);
    thread_args[i].p_curr_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
    COPY_EVENT(p_netmodel, p_sigmodel, thread_args[i].p_curr_event, p_curr_event);
    thread_args[i].p_best_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
    COPY_EVENT(p_netmodel, p_sigmodel, thread_args[i].p_best_event, p_best_event);

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
      COPY_EVENT(p_netmodel, p_sigmodel, p_best_event, thread_args[i].p_best_event);
    }
    free_event(thread_args[i].p_event);
    free_event(thread_args[i].p_curr_event);
    free_event(thread_args[i].p_best_event);
  }

  /* return the overall best event */
  COPY_EVENT(p_netmodel, p_sigmodel, p_event, p_best_event);

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

  NetModel_t * p_netmodel = params->p_netmodel;
  SigModel_t * p_sigmodel = params->p_sigmodel;
  assert(p_netmodel == NULL || p_sigmodel == NULL);

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

    Event_t * p_best_nonzero_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
    p_best_nonzero_event->evscore = -1 * DBL_MAX;

    if (p_skip_inv[det_off])
      continue;

    if (p_netmodel != NULL) p_inv_det = p_netmodel->p_detections + inv_detnum;
    else p_inv_det = p_sigmodel->p_detections + inv_detnum;

    /* save the longitude and latitude */
    lon = (p_inv_events + det_off)->evlon;
    lat = (p_inv_events + det_off)->evlat;

    /* fix the depth to 0 */
    p_event->evdepth = 0;

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

          trvtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                           p_event->evlat, p_event->evdepth,0,
                                           EARTH_PHASE_P, p_inv_det->site_det);
          if (trvtime < 0)
            continue;

          p_event->evtime = p_inv_det->time_det - trvtime;

          if ((p_event->evtime < time_low) || (p_event->evtime > time_high))
          {
            continue;
          }


	  if (p_netmodel != NULL) {
          /* score this event using the best detections available */
	    propose_best_detections(p_netmodel, p_event, det_low, det_high,
				    p_skip_det, 0 /* all phases */);
	  } else {
	    optimize_arrivals(p_sigmodel, p_event, num_other_events, pp_other_events);
	  }

	  //Compare to the score in p_best_events with index tid (for this thread)
          if (p_event->evscore > params->p_best_event->evscore)
          {

	    COPY_EVENT(p_netmodel, p_sigmodel, params->p_best_event, p_event);

            printf("CURR BEST: ");
            print_event(params->p_best_event);
          }

	  if (p_event->evscore > p_best_nonzero_event->evscore)
          {

	    COPY_EVENT(p_netmodel, p_sigmodel, p_best_nonzero_event, p_event);
          }
        }
      }

      
      

    }
    
    if (p_sigmodel != NULL && log_segment_cb != NULL) {
	printf("writing signal logs\n");
	char txt[50];
	printf("best event ");
	print_event(p_best_nonzero_event);
	snprintf(txt, 50, "inverted %d score %lf", inv_detnum, p_best_nonzero_event->evscore);
	log_segments_events(p_sigmodel, log_segment_cb, 1, (const Event_t **) &p_best_nonzero_event, DBL_MAX, Py_BuildValue("s", txt));
    }

    free(p_best_nonzero_event);

    clock_t end = clock();
    printf("finished inverting detection %d, best score %lf. time elapsed : %lf seconds \n ", inv_detnum, params->p_best_event->evscore, (end-start) / (double)CLOCKS_PER_SEC);

  }

  
  return NULL;
}

/* propose events by inverting detections and keeping the best
 * inverted detections 
 * this version does a fixed number of steps per detection */
/* can be very slow when there are a large number of detections, for example:
  python -m utils.check_propose -k 1299822400 -l tohoku -d 2.5 -w 1800  -r 1.2
*/
int propose_invert_step(NetModel_t * p_netmodel, 
			SigModel_t * p_sigmodel,
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

  if (p_netmodel != NULL)  p_earth = p_netmodel->p_earth;
  else p_earth = p_sigmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  if (p_netmodel != NULL)
    p_skip_det = (int *) calloc(p_netmodel->numdetections, sizeof(*p_skip_det));
  else
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
    
    
    if (p_netmodel != NULL) p_inv_det = p_netmodel->p_detections + inv_detnum;
    else p_inv_det = p_sigmodel->p_detections + inv_detnum;

    p_event = p_inv_events+det_off;
    
    inv_status = invert_detection(p_earth, p_inv_det, p_event,
                                  0 /* don't perturb */);

    printf("inverted detection to ");
    print_event(p_event);

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
	thread_args[i].p_netmodel = p_netmodel;
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

    printf("spawning %d proposal threads\n", numthreads);

    /* Initialize what will be the overall best event */
    p_best_event = ALLOC_EVENT(p_netmodel, p_sigmodel);

    p_best_event->evscore = 0;

    /* Create threads and have them run the proposal loop */
    for (int i = 0; i < numthreads; i++)
    {
      thread_args[i].p_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
      thread_args[i].p_best_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
      thread_args[i].p_best_event->evscore = 0;
      pthread_create(&threads[i], NULL, propose_invert_step_helper,
                     (void *) &thread_args[i]);
    }

    /* Wait for all threads to finish */
    for (int i = 0; i < numthreads; i++)
    {
      pthread_join(threads[i], NULL);
    }

    printf("all proposal threads have finished!\n");
	
    /* Get the best event from all threads combined */
    for (int i = 0; i < numthreads; ++i)
    {
      if (thread_args[i].p_best_event->evscore > p_best_event->evscore)
      {
	COPY_EVENT(p_netmodel, p_sigmodel, p_best_event, thread_args[i].p_best_event);

      } else {
	printf("thread %d score %lf is not better than best %lf\n", i, thread_args[i].p_best_event->evscore, p_best_event->evscore);
      }
        free_event(thread_args[i].p_best_event);
        free_event(thread_args[i].p_event);
    }
    
    

    printf("best proposed event is ");
    print_event(p_best_event);

    /* finished inverting all detections and trying events in a ball around
     * them now let's see if we got something good */
    if (0 == p_best_event->evscore)
    {
      free_event(p_best_event);
      break;
    }
    
    /* now, improve this event to take advantage of its detections */

    printf("now improving this event ...\n");
      
    clock_t improve_start = clock();

    propose_best_event(p_netmodel, p_sigmodel, p_best_event, 
		       numevents, (const Event_t **)pp_events, 
		       det_low, det_high,
			 p_skip_det, time_low, time_high, 1, numthreads);
    
    printf("improvement round 1 done, time %lf, resulting best event ", (clock() - improve_start) / (double)CLOCKS_PER_SEC);
    print_event(p_best_event);
    improve_start = clock();

    propose_best_event(p_netmodel, p_sigmodel, p_best_event, 
		       numevents, (const Event_t **)pp_events,  
		       det_low, det_high,
			 p_skip_det, time_low, time_high, .1, numthreads);

    printf("improvement round 2 done, time %lf, resulting best event ", (clock() - improve_start) / (double)CLOCKS_PER_SEC);
    print_event(p_best_event);

    if (p_netmodel != NULL) {
      /* and, once more find the best detections for this event */
      propose_best_detections(p_netmodel, p_best_event, det_low, det_high,
			      p_skip_det, 0 /* all phases */);
    } else {
      //printf("now optimizing arrivals...\n");
      optimize_arrivals(p_sigmodel, p_best_event, numevents, 
			(const Event_t **)pp_events);
    }
    
    
    printf("BEST:");
    print_event(p_best_event);
    
    if (p_netmodel != NULL) {
      printf("Best detections ");
      print_event_detections(p_earth, p_best_event);


      /*
       * we will identify the detections used by the best event and
       * make them off-limits for future events
       */
      for (siteid = 0; siteid < numsites; siteid ++) {
	for (phase = 0; phase < MAX_PHASE(numtimedefphases); phase ++) {
	  if (p_best_event->p_num_dets[siteid * numtimedefphases + phase] > 0) {
	    detnum = p_best_event->p_all_detids[(siteid * numtimedefphases
						 + phase)*MAX_PHASE_DET];
	    p_skip_det[detnum] = 1;
	  }
	}
      }
    }
    
    
    printf("adding best event, and the cycle repeats!\n");
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


PyObject * py_propose(NetModel_t * p_netmodel, PyObject * args)
{
  double time_low;
  double time_high;
  int det_low;
  int det_high;
  double degree_delta;
  int num_step;
  
  PyObject * eventsobj;
  PyObject * evdetlistobj;
  PyObject * retobj;
  int i;
  int numevents;
  Event_t ** pp_events;
  
  if (!PyArg_ParseTuple(args, "ddiidi", &time_low, &time_high, &det_low,
                        &det_high, &degree_delta, &num_step))
    return NULL;

  pp_events = (Event_t **) calloc(p_netmodel->numdetections,
                                  sizeof(*pp_events));

  numevents = propose_invert_step(p_netmodel, NULL, pp_events, 
				  time_low, time_high, det_low, det_high,
				  degree_delta, num_step,1, 0, NULL);

  if (numevents < 0)
  {
    retobj = PyErr_NoMemory();
    goto free_pp_events;
  }

  convert_events_dets_to_pyobj(p_netmodel->p_earth,
                          (const Event_t **)pp_events,
                          numevents, &eventsobj, &evdetlistobj);
  
  retobj = Py_BuildValue("(OO)", eventsobj, evdetlistobj);

  /* BuildValue increments the ref count so we need to decrement our ref */
  Py_DECREF(eventsobj);
  Py_DECREF(evdetlistobj);
  
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event;
    
    p_event = pp_events[i];

    free_event(p_event);
  }

 free_pp_events:

  free(pp_events);

  return retobj;
}
