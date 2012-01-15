#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <pthread.h>

#include "../netvisa.h"
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
  Event_t *p_inv_events;
  Event_t *p_event;
  EarthModel_t *p_earth;
  
  int N;
  double scale;
  Event_t *p_curr_event;
  Event_t *p_best_event;

  int tid;
  int numthreads;
};


/* Hough transform-based method for proposing events */
int propose_hough(NetModel_t * p_netmodel, Event_t **pp_events,
                  double time_low, double time_high, int det_low,
                  int det_high, double degree_step, double time_step)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  
  int numlon;
  int numlat;
  int numtime;
  double z_step;
  
  double * p_bucket_score;               /* numlon x numlat x numtime */

  double * p_site_ttime;                     /* numsites */
  double * p_site_score;                     /* numsites */
  int * p_skip_det;                          /* numdetections */
  
  int lonidx, latidx, timeidx;
  double lon, lat, time;

  int sitenum;
  int detnum;

  double score;  
  double best_lon, best_lat, best_time, best_score;

  int numevents;
  int event_numdet;
  
  Event_t * p_event;
  
  p_earth = p_netmodel->p_earth;
  
  numlon = (int) ceil(360.0 / degree_step);
  numlat = numlon /2;
  numtime = (int) ceil((time_high - time_low) / time_step);
  z_step = 2.0 / numlat;
  
  p_bucket_score = (double *)malloc(numlon * numlat * numtime
                                    * sizeof(*p_bucket_score));

  if (!p_bucket_score)
  {
    printf("can't allocate %d x %d x %d x %d bytes\n", numlon, numlat,
           numtime, (int)sizeof(*p_bucket_score));
    return -1;
  }

  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_site_ttime = (double *) calloc(numsites, sizeof(*p_site_ttime));

  p_site_score = (double *) calloc(numsites, sizeof(*p_site_score));

  p_skip_det = (int *) calloc(p_netmodel->numdetections, sizeof(*p_skip_det));
  
  if (!p_skip_det)
  {
    printf("Can't allocate %d x %d bytes\n", p_netmodel->numdetections,
           (int)sizeof(*p_skip_det));
    free(p_bucket_score);
    return -1;
  }
  
  /* we skip all the non P detections */

  for (detnum = det_low; detnum < det_high; detnum ++)
  {
    Detection_t * p_det;
    
    p_det = p_netmodel->p_detections + detnum;
  
    if (EARTH_PHASE_P == p_det->phase_det)
      p_skip_det[detnum] = 0;

    else
      p_skip_det[detnum] = 1;
  }

  numevents = 0;

  do
  {
    
    /* 
     * First, we will compute the number of detections which hit each bucket 
     */

    memset(p_bucket_score, 0, numlon * numlat * numtime
           * sizeof(*p_bucket_score));
    
    BEGIN_LON_LOOP;
    BEGIN_LAT_LOOP;
  
    /* compute the distance and azimuth and travel time to each site */
    for (sitenum = 0; sitenum < numsites; sitenum ++)
    {
      p_site_ttime[sitenum] = EarthModel_ArrivalTime(p_earth, lon,
                                                     lat, 0, 0,
                                                     EARTH_PHASE_P, sitenum);
    }
      
    for (detnum = det_low; detnum < det_high; detnum ++)
    {
      Detection_t * p_det;
    
      if (p_skip_det[detnum])
        continue;
    
      p_det = p_netmodel->p_detections + detnum;
  
      /* if the event can be detected at this site */
      if (p_site_ttime[p_det->site_det] > 0)
      {
        time = p_det->time_det - p_site_ttime[p_det->site_det];
        timeidx = (int) ((time-time_low) / time_step);

        if ((timeidx >= 0) && (timeidx < numtime))
        {
          BUCKET(lonidx, latidx, timeidx) += 1;
        
          if ((timeidx+1) < numtime)
            BUCKET(lonidx, latidx, timeidx+1) += .5;
        
          if ((timeidx-1) >= 0)
            BUCKET(lonidx, latidx, timeidx-1) += .5;
        }
      }
    }

    END_LAT_LOOP;
    END_LON_LOOP;

    /*
     * Second, we will find the best bucket and construct an event from it
     */

    best_score = best_lon = best_lat = best_time = 0;
  
    BEGIN_LON_LOOP;  
    BEGIN_LAT_LOOP;
    BEGIN_TIME_LOOP;
  
    score = BUCKET(lonidx, latidx, timeidx);
  
    if (score > best_score)
    {
      /*
        printf("current best %.1f: lon %.1f lat %.1f time %.1f (%d, %d, %d)\n",
        score, lon, lat, time, lonidx, latidx, timeidx);
      */
      best_score = score;
      best_lon = lon;
      best_lat = lat;
      best_time = time;

    }
  
    END_TIME_LOOP;
    END_LAT_LOOP;
    END_LON_LOOP;
  
    if (best_score > 6)
    {
      int i;
      int j;
    
      p_event = alloc_event(p_netmodel);
    
      p_event->evlon = best_lon;
      p_event->evlat = best_lat;
      p_event->evdepth = 0;
      p_event->evtime = best_time;
      p_event->evmag = MIN_MAGNITUDE;

      for (i=0; i<numsites; i++)
        for (j=0; j<numtimedefphases; j++)
          p_event->p_num_dets[i * numtimedefphases + j] = 0;
    
      pp_events[numevents++] = p_event;
    }
    else
      break;

    /*
    printf("best event lon %.1f lat %.1f time %.1f bucket %.1f\n",
           p_event->evlon, p_event->evlat, p_event->evtime, best_score);
    */
    
    /* 
     * Third, we will identify the best detections which associate with this
     * event at each site
     */

    for (detnum = det_low; detnum < det_high; detnum ++)
    {
      Detection_t * p_det;
      int poss;
      double detscore;
      int old_detnum;
    
      if (p_skip_det[detnum])
        continue;
    
      p_det = p_netmodel->p_detections + detnum;
  
      /* save the old detnum */
      if (p_event->p_num_dets[p_det->site_det * numtimedefphases 
                              + EARTH_PHASE_P] > 0)
        old_detnum = p_event->p_all_detids[(p_det->site_det * numtimedefphases 
                                            + EARTH_PHASE_P) * MAX_PHASE_DET];
      else
        old_detnum = -1;
      
      p_event->p_num_dets[p_det->site_det * numtimedefphases
                          + EARTH_PHASE_P] = 1;
      p_event->p_all_detids[(p_det->site_det * numtimedefphases 
                             + EARTH_PHASE_P) * MAX_PHASE_DET] = detnum;
      
      poss = score_event_site_phase_simple(p_netmodel, p_event, 
                                           p_det->site_det, 
                                           EARTH_PHASE_P, &detscore);

      if ((1 == poss) && (detscore > 0) 
          && ((-1 == old_detnum) ||(detscore > p_site_score[p_det->site_det])))
      {
        p_site_score[p_det->site_det] = detscore;
      }
      else
      {
        /* restore the old detnum */
        if (-1 == old_detnum)
          p_event->p_num_dets[p_det->site_det * numtimedefphases 
                              + EARTH_PHASE_P] = 0;
        else
        {
          p_event->p_num_dets[p_det->site_det * numtimedefphases 
                              + EARTH_PHASE_P] = 1;
          p_event->p_all_detids[(p_det->site_det * numtimedefphases 
                                 + EARTH_PHASE_P) * MAX_PHASE_DET] = old_detnum;
        }
      }
    }

    /*
     * Fourth, we will skip these best detections in future iterations
     */
    event_numdet = 0;
    for (sitenum = 0; sitenum < numsites; sitenum ++)
    {
      if (p_event->p_num_dets[sitenum * numtimedefphases + EARTH_PHASE_P] > 0)
        detnum = p_event->p_all_detids[(sitenum * numtimedefphases 
                                        + EARTH_PHASE_P) * MAX_PHASE_DET];
      else
        detnum = -1;

      if (detnum != -1)
      {
        p_skip_det[detnum] = 1;
        event_numdet ++;
      }
    }
  } while (event_numdet >= 3);
  
  free(p_site_ttime);
  free(p_site_score);
  free(p_skip_det);
  free(p_bucket_score);
  
  return numevents;
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
  Event_t *p_event = params->p_event;
  Event_t *p_curr_event = params->p_curr_event;

  double scale = params->scale;

  srand(time(NULL) + params->tid);
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

    propose_best_detections(p_netmodel, p_event, det_low, det_high,
                            p_skip_det, 0);

    /* maintain the overall best event */
    if (p_event->evscore > params->p_best_event->evscore)
      copy_event(p_netmodel, params->p_best_event, p_event);

    if (p_event->evscore > p_curr_event->evscore)
      copy_event(p_netmodel, p_curr_event, p_event);

#ifdef SIM_ANNEAL    
    else
    {
      double temp = 20.0 / log(params->N+2);
      
      if (RAND_DOUBLE < exp((p_event->evscore - p_curr_event->evscore) / temp))
        copy_event(p_netmodel, p_event, p_curr_event);
    }
#endif

  }
  return NULL;
}

static void propose_best_event(NetModel_t * p_netmodel,
                               Event_t * p_event,
                               int det_low, int det_high,
                               int * p_skip_det,
                               double time_low, double time_high,
                               double scale, int numthreads)
{
  Event_t * p_best_event;
  Event_t * p_curr_event;
  const int N = 1000;
  
  p_best_event = alloc_event(p_netmodel);
  p_curr_event = alloc_event(p_netmodel);
  
  /* the initial event is the starting event as well as the initial best */
  copy_event(p_netmodel, p_best_event, p_event);
  copy_event(p_netmodel, p_curr_event, p_event);

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
    thread_args[i].p_event = alloc_event(p_netmodel);
    copy_event(p_netmodel, thread_args[i].p_event, p_event);
    thread_args[i].p_curr_event = alloc_event(p_netmodel);
    copy_event(p_netmodel, thread_args[i].p_curr_event, p_curr_event);
    thread_args[i].p_best_event = alloc_event(p_netmodel);
    copy_event(p_netmodel, thread_args[i].p_best_event, p_best_event);
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
    if (thread_args[i].p_best_event->evscore > p_best_event->evscore)
      copy_event(p_netmodel, p_best_event, thread_args[i].p_best_event);
    free_event(thread_args[i].p_event);
    free_event(thread_args[i].p_curr_event);
    free_event(thread_args[i].p_best_event);
  }

  /* return the overall best event */
  copy_event(p_netmodel, p_event, p_best_event);

  free_event(p_curr_event);
  free_event(p_best_event);
  free(threads);
  free(thread_args);
}

/* propose events by inverting detections and keeping the best
 * inverted detections 
 * this version does a timed number of steps */
/* degee delta is the max change in degree proposed for an event
   num_secs is the number of seconds to spend in proposing birth moves */
int propose_invert_timed (NetModel_t * p_netmodel, Event_t **pp_events,
                          double time_low, double time_high, int det_low,
                          int det_high, double degree_delta, int num_secs,
                          int numthreads)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  
  int detnum, inv_detnum;

  int numevents;

  Event_t * p_best_event;
  Event_t * p_event;
  int * p_skip_det;                          /* numdetections */
  
  Event_t * p_inv_events;
  int * p_skip_inv;
  int start_time, end_time;
  
  p_earth = p_netmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_skip_det = (int *) calloc(p_netmodel->numdetections, sizeof(*p_skip_det));
  
  if (!p_skip_det)
  {
    return -1;
  }

  p_inv_events = (Event_t *)calloc(det_high - det_low, sizeof(*p_inv_events));
  if (!p_inv_events)
  {
    return -1;
  }
  
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
      
    p_inv_det = p_netmodel->p_detections + inv_detnum;

    p_event = p_inv_events+det_off;
    
    inv_status = invert_detection(p_earth, p_inv_det, p_event,
                                  0 /* don't perturb */);

    /* if the detection can't be inverted or the inverted event time is
     * outside the bounds of the desired window then skip it */
    if ((0 != inv_status) || (p_event->evtime < time_low)
        || (p_event->evtime > time_high))
    {
      p_skip_inv[det_off] = 1;
    }
    /* otherwise, check if there is an earlier detection which inverts to the
     * same location, and if so then skip this one */
    else
    {
      int inv_detnum2;
      
      for (inv_detnum2 = det_low; inv_detnum2 < inv_detnum; inv_detnum2++)
      {
        Event_t * p_event2;
        int det_off2 = inv_detnum2 - det_low;
        
        /* ensure that the prior detection is not being skipped */
        if (!p_skip_inv[det_off2])
        {
          p_event2 = p_inv_events + det_off2;
          
          /* if the two inverted events are very close to each other then
           * skip this one */
          if ((ABS(p_event->evtime - p_event2->evtime) < 50)
              && (simple_distance_deg(p_event->evlon, p_event->evlat,
                                      p_event2->evlon, p_event2->evlat) < 5))
          {
            p_skip_inv[det_off] = 1;
          }
        }
      }
    }
  }

  p_event = alloc_event(p_netmodel);
  p_best_event = alloc_event(p_netmodel);

  numevents = 0;

  start_time = time(NULL);
  
  do
  {
    int siteid;
    int phase;

    p_best_event->evscore = 0;
    
    for (inv_detnum = det_low; inv_detnum < det_high; inv_detnum ++)
    {
      Detection_t * p_inv_det;
      double trvtime;
      int det_off = inv_detnum - det_low;

      if (p_skip_inv[det_off])
        continue;

      p_event->evmag = RAND_UNIFORM(3.0, 4.0);
      
      /* either propose an event by inverting a detection */
      if (RAND_DOUBLE < .5)
      {
        p_inv_det = p_netmodel->p_detections + inv_detnum;

        p_event->evdepth = 0;

        p_event->evlon = p_inv_events[det_off].evlon
          + RAND_UNIFORM(-degree_delta, degree_delta);
        FIXUP_EVLON(p_event);
        
        p_event->evlat = p_inv_events[det_off].evlat
          + RAND_UNIFORM(-degree_delta, degree_delta);
        FIXUP_EVLAT(p_event);
      
        trvtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                         p_event->evlat, p_event->evdepth, 0,
                                         EARTH_PHASE_P, p_inv_det->site_det);
        if (trvtime < 0)
        {
          p_skip_inv[det_off] = 1;
          continue;
        }
          
        p_event->evtime = p_inv_det->time_det - trvtime;

        if ((p_event->evtime < time_low) || (p_event->evtime > time_high))
        {
          p_skip_inv[det_off] = 1;
          continue;
        }
      }
      /* or sample from the prior */
      else
      {
        EventLocationPrior_Sample(&p_netmodel->event_location_prior,
                                  &p_event->evlon, &p_event->evlat,
                                  &p_event->evdepth);
        
        p_event->evtime = RAND_UNIFORM(time_low, time_high);
      }
      
      /* score this event using the best detections available */
      propose_best_detections(p_netmodel, p_event, det_low, det_high,
                              p_skip_det, 0 /* all phases */);

#ifdef VERBOSE
      printf("detnum %d: ", inv_detnum);
      print_event(p_event);
      print_event_detections(p_earth, p_event);
#endif
      if (p_event->evscore > p_best_event->evscore)
      {
        copy_event(p_netmodel, p_best_event, p_event);

#ifdef NEVER
        printf("CURR BEST: ");
        print_event(p_best_event);
#endif
      }
    }

    /* finished inverting all detections and trying events in a ball around
     * them now let's see if we got something good */
    
    if (p_best_event->evscore > 0)
    {
      /* now, improve this event to take advantage of its detections */
      propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                         p_skip_det, time_low, time_high, 1, numthreads);
      propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                         p_skip_det, time_low, time_high, .1, numthreads);

#ifdef VERBOSE
      printf("BEST:");
      print_event(p_best_event);
      print_event_detections(p_earth, p_best_event);
#endif

      /*
       * we will identify the detections used by the best event and make them
       * off-limits for future events
       */
      /* TODO: also make detections from nearby events off-limits to avoid
       * proposing duplicate events */
      for (siteid = 0; siteid < numsites; siteid ++)
      {
        for (phase = 0; phase < numtimedefphases; phase ++)
        {
          if (p_best_event->p_num_dets[siteid * numtimedefphases + phase] > 0)
          {
            detnum = p_best_event->p_all_detids[(siteid * numtimedefphases 
                                                 + phase) * MAX_PHASE_DET];
          
            p_skip_det[detnum] = 1;
            /* no point inverting these detections either */
            p_skip_inv[detnum - det_low] = 1;
          }
        }
      }
    
      /* add the best event to the list of events */
      pp_events[numevents ++] = p_best_event;
      
      /* and allocate a new one for future events */
      p_best_event = alloc_event(p_netmodel);
    }
    
    end_time = time(NULL);
  } while ((end_time - start_time) < num_secs);

  free_event(p_best_event);
  free_event(p_event);
  free(p_skip_inv);
  free(p_inv_events);
  free(p_skip_det);
  
  return numevents;
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

  NetModel_t * p_netmodel = params->p_netmodel;
  Event_t *p_inv_events = params->p_inv_events;
  Event_t *p_event = params->p_event;
  EarthModel_t * p_earth = params->p_earth;

  int tid = params->tid;

  int inv_detnum, lonidx, latidx, depth;
  double lon, lat, mag;

  /* Loop increments by NUM_THREADS to ensure that each thread gets a
   * different datum */
  for (inv_detnum = det_low+tid; inv_detnum < det_high; 
       inv_detnum += numthreads)
  {
    Detection_t * p_inv_det;
    int det_off = inv_detnum - det_low;

    if (p_skip_inv[det_off])
      continue;

    p_inv_det = p_netmodel->p_detections + inv_detnum;

    /* save the longitude and latitude */
    lon = (p_inv_events + det_off)->evlon;
    lat = (p_inv_events + det_off)->evlat;

    for (mag=3; mag <4.1; mag+=1)
    {
      p_event->evmag = mag;

      for (lonidx=-num_step; lonidx<=num_step; lonidx++)
      {
        p_event->evlon = lon + lonidx * degree_step;
        FIXUP_EVLON(p_event);

        for (latidx=-num_step; latidx<=num_step; latidx ++)
        {
          p_event->evlat = lat + latidx * degree_step;
          FIXUP_EVLAT(p_event);

          for (depth=0; depth<MAX_DEPTH; depth+=300)
          {
            double trvtime;
            p_event->evdepth = depth;

            
            trvtime = EarthModel_ArrivalTime(p_earth, p_event->evlon,
                                             p_event->evlat, p_event->evdepth,0,
                                             EARTH_PHASE_P,p_inv_det->site_det);
            if (trvtime < 0)
              continue;

            p_event->evtime = p_inv_det->time_det - trvtime;

            if ((p_event->evtime < time_low) || (p_event->evtime > time_high))
            {
              continue;
            }

            /* score this event using the best detections available */
            propose_best_detections(p_netmodel, p_event, det_low, det_high,
                                    p_skip_det, 0 /* all phases */);

#ifdef VERBOSE
            if ((!lonidx) && (!latidx))
            {
              printf("detnum %d: ", inv_detnum);
              print_event(p_event);
              print_event_detections(p_earth, p_event);
            }
#endif
            /* Compare to the score in p_best_events with index tid
             * (for this thread) */
            if (p_event->evscore > params->p_best_event->evscore)
            {
              copy_event(p_netmodel, params->p_best_event, p_event);

#ifdef NEVER
              printf("CURR BEST: ");
              print_event(params->p_best_event);
#endif
            }
          }
        }
      }
    }
  }
  return NULL;
}

/* propose events by inverting detections and keeping the best
 * inverted detections 
 * this version does a fixed number of steps per detection */
/* can be very slow when there are a large number of detections, for example:
  python -m utils.check_propose -k 1299822400 -l tohoku -d 2.5 -w 1800  -r 1.2
*/
int propose_invert_step(NetModel_t * p_netmodel, Event_t **pp_events,
                        double time_low, double time_high, int det_low,
                        int det_high, double degree_step, int num_step,
                        int numthreads)
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

  p_earth = p_netmodel->p_earth;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_skip_det = (int *) calloc(p_netmodel->numdetections, sizeof(*p_skip_det));
  
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
    
    p_inv_det = p_netmodel->p_detections + inv_detnum;

    p_event = p_inv_events+det_off;
    
    inv_status = invert_detection(p_earth, p_inv_det, p_event,
                                  0 /* don't perturb */);

    /* if the detection can't be inverted or the inverted event time is
     * outside the bounds of the desired window then skip it */
    if ((0 != inv_status) || (p_event->evtime < time_low)
        || (p_event->evtime > time_high))
    {
      p_skip_inv[det_off] = 1;
    }
  }
  
#ifdef VERBOSE
  {
    int cnt;
    cnt = 0;
    for (inv_detnum = det_low; inv_detnum < det_high; inv_detnum ++)
    {
      if (p_skip_inv[inv_detnum - det_low])
        cnt ++;
    }
    printf("invert_step: %d detection %d skipped\n", det_high-det_low,
           cnt);
  }
#endif
  
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
      thread_args[i].p_inv_events = p_inv_events;
      thread_args[i].p_earth = p_earth;
      thread_args[i].tid = i;
  }

  do
  {
    int siteid;
    int phase;

    /* Initialize what will be the overall best event */
    p_best_event = alloc_event(p_netmodel);
    p_best_event->evscore = 0;

    /* Create threads and have them run the proposal loop */
    for (int i = 0; i < numthreads; i++)
    {
      thread_args[i].p_event = alloc_event(p_netmodel);
      thread_args[i].p_best_event = alloc_event(p_netmodel);
      thread_args[i].p_best_event->evscore = 0;
      pthread_create(&threads[i], NULL, propose_invert_step_helper,
                     (void *) &thread_args[i]);
    }

    /* Wait for all threads to finish */
    for (int i = 0; i < numthreads; i++)
    {
      pthread_join(threads[i], NULL);
    }
	
    /* Get the best event from all threads combined */
    for (int i = 0; i < numthreads; ++i)
    {
      if (thread_args[i].p_best_event->evscore > p_best_event->evscore)
      {
        copy_event(p_netmodel, p_best_event, thread_args[i].p_best_event);
      }
      free_event(thread_args[i].p_best_event);
      free_event(thread_args[i].p_event);
    }
    
    /* finished inverting all detections and trying events in a ball around
     * them now let's see if we got something good */
    if (0 == p_best_event->evscore)
    {
      free_event(p_best_event);
      break;
    }
    
    /* now, improve this event to take advantage of its detections */
    propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                       p_skip_det, time_low, time_high, 1, numthreads);
    propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                       p_skip_det, time_low, time_high, .1, numthreads);

    /* and, once more find the best detections for this event */
    propose_best_detections(p_netmodel, p_best_event, det_low, det_high,
                            p_skip_det, 0 /* all phases */);
    
#ifdef VERBOSE
    printf("BEST:");
    print_event(p_best_event);
    print_event_detections(p_earth, p_best_event);
#endif

    /*
     * we will identify the detections used by the best event and make them
     * off-limits for future events
     */
    for (siteid = 0; siteid < numsites; siteid ++)
    {
      for (phase = 0; phase < numtimedefphases; phase ++)
      {
        if (p_best_event->p_num_dets[siteid * numtimedefphases + phase] > 0)
        {
          detnum = p_best_event->p_all_detids[(siteid * numtimedefphases
                                               + phase)*MAX_PHASE_DET];
          p_skip_det[detnum] = 1;
        }
      }
    }
    
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

//Essentially just the for loop above, along with parameter extraction
/* uniformly propose the event all over the earth */
/* python -m utils.check_propose python -k 1237700200 -r .75 -d 10 -t 10
 * found the event 5288798 but took 5 hours over a 5 minute window
 */
int propose_uniform(NetModel_t * p_netmodel, Event_t **pp_events,
                    double time_low, double time_high, int det_low,
                    int det_high, double degree_step, double time_step,
                    double depth_step, double mag_step, int numthreads)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  
  int numlon;
  int numlat;
  double z_step;

  int detnum;

  int numevents;

  Event_t * p_best_event;
  Event_t * p_event;
  int * p_skip_det;                          /* numdetections */
  
  p_earth = p_netmodel->p_earth;
  
  numlon = (int) ceil(360.0 / degree_step);
  numlat = numlon /2;
  z_step = 2.0 / numlat;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_skip_det = (int *) calloc(p_netmodel->numdetections, sizeof(*p_skip_det));
  
  if (!p_skip_det)
  {
    return -1;
  }

  p_event = alloc_event(p_netmodel);

  numevents = 0;

  do
  {
    int siteid;
    int phase;

    p_best_event = alloc_event(p_netmodel);
    p_best_event->evscore = 0;

    for (p_event->evlon = -180; p_event->evlon < 180;
         p_event->evlon += degree_step)
    {
      double z;
      for (z=-1; z<=1; z+= z_step)
      {
        p_event->evlat = asin(z) * RAD2DEG;
        
        for (p_event->evdepth = MIN_DEPTH; p_event->evdepth <= MAX_DEPTH;
             p_event->evdepth += depth_step)
        {

          for (p_event->evtime = time_low; p_event->evtime <= time_high;
               p_event->evtime += time_step)
          {
            
            for (p_event->evmag = 3; p_event->evmag < 4.1; p_event->evmag += 1)
            {
              /* score this event using the best detections available */
              propose_best_detections(p_netmodel, p_event, det_low, det_high,
                                      p_skip_det, 1);

              if (p_event->evscore > p_best_event->evscore)
              {
                copy_event(p_netmodel, p_best_event, p_event);

#ifdef NEVER
                printf("CURR BEST: ");
                print_event(p_best_event);
#endif
              }
            }
          }
        }
      }
    }
    
    /* now lets see if we found a good event */
    
    if (0 == p_best_event->evscore)
    {
      free_event(p_best_event);
      free_event(p_event);
      break;
    }
    
    /* now, improve this event to take advantage of its detections */
    propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                       p_skip_det, time_low, time_high, 1, numthreads);
    propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                       p_skip_det, time_low, time_high, .1, numthreads);
    
    /*
     * we will identify the detections used by the best event and make them
     * off-limits for future events
     */
    for (siteid = 0; siteid < numsites; siteid ++)
    {
      for (phase = 0; phase < numtimedefphases; phase ++)
      {
        if (p_best_event->p_num_dets[siteid * numtimedefphases + phase] > 0)
        {
          detnum = p_best_event->p_all_detids[(siteid * numtimedefphases 
                                               + phase) * MAX_PHASE_DET];

          p_skip_det[detnum] = 1;
        }
      }
    }
    
    /* add the best event to the list of events */
    pp_events[numevents ++] = p_best_event;
    
  } while (1);

  free(p_skip_det);

  /* free_event(p_event); */

  return numevents;
}

/* python -m utils.check_propose -k 1238096900 -r .6 */
int propose_invert_grid(NetModel_t * p_netmodel, Event_t **pp_events,
                        double time_low, double time_high, int det_low,
                        int det_high, double degree_step, double time_step)
{
  double * p_site_ttime;                     /* numsites */
  int * p_skip_det;                          /* numdetections */

  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  
  int numlon;
  int numlat;
  int numtime;
  double z_step;

  unsigned short * p_bucket_score;     /* numlon x numlat x numtime */

  
  int lonidx, latidx, timeidx;
  double lon, lat, time;

  int detnum, sitenum;

  double bucket_score;  
  double best_lon, best_lat, best_time, best_bucket_score;
  int numevents;
  Event_t * p_event;

  Event_t ** pp_inv_event;
  
  p_skip_det = (int *) calloc(p_netmodel->numdetections, sizeof(*p_skip_det));
  
  p_earth = p_netmodel->p_earth;
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_site_ttime = (double *) malloc(numsites * sizeof(*p_site_ttime));

  numlon = (int) ceil(360.0 / degree_step);
  numlat = numlon /2;
  numtime = (int) ceil((time_high - time_low) / time_step);
  z_step = 2.0 / numlat;
  
  p_bucket_score = (unsigned short *)malloc(numlon * numlat * numtime
                                            * sizeof(*p_bucket_score));

  if (!p_bucket_score)
  {
    printf("can't allocate %d x %d x %d x %d bytes\n", numlon, numlat,
           numtime, (int)sizeof(*p_bucket_score));
    return -1;
  }

  pp_inv_event = (Event_t **)calloc(det_high-det_low, sizeof(*pp_inv_event));
  
  for (detnum=det_low; detnum<det_high; detnum ++)
  {
    Detection_t * p_det;
    int inv_status;

    pp_inv_event[detnum] = alloc_event(p_netmodel);
      
    p_det = p_netmodel->p_detections + detnum;

    inv_status = invert_detection(p_earth, p_det, pp_inv_event[detnum],
                                  0 /* don't perturb */);
    if (0 != inv_status)
    {
      free(pp_inv_event[detnum]);
      pp_inv_event[detnum] = NULL;
    }
  }
  
  
  numevents = 0;
  p_event = alloc_event(p_netmodel);
  
  do {
    /* 
     * First, we will compute the number of detections which hit each bucket 
     */

    memset(p_bucket_score, 0, numlon * numlat * numtime
           * sizeof(*p_bucket_score));

    BEGIN_LON_LOOP;
    BEGIN_LAT_LOOP;

    /* compute the distance to each site */
    for (sitenum = 0; sitenum < numsites; sitenum ++)
    {
      p_site_ttime[sitenum] = EarthModel_ArrivalTime(p_earth, lon,
                                                     lat, 0, 0,
                                                     EARTH_PHASE_P, sitenum);
    }
    
    for (detnum = det_low; detnum < det_high; detnum ++)
    {
      Detection_t * p_det;
      
      if (p_skip_det[detnum] || (!pp_inv_event[detnum]) ||
          (simple_distance_deg(lon, lat, pp_inv_event[detnum]->evlon, 
                               pp_inv_event[detnum]->evlat) > 10))
        continue;
      
      p_det = p_netmodel->p_detections + detnum;

      /* if the event can be detected at this site */
      if (p_site_ttime[p_det->site_det] > 0)
      {
        double evtime;
        
        evtime = p_det->time_det - p_site_ttime[p_det->site_det];

        if ((evtime < time_low) || (evtime >= time_high))
          continue;
        
        /* give a count of 2 to the nearest bucket and a smaller count
        * to nearby buckets */
        timeidx = (int) ((evtime + (time_step/2) - time_low) / time_step);

        if ((timeidx >= 0) && (timeidx < numtime))
        {
          BUCKET(lonidx, latidx, timeidx) += 2;
/*
          if ((timeidx+1) < numtime)
            BUCKET(lonidx, latidx, timeidx+1) += 1;
        
          if ((timeidx-1) >= 0)
            BUCKET(lonidx, latidx, timeidx-1) += 1;
*/


          if ((lonidx==25) && (latidx==213) && (timeidx==7))
          {
            printf("phase %d lon %.1f lat %.1f detnum %d dettime %.1f site %d site trvtime %.1f\n", p_det->phase_det, lon, lat, detnum, p_det->time_det, p_det->site_det, p_site_ttime[p_det->site_det]);
            printf("lonidx %d latidx %d timeidx %d\n", lonidx, latidx, timeidx);
            
          }
        }
        
      }
    }
    
    END_LAT_LOOP;
    END_LON_LOOP;

    /*
     * Second, we will find the best bucket and construct an event from it
     */

    best_bucket_score = best_lon = best_lat = best_time = 0;
  
    BEGIN_LON_LOOP;  
    BEGIN_LAT_LOOP;
    BEGIN_TIME_LOOP;
  
    bucket_score = BUCKET(lonidx, latidx, timeidx);
  
    if (bucket_score > best_bucket_score)
    {
      printf("current best %.1f: lon %.1f lat %.1f time %.1f (%d, %d, %d)\n",
             bucket_score, lon, lat, time, lonidx, latidx, timeidx);
      
#ifdef NEVER
      p_event->evlon = lon;
      p_event->evlat = lat;
      p_event->evdepth = 0;
      p_event->evtime = time;
      p_event->evmag = 3.0;

      propose_best_detections(p_netmodel, p_event, det_low, det_high,
                              p_skip_det, 0 /* all phases */);

      /* now, improve this event to take advantage of its detections */
      propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                         p_skip_det, time_low, time_high, 1);
      propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                         p_skip_det, time_low, time_high, .1);

      if (p_event->evscore > 0)
#endif
      {
        best_bucket_score = bucket_score;
        best_lon = lon;
        best_lat = lat;
        best_time = time;
      }
    }
  
    END_TIME_LOOP;
    END_LAT_LOOP;
    END_LON_LOOP;

    if (best_bucket_score > 2)
    {
      p_event->evlon = best_lon;
      p_event->evlat = best_lat;
      p_event->evdepth = 0;
      p_event->evtime = best_time;
      p_event->evmag = 3.0;

      propose_best_detections(p_netmodel, p_event, det_low, det_high,
                              p_skip_det, 0 /* all phases */);

      pp_events[numevents++] = p_event;
      p_event = alloc_event(p_netmodel);
    }
    else
      break;
    
  } while (0);

  for(detnum=det_low; detnum < det_high; detnum++)
  {
    if (pp_inv_event[detnum])
    {
      free(pp_inv_event[detnum]);
    }
  }
  free(pp_inv_event);
  
  free(p_event);
  free(p_bucket_score);
  free(p_site_ttime);
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

  numevents = propose_invert_step(p_netmodel, pp_events, time_low, time_high, 
                                  det_low, det_high, degree_delta, num_step,1);

#ifdef NEVER
  numevents = propose_invert_timed(p_netmodel, pp_events, time_low, time_high, 
                                   det_low, det_high,degree_delta,num_seconds);
#endif

#ifdef NEVER
  numevents = propose_invert_grid(p_netmodel, pp_events, time_low, time_high, 
                                  det_low, det_high, .5, 100.0);
#endif

#ifdef NEVER
  
  numevents = propose_uniform(p_netmodel, pp_events, time_low, time_high,
                              det_low, det_high, degree_step, 
                              degree_step * 10 /* time_step */,
                              350 /* depth_step */, 2 /* mag_step */);
#endif

  if (numevents < 0)
  {
    retobj = PyErr_NoMemory();
    goto free_pp_events;
  }

  convert_events_to_pyobj(p_netmodel->p_earth,
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
