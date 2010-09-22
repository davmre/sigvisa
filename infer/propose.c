#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

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
          p_event->p_detids[i * numtimedefphases + j] = -1;
    
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
      old_detnum = p_event->p_detids[p_det->site_det * numtimedefphases 
                                     + EARTH_PHASE_P];

      p_event->p_detids[p_det->site_det * numtimedefphases + EARTH_PHASE_P]
        = detnum;
    
      poss = score_event_site_phase_simple(p_netmodel, p_event, 
                                           p_det->site_det, 
                                           EARTH_PHASE_P, &detscore);

      if ((1 == poss) && (detscore > 0) 
          && ((-1 == old_detnum) ||(detscore > p_site_score[p_det->site_det])))
      {
        p_site_score[p_det->site_det] = detscore;
      }
      else
        /* restore the old detnum */
        p_event->p_detids[p_det->site_det * numtimedefphases + EARTH_PHASE_P] 
          = old_detnum;
    }

    /*
     * Fourth, we will skip these best detections in future iterations
     */
    event_numdet = 0;
    for (sitenum = 0; sitenum < numsites; sitenum ++)
    {
      detnum = p_event->p_detids[sitenum * numtimedefphases + EARTH_PHASE_P];

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

/* brute force -- find all events at each possible time which have a good
 * score */
int propose2(NetModel_t * p_netmodel, Event_t **pp_events,
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
  
  int lonidx, latidx, timeidx;
  double lon, lat, time;

  int detnum;

  int numevents;

  Event_t * p_best_event;
  Event_t * p_event;
  int * p_event_best_detids;                 /* numsites * numtimedefphases */
  double * p_event_best_score;               /* numsites * numtimedefphases */
  int * p_skip_det;                          /* numdetections */
  
  p_earth = p_netmodel->p_earth;
  
  numlon = (int) ceil(360.0 / degree_step);
  numlat = numlon /2;
  numtime = (int) ceil((time_high - time_low) / time_step);
  z_step = 2.0 / numlat;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_event_best_detids = (int *) malloc(numsites * numtimedefphases 
                                       * sizeof (*p_event_best_score));
  p_event_best_score = (double *) malloc(numsites * numtimedefphases 
                                         * sizeof (*p_event_best_score));
  
  p_skip_det = (int *) calloc(p_netmodel->numdetections, sizeof(*p_skip_det));
  
  if (!p_skip_det || !p_event_best_score || !p_event_best_detids)
  {
    return -1;
  }

  p_event = alloc_event(p_netmodel);
  
  numevents = 0;

  do
  {
    int i;
    int siteid;
    int phase;

    p_best_event = alloc_event(p_netmodel);
    p_best_event->evscore = 0;
    
    BEGIN_LON_LOOP;
    BEGIN_LAT_LOOP;
    BEGIN_TIME_LOOP;

    p_event->evlon = lon;
    p_event->evlat = lat;
    p_event->evtime = time;
    p_event->evmag = 3.0;
    p_event->evdepth = 0.0;
    
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
    
      if (p_skip_det[detnum])
        continue;
    
      p_det = p_netmodel->p_detections + detnum;
      siteid = p_det->site_det;
  
      best_phase = -1;
      best_phase_score = 0;

      for (phase=0; phase < numtimedefphases; phase++)
      {
        double distance, pred_az;
        int poss;
        double detscore;

        distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                    siteid);

        pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                            p_event->evlat, siteid);
  
        p_event->p_detids[siteid * numtimedefphases + phase] = detnum;

        poss = score_event_site_phase(p_netmodel, p_event, siteid, phase,
                                      distance, pred_az, &detscore);

        if (poss && (detscore > 0)
            && ((-1 == best_phase) || (detscore > best_phase_score)))
        {
          best_phase = phase;
          best_phase_score = detscore;
        }
      }

      if ((-1 != best_phase) 
          && ((-1 == p_event_best_detids[siteid * numtimedefphases 
                                         + best_phase])
              || (best_phase_score 
                  > p_event_best_score[siteid * numtimedefphases 
                                       + best_phase])))
      {
        p_event_best_detids[siteid * numtimedefphases + best_phase] = detnum;
        p_event_best_score[siteid * numtimedefphases + best_phase] 
          = best_phase_score;
      }
    }
  
    /* score the best such event */
    for (i=0; i < numsites * numtimedefphases; i++)
    {
      p_event->p_detids[i] = p_event_best_detids[i];
    }
    p_event->evscore = score_event(p_netmodel, p_event);
    
    if (p_event->evscore > p_best_event->evscore)
    {
      copy_event(p_netmodel, p_best_event, p_event);

      printf("CURR BEST: ");
      print_event(p_best_event);
    }
    
    END_LON_LOOP;  
    END_LAT_LOOP;
    END_TIME_LOOP;
  
    if (0 == p_best_event->evscore)
    {
      free_event(p_best_event);
      free_event(p_event);
      break;
    }
    
    /*
     * we will identify the detections used by the best event and make them
     * off-limits for future events
     */
    for (siteid = 0; siteid < numsites; siteid ++)
    {
      for (phase = 0; phase < numtimedefphases; phase ++)
      {
        detnum = p_best_event->p_detids[siteid * numtimedefphases + phase];

        if (detnum != -1)
        {
          p_skip_det[detnum] = 1;
        }
      }
    }
    
    /* add the best event to the list of events */
    pp_events[numevents ++] = p_best_event;
    
  } while (1);

  free(p_event_best_detids);
  free(p_event_best_score);
  free(p_skip_det);
  
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

    /* tx phases are causing a lot of confusion */
    /*
    if (EARTH_PHASE_tx == p_det->phase_det)
      continue;
    */
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
      
      /* pP phases have too much variance, they are not helping ! */
      if (EARTH_PHASE_pP == phase)
        continue;
              
      p_event->p_detids[siteid * numtimedefphases + phase] = detnum;

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
    p_event->p_detids[i] = p_event_best_detids[i];
  }
  p_event->evscore = score_event(p_netmodel, p_event);

  free(p_event_best_detids);
  free(p_event_best_score);
}

/* find the best event which explains the given detections */
static void propose_best_event(NetModel_t * p_netmodel,
                               Event_t * p_event,
                               int det_low, int det_high,
                               int * p_skip_det,
                               double time_low, double time_high,
                               double scale)
{
  double init_lon, init_lat, init_time;
  int lonidx, latidx, timeidx;
  Event_t * p_best_event;
  
  p_best_event = alloc_event(p_netmodel);
  
  /* first make a copy of the current event */
  copy_event(p_netmodel, p_best_event, p_event);  
  
  init_lon = p_event->evlon;
  init_lat = p_event->evlat;
  init_time = p_event->evtime;

  /* try to improve the existing event */
  for (lonidx = -2; lonidx < 3; lonidx ++)
  {
    p_event->evlon = init_lon + lonidx * 2.5 * scale;
    FIXUP_EVLON(p_event);
    for (latidx = -2; latidx < 3; latidx ++)
    {
      p_event->evlat = init_lat + latidx * 2.5 * scale;
      FIXUP_EVLAT(p_event);

      for (timeidx = -2; timeidx < 3; timeidx ++)
      {
        p_event->evtime = init_time + timeidx * 25.0 * scale;
        if ((p_event->evtime < time_low) || (p_event->evtime > time_high))
          continue;

        for (p_event->evdepth = 0; p_event->evdepth < MAX_DEPTH;
             p_event->evdepth += (MAX_DEPTH/4))
        {

          for (p_event->evmag = MIN_MAGNITUDE;
               p_event->evmag < (MIN_MAGNITUDE+4);
               p_event->evmag += 1)
          {
            propose_best_detections(p_netmodel, p_event, det_low, det_high,
                                    p_skip_det, 0);
            if (p_event->evscore > p_best_event->evscore)
            {
              copy_event(p_netmodel, p_best_event, p_event);
            }
          }
        }
      }
    }
  }

  /* finally, return the best event */
  copy_event(p_netmodel, p_event, p_best_event);
  
  free_event(p_best_event);
}

/* propose events by inverting detections and keeping the best
 * inverted detections */
int propose_invert(NetModel_t * p_netmodel, Event_t **pp_events,
                   double time_low, double time_high, int det_low,
                   int det_high, double degree_step, int num_step)
{
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  
  int lonidx, latidx;
  double lon, lat, mag;

  int detnum;

  int numevents;

  Event_t * p_best_event;
  Event_t * p_event;
  int * p_skip_det;                          /* numdetections */
  
  p_earth = p_netmodel->p_earth;
  
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
    int inv_detnum;

    p_best_event = alloc_event(p_netmodel);
    p_best_event->evscore = 0;
    
    for (inv_detnum = det_low; inv_detnum < det_high; inv_detnum ++)
    {
      Detection_t * p_inv_det;
      int inv_status;
      
      p_inv_det = p_netmodel->p_detections + inv_detnum;
      
      inv_status = invert_detection(p_earth, p_inv_det, p_event,
                                    0 /* don't perturb */);
      if ((0 != inv_status) || (p_event->evtime < time_low)
          || (p_event->evtime > time_high))
      {
        continue;
      }

      /* save the longitude and latitude */
      lon = p_event->evlon;
      lat = p_event->evlat;
      
      for (mag=3; mag <3.1; mag+=1)
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
    /* finished inverting all detections and trying events in a ball around
     * them now let's see if we got something good */
    
    if (0 == p_best_event->evscore)
    {
      free_event(p_best_event);
      free_event(p_event);
      break;
    }
    
    /* now, improve this event to take advantage of its detections */
    propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                       p_skip_det, time_low, time_high, 1);
    propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                       p_skip_det, time_low, time_high, .1);

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
        detnum = p_best_event->p_detids[siteid * numtimedefphases + phase];

        if (detnum != -1)
        {
          p_skip_det[detnum] = 1;
        }
      }
    }
    
    /* add the best event to the list of events */
    pp_events[numevents ++] = p_best_event;
    
  } while (1);

  free(p_skip_det);
  
  return numevents;
}

/* uniformly propose the event all over the earth */
/* python -m utils.check_propose python -k 1237700200 -r .75 -d 10 -t 10
 * found the event 5288798 but took 5 hours over a 5 minute window
 */
int propose_uniform(NetModel_t * p_netmodel, Event_t **pp_events,
                    double time_low, double time_high, int det_low,
                    int det_high, double degree_step, double time_step,
                    double depth_step, double mag_step)
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
                       p_skip_det, time_low, time_high, 1);
    propose_best_event(p_netmodel, p_best_event, det_low, det_high,
                       p_skip_det, time_low, time_high, .1);
    
    /*
     * we will identify the detections used by the best event and make them
     * off-limits for future events
     */
    for (siteid = 0; siteid < numsites; siteid ++)
    {
      for (phase = 0; phase < numtimedefphases; phase ++)
      {
        detnum = p_best_event->p_detids[siteid * numtimedefphases + phase];

        if (detnum != -1)
        {
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
  double degree_step;
  int num_step;
  
  PyObject * eventsobj;
  PyObject * evdetlistobj;
  PyObject * retobj;
  int i;
  int numevents;
  Event_t ** pp_events;
  
  if (!PyArg_ParseTuple(args, "ddiidi", &time_low, &time_high, &det_low,
                        &det_high, &degree_step, &num_step))
    return NULL;

  pp_events = (Event_t **) calloc(p_netmodel->numdetections,
                                  sizeof(*pp_events));

  numevents = propose_invert(p_netmodel, pp_events, time_low, time_high, 
                             det_low, det_high, degree_step, num_step);

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
