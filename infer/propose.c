#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "../netvisa.h"

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

static int propose(NetModel_t * p_netmodel, Event_t **pp_events,
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
  
  int lonidx, latidx, timeidx;
  double lon, lat, time;

  int sitenum;
  int detnum;

  double score;  
  double best_lon, best_lat, best_time, best_score;

  int numevents;
  Event_t * p_event;
  
  p_earth = p_netmodel->p_earth;
  
  numlon = (int) ceil(360.0 / degree_step);
  numlat = numlon /2;
  numtime = (int) ceil((time_high - time_low) / time_step);
  z_step = 2.0 / numlat;
  
  p_bucket_score = (double *)calloc(numlon * numlat * numtime,
                                    sizeof(*p_bucket_score));

  if (!p_bucket_score)
    return -1;

  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  p_site_ttime = (double *) calloc(numsites, sizeof(*p_site_ttime));
  
  /* compute the score of each bucket */
  BEGIN_LON_LOOP;
  BEGIN_LAT_LOOP;
  
  /* compute the distance and azimuth and travel time to each site */
  for (sitenum = 0; sitenum < numsites; sitenum ++)
  {
    p_site_ttime[sitenum] = EarthModel_ArrivalTime(p_earth, lon,
                                                   lat, 0, 0,
                                                   EARTH_PHASE_P, sitenum);
  }
      
  for (detnum = 0; detnum < p_netmodel->numdetections; detnum ++)
  {
    Detection_t * p_det;
    
    p_det = p_netmodel->p_detections + detnum;
  
    if (p_det->phase_det != EARTH_PHASE_P)
      continue;
    
    /* if the event can be detected at this site */
    if (p_site_ttime[p_det->site_det] > 0)
    {
      time = p_det->time_det - p_site_ttime[p_det->site_det];
      timeidx = (int) ((time-time_low) / time_step);

      /* debug leb orid 5288222
      if ((lon > -16) && (lon < -10) && (lat > -16) && (lat < -6))
        printf("detnum %d: lon %.1f lat %.1f time %.1f (%d, %d, %d)\n",
               detnum, lon, lat, time, lonidx, latidx, timeidx);
      */
      /*
      printf("detnum %d site %d iphase %d: lon %.1f lat %.1f time %.1f (%d, "
      "%d, %d)\n",
      detnum, p_det->site_det, p_det->phase_det, lon, lat, time, lonidx, 
      latidx, timeidx);
      */
      
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

  numevents = 0;

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
  
  if (best_score > 0)
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
  
  free(p_site_ttime);
  
  return numevents;
}

PyObject * py_propose(NetModel_t * p_netmodel, PyObject * args)
{
  double time_low;
  double time_high;
  int det_low;
  int det_high;
  double degree_step;
  double time_step;
  
  PyObject * eventsobj;
  PyObject * evdetlistobj;
  PyObject * retobj;
  int i;
  int numevents;
  Event_t ** pp_events;
  
  if (!PyArg_ParseTuple(args, "ddiidd", &time_low, &time_high, &det_low,
                        &det_high, &degree_step, &time_step))
    return NULL;

  pp_events = (Event_t **) calloc(p_netmodel->numdetections,
                                  sizeof(*pp_events));
  
  numevents = propose(p_netmodel, pp_events, time_low, time_high, det_low,
                      det_high, degree_step, time_step);

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
