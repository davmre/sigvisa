#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "../netvisa.h"

#define INFER_WINDOW_SIZE 1800
#define INFER_WINDOW_STEP  300

#define DEBUG

typedef struct World_t
{
  /* the inferred events, events be store in ascending order of time */
  int maxevents;
  Event_t ** pp_events;
  
  /* events below this number will not be modified since any detection
   * at low_detnum or higher couldn't have been caused by those below
   * low_evnum
   */
  int low_evnum;
  
  /* the maximum number of events in the world */
  int high_evnum;
  
  /* the minimum time that we will hypothesize for new events */
  double low_evtime;

  /* the maximum time that we will hypothesize events */
  double high_evtime;
  
  /* this is the earliest detection which could have affected a new
   * event */
  int low_detnum;
  
  /* detections above this will not be looked at since they couldn't have been
   * caused by any event currently being hypothesized 
   * high_detnum are all detections below high_evtime + MAX_TRAVEL_TIME */
  int high_detnum;

  /*

  low_evnum             low_evtime  high_evnum   high_evtime
     |                      |           |          |
     |                      |           |          |
     v                      v           v          v
     <-- MAX_TRAVEL_TIME --> <--  WINDOW_SIZE   --> <-- MAX_TRAVEL_TIME -->
                            ^                                              ^
                            |                                              |
                            |                                              |
                        low_detnum                                  high_detnum
                                                                              
    The window will move forward in units of WINDOW_STEP                      

  */

  int inv_detnum;           /* detection number to be inverted next */
  int drop_evnum;                /* event number to be dropped next */
  
  double world_score;
  int ev_orid_sequence;
} World_t;

static Event_t * alloc_event(NetModel_t * p_netmodel)
{
  Event_t * p_event;
  int numsites;
  int numtimedefphases;
  
  p_event = (Event_t *) calloc(1, sizeof(*p_event));
  
  numsites = EarthModel_NumSites(p_netmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_netmodel->p_earth);
 
  p_event->p_detids = (int *)malloc(numsites * numtimedefphases *
                                    sizeof(*p_event->p_detids));

  return p_event;
}

static void free_event(Event_t * p_event)
{
  free(p_event->p_detids);
  free(p_event);
}

static void insert_event(NetModel_t * p_netmodel,
                         World_t * p_world, Event_t * p_event)
{
  int numsites;
  int numtimedefphases;
  int j;
  int evnum;
  
  numsites = EarthModel_NumSites(p_netmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_netmodel->p_earth);

  /* initialize all detections to -1, i.e. no detection */
  for (j=0; j<numsites * numtimedefphases; j++)
    p_event->p_detids[j] = -1; 

  p_event->evscore = score_event(p_netmodel, p_event);
  p_world->world_score += p_event->evscore;

  /* insert from the end */
  for(evnum = p_world->high_evnum-1; 
      (evnum >= 0) && (p_world->pp_events[evnum]->evtime > p_event->evtime);
      evnum --)
  {
    p_world->pp_events[evnum+1] = p_world->pp_events[evnum];
  }

  p_world->pp_events[evnum+1] = p_event;
  
  p_world->high_evnum ++;
  
  assert(p_world->high_evnum < p_world->maxevents);
}

#ifdef DEBUG
static void print_event(const Event_t * p_event)
{
  printf("%4.1f E %4.1f N %.0f km %.0f s %.1f mb score %.0f orid %d\n",
         p_event->evlon, p_event->evlat, p_event->evdepth,
         p_event->evtime, p_event->evmag, p_event->evscore,
         p_event->orid);
}
#endif /* DEBUG */

static void delete_event(World_t * p_world, Event_t * p_event)
{
  int evnum;
  
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum++)
  {
    if (p_world->pp_events[evnum] == p_event)
      break;
  }
  
  assert(evnum < p_world->high_evnum);

  p_world->world_score -= p_world->pp_events[evnum]->evscore;
  
  while (evnum < (p_world->high_evnum-1))
  {
    p_world->pp_events[evnum] = p_world->pp_events[evnum + 1];
    evnum ++;
  }
  
  p_world->high_evnum --;
  
  assert(p_world->high_evnum >= 0);
  
  p_world->low_evnum = MIN(p_world->low_evnum, p_world->high_evnum);
}

static Event_t * drop_event(NetModel_t * p_netmodel, World_t * p_world)
{
  if (p_world->drop_evnum < p_world->low_evnum)
    p_world->drop_evnum = p_world->low_evnum;
  
  while((p_world->drop_evnum < p_world->high_evnum)
        && (p_world->pp_events[p_world->drop_evnum]->evtime
            < p_world->low_evtime))
    p_world->drop_evnum ++;
  
  if (p_world->drop_evnum < p_world->high_evnum)  
  {
    Event_t * p_old_event;

    p_old_event = p_world->pp_events[p_world->drop_evnum];

    delete_event(p_world, p_old_event);

    return p_old_event;
  }
  else
  {
    p_world->drop_evnum = p_world->low_evnum;

    return NULL;
  }
}

static Event_t * add_event(NetModel_t * p_netmodel, World_t * p_world)
{
  int status;
  Event_t * p_event;
  
  p_event = alloc_event(p_netmodel);
  
  /* sample the event attributes uniformly */
  /*
  p_event->evlon = RAND_UNIFORM(-180, 180);
  p_event->evlat = asin(RAND_UNIFORM(-1, 1)) * RAD2DEG;
  p_event->evdepth = RAND_UNIFORM(MIN_DEPTH, MAX_DEPTH);
  */

  /* sample the event from the prior */
  /*
  EventLocationPrior_Sample(&p_netmodel->event_location_prior, &p_event->evlon,
                            &p_event->evlat, &p_event->evdepth);
  
  p_event->evtime = RAND_UNIFORM(p_world->low_evtime, p_world->high_evtime);
  p_event->evmag = RAND_UNIFORM(MIN_MAGNITUDE, MAX_MAGNITUDE);
  */

  /* invert a detection */
  do {
    
    if ((p_world->inv_detnum < p_world->low_detnum)
        || (p_world->inv_detnum >= p_world->high_detnum))
      p_world->inv_detnum = p_world->low_detnum;
    
    status = invert_detection(p_netmodel->p_earth, 
                              p_netmodel->p_detections + p_world->inv_detnum,
                              p_event);
    
    p_world->inv_detnum ++;
    
  } while ((status != 0) || (p_event->evtime < p_world->low_evtime)
           || (p_event->evtime > p_world->high_evtime));
  
  p_event->orid = p_world->ev_orid_sequence ++;
 
  insert_event(p_netmodel, p_world, p_event);

  /*
  printf("add_event: orid %d score %.1f\n", p_event->orid, p_event->evscore);
  score_world(p_netmodel, 1, p_event, 1);
  */
  /*
    printf("added event:\n");
  printf("lon %.1f lat %.1f depth %.1f time %.1f mb %.1f\n",
         p_event->evlon, p_event->evlat, p_event->evdepth, p_event->evtime,
         p_event->evmag);
  */

  return p_event;
}

/* greedily find the location for each event */
static void change_events(NetModel_t * p_netmodel, World_t * p_world)
{
  int evnum;
  
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum ++)
  {
    Event_t * p_event;
    Event_t new_event;
    
    p_event = p_world->pp_events[evnum];
    
    if (p_event->evtime < p_world->low_evtime)
      continue;
    
    /* make a copy of the event */
    new_event = *p_event;

    /* modify the copy */

    /* longitude  from  -180 to 179.9999 with wrap around */
    new_event.evlon = RAND_UNIFORM(new_event.evlon - 2, new_event.evlon + 2);
    if (new_event.evlon < -180)
      new_event.evlon += 360;
    else if (new_event.evlon >= 180)
      new_event.evlon -= 360;

    /* latitude from -90 to 90 with reflection */
    new_event.evlat = RAND_UNIFORM(new_event.evlat - 2, new_event.evlat + 2);
    if (new_event.evlat < -90)
      new_event.evlat = -180 - new_event.evlat;
    else if (new_event.evlat > 90)
      new_event.evlat = 180 - new_event.evlat;

    /* latitude from MIN_DEPTH to MAX_DEPTH with reflection */
    new_event.evdepth = RAND_UNIFORM(new_event.evdepth - 100, 
                                     new_event.evdepth + 100);
    if (new_event.evdepth < MIN_DEPTH)
      new_event.evdepth = 2 * MIN_DEPTH - new_event.evdepth;
    else if (new_event.evdepth > MAX_DEPTH)
      new_event.evdepth = 2 * MAX_DEPTH - new_event.evdepth;

    /* event time from low_evtime to high_evtime with reflection */
    new_event.evtime = RAND_UNIFORM(new_event.evtime - 5, 
                                    new_event.evtime + 5);
    if (new_event.evtime < p_world->low_evtime)
      new_event.evtime = 2 * p_world->low_evtime - new_event.evtime;
    else if (new_event.evtime > p_world->high_evtime)
      new_event.evtime = 2 * p_world->high_evtime - new_event.evtime;

    /* magnitude from MIN_MAGNITUDE to MAX_MAGNITUDE with reflection */
    new_event.evmag = RAND_UNIFORM(new_event.evmag - .5, 
                                   new_event.evmag + .5);
    if (new_event.evmag < MIN_MAGNITUDE)
      new_event.evmag = 2 * MIN_MAGNITUDE - new_event.evmag;
    else if (new_event.evmag > MAX_MAGNITUDE)
      new_event.evmag = 2 * MAX_MAGNITUDE - new_event.evmag;


    new_event.evscore = score_event(p_netmodel, &new_event);
    
    if (new_event.evscore > p_event->evscore)
    {
      int curr_evnum;
      
      /*
      printf("change_events: orid %d score %.1f -> %.1f\n", p_event->orid,
             p_event->evscore, new_event.evscore);
      */
      p_world->world_score += new_event.evscore - p_event->evscore;
      *p_event = new_event;

      /* we might need to move this event around to keep the list of
       * events sorted by time */
      curr_evnum = evnum;
      while(1)
      {
        if (((curr_evnum-1)>=0) && 
            (p_world->pp_events[curr_evnum-1]->evtime 
             > p_world->pp_events[curr_evnum]->evtime))
        {
          Event_t * temp;
          temp = p_world->pp_events[curr_evnum-1];
          p_world->pp_events[curr_evnum-1] = p_world->pp_events[curr_evnum];
          p_world->pp_events[curr_evnum] = temp;
          
          curr_evnum --;
        }
        else if (((curr_evnum+1) < p_world->high_evnum) && 
                 (p_world->pp_events[curr_evnum+1]->evtime 
                  < p_world->pp_events[curr_evnum]->evtime))
        {
          Event_t * temp;
          temp = p_world->pp_events[curr_evnum+1];
          p_world->pp_events[curr_evnum+1] = p_world->pp_events[curr_evnum];
          p_world->pp_events[curr_evnum] = temp;
          
          curr_evnum ++;
        }
        else
          break;
      }
    }
  }
}

/* greedily find the best event-phase for a detection */
static void change_detections(NetModel_t * p_netmodel, World_t * p_world)
{
  EarthModel_t * p_earth;
  int detnum;
  int numtimedefphases;

  p_earth = p_netmodel->p_earth;
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  
  for (detnum = p_world->low_detnum; detnum < p_world->high_detnum; detnum ++)
  {
    Detection_t * p_detection;
    int evnum;
    int best_evnum;
    int best_phaseid;
    double best_score;
    
    best_evnum = best_phaseid = -1;
    best_score = 0;
    
    p_detection = p_netmodel->p_detections + detnum;

    /* tx phases are causing a lot of confusion */
    if (EARTH_PHASE_tx == p_detection->phase_det)
      continue;

    for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum++)
    {
      Event_t * p_event;
      int phaseid;
      double distance, pred_az;
    
      p_event = p_world->pp_events[evnum];

      distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                  p_detection->site_det);
      
      pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                          p_event->evlat,
                                          p_detection->site_det);
      
      for (phaseid=0; phaseid < numtimedefphases; phaseid ++)
      {
        int old_detnum;
        double score;
        double old_score;
        int poss;
        int replace;

        /* pP phases have too much variance, they are not helping ! */
        if (EARTH_PHASE_pP == phaseid)
          continue;

        old_detnum = p_event->p_detids[p_detection->site_det * numtimedefphases
                                       + phaseid];

        /* if this detection was already associated with an event-phase then
         * clear it out */
        if (old_detnum == detnum)
        {
          replace = 1;
          old_detnum = -1;
        }
        else
          replace = 0;

        p_event->p_detids[p_detection->site_det * numtimedefphases + phaseid]
          = detnum;

        poss = score_event_site_phase(p_netmodel, p_event,
                                      p_detection->site_det, phaseid,
                                      distance, pred_az, &score);

        /* restore old association */
        p_event->p_detids[p_detection->site_det * numtimedefphases + phaseid]
          = old_detnum;

        poss = score_event_site_phase(p_netmodel, p_event,
                                      p_detection->site_det, phaseid,
                                      distance, pred_az, &old_score);
        
        score -= old_score;
     
        if (replace)
        {
          p_event->evscore -= score;
          p_world->world_score -= score;
        }
        
        if (score > best_score)
        {
          best_evnum = evnum;
          best_phaseid = phaseid;
          best_score = score;
        }
      }
    }

    if (best_score > 0)
    {
      Event_t * p_event;
      
      p_event = p_world->pp_events[best_evnum];

#ifdef NEVER
      if (2 == detnum)
      printf("change_detections: orid %d score %f -> %f\n", p_event->orid,
             p_event->evscore, p_event->evscore + best_score);
#endif
      
      p_event->p_detids[p_detection->site_det * numtimedefphases 
                        + best_phaseid] = detnum;
      p_event->evscore += best_score;
      p_world->world_score += best_score;
    }
  }
}


PyObject * infer(NetModel_t * p_netmodel, int numsamples)
{
  int i;
  World_t * p_world;
  PyObject * retobj;
  PyObject * eventsobj;
  npy_intp dims[2];
  PyObject * evdetlistobj;
  time_t t1;

  t1 = time(NULL);

  /* initialize world */
  /* assume that we can't have more events then detections */
  p_world = (World_t *) calloc(1, sizeof(*p_world));
  p_world->maxevents = p_netmodel->numdetections;
  p_world->pp_events = (Event_t **) calloc(p_world->maxevents,
                                           sizeof(*p_world->pp_events));
  
  p_world->low_evnum = 0;
  p_world->high_evnum = 0;
  p_world->low_evtime = p_netmodel->start_time;
  p_world->high_evtime = 0;
  p_world->low_detnum = 0;
  p_world->high_detnum = 0;

  do 
  {
    /* initialize high_evtime */
    p_world->high_evtime = MIN(p_world->low_evtime + INFER_WINDOW_SIZE,
                               p_netmodel->end_time);

    /* initialize low_evnum */
    for ( ; p_world->low_evnum < p_world->high_evnum; p_world->low_evnum ++)
    {
      Event_t * p_event;
      
      p_event = p_world->pp_events[p_world->low_evnum];

      if (p_event->evtime >= (p_world->low_evtime - MAX_TRAVEL_TIME))
        break;
    }

    /* initialize low_detnum */
    for ( ; p_world->low_detnum < p_netmodel->numdetections;
          p_world->low_detnum ++)
    {
      Detection_t * p_det;
      
      p_det = p_netmodel->p_detections + p_world->low_detnum;
      
      if (p_det->time_det >= p_world->low_evtime)
        break;
    }

    /* initialize high_detnum */
    for( ; p_world->high_detnum < p_netmodel->numdetections;
         p_world->high_detnum ++)
    {
      Detection_t * p_det;
      
      p_det = p_netmodel->p_detections + p_world->high_detnum;
      
      if (p_det->time_det >= (p_world->high_evtime + MAX_TRAVEL_TIME))
        break;
    }


    t1 = time(NULL);
    
    for (i=0; i<(numsamples * INFER_WINDOW_STEP); i++)
    {
      /* birth move */
      if (RAND_DOUBLE < .5)
      {
        int j;
        double old_score;
        Event_t * p_new_event;
        
        old_score = p_world->world_score;
        
        /* add an event */
        p_new_event = add_event(p_netmodel, p_world);
        
        /* change the rest of the world to try and use this event */
        for (j=0; j<2; j++)
        {
          change_detections(p_netmodel, p_world);
          
          change_events(p_netmodel, p_world);
        }

        /* if the new event didn't help then get rid of it */
        if (p_world->world_score < old_score)
        {
          delete_event(p_world, p_new_event);
          free_event(p_new_event);
        }
#ifdef DEBUG
        else
        {
          printf("birth + %.0f: ", p_world->world_score - old_score);
          print_event(p_new_event);
        }
#endif /* DEBUG */
      }
      /* death move */
      else
      {
        double old_score;
        Event_t * p_old_event;

        old_score = p_world->world_score;
        
        /* try to drop an event */
        p_old_event = drop_event(p_netmodel, p_world);
        
        if (p_old_event)
        {
          int j;
          /* change the rest of the world to adjust to one fewer event */
          for (j=0; j<2; j++)
          {
            change_detections(p_netmodel, p_world);
            
            change_events(p_netmodel, p_world);
          }

          /* if dropping the event didn't help then add it back */
          if (p_world->world_score < old_score)
          {
            insert_event(p_netmodel, p_world, p_old_event);
          }
          /* otherwise, get rid of the event permanently */
          else
          {
#ifdef DEBUG
            printf("death + %.0f: ", p_world->world_score - old_score);
            print_event(p_old_event);
#endif /*DEBUG*/
            free_event(p_old_event);
          }
        }
      }

      change_detections(p_netmodel, p_world);

      change_events(p_netmodel, p_world);
    }

    printf("evnum %d-%d evtime %.0f-%.0f detnum %d-%d ela=%ds score=%.1f\n",
           p_world->low_evnum, p_world->high_evnum,
           p_world->low_evtime, p_world->high_evtime,
           p_world->low_detnum, p_world->high_detnum, (int)(time(NULL)-t1),
           p_world->world_score);
    

    /* move the window forward */
    p_world->low_evtime += INFER_WINDOW_STEP;
    
  } while (p_world->high_evtime < p_netmodel->end_time);

  printf("World Score %.1f\n", p_world->world_score);
  
  /* now create an array of the events */
  dims[0] = p_world->high_evnum;
  dims[1] = EV_NUM_COLS;
  eventsobj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  
  /* and a list of event detections */
  evdetlistobj = PyList_New(0);
  
  for (i=0; i<p_world->high_evnum; i++)
  {
    PyObject * detlistobj;
    Event_t * p_event;
    p_event = p_world->pp_events[i];
    int numsites;
    int numtimedefphases;
    int siteid;
    int phaseid;

    ARRAY2(eventsobj, i, EV_LON_COL) = p_event->evlon;
    ARRAY2(eventsobj, i, EV_LAT_COL) = p_event->evlat;
    ARRAY2(eventsobj, i, EV_DEPTH_COL) = p_event->evdepth;
    ARRAY2(eventsobj, i, EV_TIME_COL) = p_event->evtime;
    ARRAY2(eventsobj, i, EV_MB_COL) = p_event->evmag;
    ARRAY2(eventsobj, i, EV_ORID_COL) = (double) p_event->orid;

    printf("orid %d - score %.1f computed score %.1f\n", p_event->orid, 
           p_event->evscore, score_event(p_netmodel, p_event));
    
    detlistobj = PyList_New(0);
    
    /* copy over the (phaseid, detnum) of the event */
    numsites = EarthModel_NumSites(p_netmodel->p_earth);
    numtimedefphases = EarthModel_NumTimeDefPhases(p_netmodel->p_earth);
    for (siteid = 0; siteid < numsites; siteid ++)
    {
      for (phaseid = 0; phaseid < numtimedefphases; phaseid ++)
      {
        int detnum;
        
        detnum = p_event->p_detids[siteid * numtimedefphases + phaseid];

        if (detnum != -1)
        {
          PyObject * phase_det_obj;
          
          phase_det_obj = Py_BuildValue("(ii)", phaseid, detnum);
          
          PyList_Append(detlistobj, phase_det_obj);
          /* List Append increments the refcount so we need to
           * decrement our ref */
          Py_DECREF(phase_det_obj);
        }
      }
    }

    PyList_Append(evdetlistobj, detlistobj);
    /* List Append increments the refcount so we need to decrement our ref */
    Py_DECREF(detlistobj);

    free_event(p_event);
  }

  retobj = Py_BuildValue("(OO)", eventsobj, evdetlistobj);

  /* BuildValue increments the ref count so we need to decrement our ref */
  Py_DECREF(eventsobj);
  Py_DECREF(evdetlistobj);
  
  free(p_world->pp_events);
  free(p_world);
  
  return retobj;
}
