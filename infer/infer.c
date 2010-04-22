#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "../netvisa.h"

#define DEBUG
/*#define DEBUG2*/

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
  int inv_detnum_wrap;          /* wrap around inverting detections */
  int drop_evnum;                /* event number to be dropped next */
  int write_evnum;               /* event number to be written next */
  
  double world_score;
  int ev_orid_sequence;

  /* static entries */
  int runid;
  int numsamples;
  int window;
  int step;
  int verbose;
  PyObject * write_events_cb;
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

void copy_event(NetModel_t * p_netmodel, Event_t * p_tgt_event,
                const Event_t * p_src_event)
{
  int * p_tgt_detids;
  int numsites;
  int numtimedefphases;
  
  numsites = EarthModel_NumSites(p_netmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_netmodel->p_earth);
 
  /* save the detids pointer */
  p_tgt_detids = p_tgt_event->p_detids;
  
  /* copy the event */
  *p_tgt_event = *p_src_event;

  /* restore the detids pointer */
  p_tgt_event->p_detids = p_tgt_detids;
  
  /* copy the detids */
  memcpy(p_tgt_event->p_detids, p_src_event->p_detids,
         numsites * numtimedefphases * sizeof(*p_src_event->p_detids));

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
  printf("%4.1f E %4.1f N %.0f km %.0f s %.1f mb score %.1f orid %d\n",
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

#ifdef SAVE_RESTORE
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
#endif

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
  if (RAND_DOUBLE < .5)
  {
    EventLocationPrior_Sample(&p_netmodel->event_location_prior, 
                              &p_event->evlon,
                              &p_event->evlat, &p_event->evdepth);
    
    p_event->evtime = RAND_UNIFORM(p_world->low_evtime, p_world->high_evtime);
    p_event->evmag = MIN_MAGNITUDE;
  }
  else 
  {
    /* invert a detection */
    do {
    
      if (p_world->inv_detnum < p_world->low_detnum)
        p_world->inv_detnum = p_world->low_detnum;
      
      else if (p_world->inv_detnum >= p_world->high_detnum)
      {
        p_world->inv_detnum = p_world->low_detnum;
        p_world->inv_detnum_wrap = 1;        /* we have wrapped around */
      }
    
      /* initially, until we have inverted all the detections once we
       * will not perturb the inverts */
      status = invert_detection(p_netmodel->p_earth, 
                                p_netmodel->p_detections + p_world->inv_detnum,
                                p_event,
                                p_world->inv_detnum_wrap /* perturb */);
    
      p_world->inv_detnum ++;
      
    } while ((status != 0) || (p_event->evtime < p_world->low_evtime)
             || (p_event->evtime > p_world->high_evtime));
  }
  
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

/* greedily find the location for each event
 * numchoices = 1 => pick only one new location of each event
 *            > 1 => pick from numchoices ^ 5 locations
 */
static void change_events(NetModel_t * p_netmodel, World_t * p_world,
                          int numchoices)
{
  int STEP_MIN, STEP_MAX, STEP_PLUS;
  
  int evnum;
  
  if (1 == numchoices)
  {
    STEP_MIN = 1;
    STEP_MAX = 2;
    STEP_PLUS = 1;
  }
  else
  {
    assert(numchoices > 1);
    
    STEP_MIN = - (numchoices/2);
    STEP_MAX = numchoices - (numchoices/2);
    STEP_PLUS = 1;
  }
  
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum ++)
  {
    Event_t * p_event;
    Event_t best_event;
    Event_t new_event;

    int lon_step;
    double diff_lon;
    int lat_step;
    double diff_lat;
    int time_step;
    double diff_time;
    int depth_step;
    double diff_depth;
    int mag_step;
    double diff_mag;
    
    p_event = p_world->pp_events[evnum];
    
    if (p_event->evtime < p_world->low_evtime)
      continue;
    
    /* make a copy of the event */
    best_event = *p_event;

    /* try all possible events in a ball around the event */
    new_event = *p_event;

    diff_lon = RAND_UNIFORM(-2, 2);
    diff_lat = RAND_UNIFORM(-2, 2);
    diff_depth = RAND_UNIFORM(-100, 100);
    diff_time = RAND_UNIFORM(-5, 5);
    diff_mag = RAND_UNIFORM(-.5, .5);

    for (lon_step = STEP_MIN; lon_step < STEP_MAX; lon_step += STEP_PLUS)
    {
      /* longitude  from  -180 to 179.9999 with wrap around */
      new_event.evlon = p_event->evlon + diff_lon * lon_step;
      if (new_event.evlon < -180)
        new_event.evlon += 360;
      else if (new_event.evlon >= 180)
        new_event.evlon -= 360;

      for (lat_step = STEP_MIN; lat_step < STEP_MAX; lat_step += STEP_PLUS)
      {
        /* latitude from -90 to 90 with reflection */
        new_event.evlat = p_event->evlat + diff_lat * lat_step;
        if (new_event.evlat < -90)
          new_event.evlat = -180 - new_event.evlat;
        else if (new_event.evlat > 90)
          new_event.evlat = 180 - new_event.evlat;

        for (depth_step = STEP_MIN; depth_step < STEP_MAX;
             depth_step += STEP_PLUS)
        {
          /* depth from MIN_DEPTH to MAX_DEPTH with reflection */
          new_event.evdepth = p_event->evdepth + diff_depth * depth_step;
          if (new_event.evdepth < MIN_DEPTH)
            new_event.evdepth = 2 * MIN_DEPTH - new_event.evdepth;
          else if (new_event.evdepth > MAX_DEPTH)
            new_event.evdepth = 2 * MAX_DEPTH - new_event.evdepth;

          for (time_step = STEP_MIN; time_step < STEP_MAX;
               time_step += STEP_PLUS)
          {
            /* event time from low_evtime to high_evtime with reflection */
            new_event.evtime = p_event->evtime + diff_time * time_step;
            if (new_event.evtime < p_world->low_evtime)
              new_event.evtime = 2 * p_world->low_evtime - new_event.evtime;
            else if (new_event.evtime > p_world->high_evtime)
              new_event.evtime = 2 * p_world->high_evtime - new_event.evtime;

            for (mag_step = STEP_MIN; mag_step < STEP_MAX;
                 mag_step += STEP_PLUS)
            {
              /* magnitude: MIN_MAGNITUDE - MAX_MAGNITUDE with reflection */
              new_event.evmag = p_event->evmag + diff_mag * mag_step;
              if (new_event.evmag < MIN_MAGNITUDE)
                new_event.evmag = 2 * MIN_MAGNITUDE - new_event.evmag;
              else if (new_event.evmag > MAX_MAGNITUDE)
                new_event.evmag = 2 * MAX_MAGNITUDE - new_event.evmag;
              
              new_event.evscore = score_event(p_netmodel, &new_event);

              if (new_event.evscore > best_event.evscore)
                best_event = new_event;
            }
          }
        }
      }
    }
    
    if (best_event.evscore > p_event->evscore)
    {
      int curr_evnum;

#ifdef DEBUG2
      printf("change_events: orid %d score %.1f -> %.1f\n", p_event->orid,
             p_event->evscore, best_event.evscore);
#endif
      
      p_world->world_score += best_event.evscore - p_event->evscore;
      *p_event = best_event;

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
static void change_one_detection(NetModel_t * p_netmodel, World_t * p_world,
  int detnum)
{
  EarthModel_t * p_earth;
  int numtimedefphases;
  Detection_t * p_detection;
  int evnum;
  int best_evnum;
  int best_phaseid;
  double best_score;
  double best_score_delta;
  
  int replaced_detnum;
  
  p_detection = p_netmodel->p_detections + detnum;
    
  /* tx phases are causing a lot of confusion */
  if (EARTH_PHASE_tx == p_detection->phase_det)
    return;

  p_earth = p_netmodel->p_earth;
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
    
  best_evnum = best_phaseid = -1;
  best_score = best_score_delta = 0;
    
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum++)
  {
    Event_t * p_event;
    int phaseid;
    double distance, pred_az;
    
    p_event = p_world->pp_events[evnum];

    /* if the event couldn't possibly have caused the detection then
     * don't bother with it */
    if (fabs(p_event->evtime - p_detection->time_det) > MAX_TRAVEL_TIME)
      continue;
    
    distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                p_detection->site_det);
      
    pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                        p_event->evlat,
                                        p_detection->site_det);
      
    for (phaseid=0; phaseid < numtimedefphases; phaseid ++)
    {
      int old_detnum;
      double score;
      double not_det_prob;
      double score_delta;
      int poss;

      /* pP phases have too much variance, they are not helping ! */
      if (EARTH_PHASE_pP == phaseid)
        continue;

      old_detnum = p_event->p_detids[p_detection->site_det * numtimedefphases
                                     + phaseid];

      /* change the detection of this phase and measure the score */
      p_event->p_detids[p_detection->site_det * numtimedefphases + phaseid]
        = detnum;

      poss = score_event_site_phase(p_netmodel, p_event,
                                    p_detection->site_det, phaseid,
                                    distance, pred_az, &score);

      /* now, compute the probability that this phase was not detected
       * at this site */
      p_event->p_detids[p_detection->site_det * numtimedefphases + phaseid]
        = -1;

      poss = score_event_site_phase(p_netmodel, p_event,
                                    p_detection->site_det, phaseid,
                                    distance, pred_az, &not_det_prob);
      
      score -= not_det_prob;

      /* if we have come across the current event/phase of the detection
       * then we need to change the event phase to noise for now */
      if (old_detnum == detnum)
      {
        p_event->evscore -= score;
        p_world->world_score -= score;

        score_delta = score;
      }
      
      else if (old_detnum == -1)
      {
        score_delta = score;
      }
      
      /* if this phase was already detected at the site then we will have
       * to consider the cost of replacing it */
      else
      {
        double old_score;
        
        p_event->p_detids[p_detection->site_det * numtimedefphases + phaseid]
          = old_detnum;

        poss = score_event_site_phase(p_netmodel, p_event,
                                      p_detection->site_det, phaseid,
                                      distance, pred_az, &old_score);
        old_score -= not_det_prob;
        
        score_delta = score - old_score;
      }
        
      if ((score_delta > 0) && (score > best_score))
      {
        best_evnum = evnum;
        best_phaseid = phaseid;
        best_score = score;
        best_score_delta = score_delta;
      }
    }
  }

  replaced_detnum = -1;
  
  if (best_score > 0)
  {
    Event_t * p_event;
      
    p_event = p_world->pp_events[best_evnum];

#ifdef DEBUG2
    printf("change_detections %d: orid %d site %d phase %d score %f -> %f\n",
           detnum, p_event->orid, p_detection->site_det, best_phaseid,
           p_event->evscore, p_event->evscore + best_score_delta);
#endif
    
    replaced_detnum = p_event->p_detids[p_detection->site_det *numtimedefphases
                                        + best_phaseid];
    
    p_event->p_detids[p_detection->site_det * numtimedefphases 
                      + best_phaseid] = detnum;
    p_event->evscore += best_score_delta;
    p_world->world_score += best_score_delta;
  }

  if (replaced_detnum != -1)
    change_one_detection(p_netmodel, p_world, replaced_detnum);
}

static void change_detections(NetModel_t * p_netmodel, World_t * p_world)
{
  int detnum;
  double old_score;
  
  /* keep changing the detections till something converges */
  do 
  {
    old_score = p_world->world_score;
    
    for (detnum = p_world->low_detnum; detnum < p_world->high_detnum;
         detnum ++)
    {
      change_one_detection(p_netmodel, p_world, detnum);
    }

  } while (fabs(old_score - p_world->world_score) > 1e-4);
}

#ifdef SAVE_RESTORE
static Event_t ** save_world(NetModel_t * p_netmodel, const World_t * p_world,
                             int * p_numsavedevents)
{
  Event_t ** pp_saved_events;
  int numsavedevents;
  int i;
  
  numsavedevents = p_world->high_evnum - p_world->low_evnum;
  
  assert(numsavedevents >= 0);
  
  if (!numsavedevents)
  {
    *p_numsavedevents = 0;
    return NULL;
  }
  
  pp_saved_events = (Event_t **)calloc(numsavedevents,
                                       sizeof(*pp_saved_events));

  for (i=0; i<numsavedevents; i++)
  {
    pp_saved_events[i] = alloc_event(p_netmodel);
    copy_event(p_netmodel, pp_saved_events[i],
               p_world->pp_events[p_world->low_evnum + i]);
  }
  
  *p_numsavedevents = numsavedevents;
  return pp_saved_events;
}

static void restore_world(NetModel_t * p_netmodel, 
                          World_t * p_world, Event_t ** pp_saved_events,
                          int numsavedevents)
{
  int i;

  assert((numsavedevents == p_world->high_evnum - p_world->low_evnum));
  
  for (i=0; i<numsavedevents; i++)
  {
    p_world->world_score += pp_saved_events[i]->evscore 
      - p_world->pp_events[p_world->low_evnum + i]->evscore;
    
    copy_event(p_netmodel, p_world->pp_events[p_world->low_evnum + i],
               pp_saved_events[i]);
  }
}

static void free_saved_events(NetModel_t * p_netmodel, 
                              World_t * p_world, Event_t ** pp_saved_events,
                              int numsavedevents)
{
  int i;

  if (!pp_saved_events)
  {
    assert(!numsavedevents);
    return;
  }

  for (i=0; i<numsavedevents; i++)
  {
    free_event(pp_saved_events[i]);
  }

  free(pp_saved_events);
}

#endif /* SAVE_RESTORE */

static void write_events(NetModel_t * p_netmodel, World_t * p_world)
{
  PyObject * retval;
  double maxtime;
  PyObject * eventsobj;
  PyObject * evdetlistobj;
  int i;
  int numevents;
  
  if (p_world->high_evtime < p_netmodel->end_time)
    maxtime = MAX(p_world->low_evtime - MAX_TRAVEL_TIME,
                  p_netmodel->start_time);
  else
    maxtime = p_world->high_evtime;
  
  /* count the number of events */
  numevents = 0;
  for (i=p_world->write_evnum;
       (i<p_world->high_evnum) && p_world->pp_events[i]->evtime < maxtime;
       i++)
    numevents ++;

  convert_events_to_pyobj(p_netmodel->p_earth, 
                          (const Event_t **) (p_world->pp_events 
                                              + p_world->write_evnum),
                          numevents, &eventsobj, &evdetlistobj);

  if (p_world->verbose)
  {
    for (i=0; i<numevents; i++)
    {
      Event_t * p_event;
      
      p_event = p_world->pp_events[p_world->write_evnum + i];
      
      printf("Write: ");
      print_event(p_event);
    }
  }
  
  retval = PyObject_CallFunction(p_world->write_events_cb, "OOOOid", 
                                 p_netmodel, p_netmodel->p_earth, 
                                 eventsobj, evdetlistobj,
                                 p_world->runid, maxtime);

  p_world->write_evnum += numevents;
  
  Py_DECREF(eventsobj);
  Py_DECREF(evdetlistobj);
  
  if (!retval)
    printf("Warning: can't write objects\n");
  else
    Py_DECREF(retval);
}

static World_t * alloc_world(NetModel_t * p_netmodel)
{
  World_t * p_world;
  
  /* assume that we can't have more events then detections */
  p_world = (World_t *) calloc(1, sizeof(*p_world));
  p_world->maxevents = p_netmodel->numdetections;
  p_world->pp_events = (Event_t **) calloc(p_world->maxevents,
                                           sizeof(*p_world->pp_events));

  return p_world;
}

static void free_world(World_t * p_world)
{
  int i;
  
  for (i=0; i<p_world->high_evnum; i++)
  {
    Event_t * p_event;
    
    p_event = p_world->pp_events[i];

    free_event(p_event);
  }

  free(p_world->pp_events);
  free(p_world);
}

static void infer(NetModel_t * p_netmodel, World_t * p_world)
{
  int i;
  time_t t1;

  /* initialize the window */
  p_world->low_evnum = 0;
  p_world->high_evnum = 0;
  p_world->low_evtime = p_netmodel->start_time;
  p_world->high_evtime = 0;
  p_world->low_detnum = 0;
  p_world->high_detnum = 0;

  do 
  {
    /* initialize high_evtime */
    p_world->high_evtime = MIN(p_world->low_evtime + p_world->window,
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

    /* keep track of whether or not we have wrapped around inverting
     * detections this will trigger further inverts to perturb around
     * the inverted location */
    p_world->inv_detnum = 0;
    p_world->inv_detnum_wrap = 0;

    t1 = time(NULL);
    
    for (i=0; i<p_world->numsamples; i++)
    {
      int j;
      int numdel;
      double old_score;
      Event_t * p_new_event;

      old_score = p_world->world_score;
      
      /* birth move */
      p_new_event = add_event(p_netmodel, p_world);

      change_detections(p_netmodel, p_world);

      if (p_world->world_score < old_score)
      {
        delete_event(p_world, p_new_event);
        free_event(p_new_event);
        change_detections(p_netmodel, p_world);
      }
      else if (p_world->verbose)
      {
        printf("birth+%.1f->%.1f: ", p_world->world_score - old_score,
          p_world->world_score);
        print_event(p_new_event);
      }

      if (p_world->world_score < (old_score - 1e-6))
      {
        printf("after birth: world score has gone down by %.3f -> %.3f\n", 
               old_score - p_world->world_score, p_world->world_score);
      }
      
      old_score = p_world->world_score;
      
      numdel = 0;
      for (j=p_world->low_evnum; j<p_world->high_evnum; j++)
      {
        Event_t * p_old_event;
        
        p_old_event = p_world->pp_events[j];
        
        if (p_old_event->evscore < 0)
        {
          if (p_world->verbose)
          {
            printf("death: ");
            print_event(p_old_event);
          }
          delete_event(p_world, p_old_event);
          free_event(p_old_event);
          numdel ++;
        }
      }

      if (numdel > 0)
      {
        change_detections(p_netmodel, p_world);
      }
      
      if (p_world->world_score < (old_score - 1e-6))
      {
        printf("after death: world score has gone down by %.3f -> %.3f\n", 
               old_score - p_world->world_score, p_world->world_score);
      }
      
      change_events(p_netmodel, p_world, 1);

      change_detections(p_netmodel, p_world);
    };

/*
    change_events(p_netmodel, p_world, 2);
    
    change_detections(p_netmodel, p_world);
*/
    t1 = time(NULL) - t1;
    
    if (p_world->verbose)
    {
      printf("evnum %d-%d evtime %.0f-%.0f detnum %d-%d ela=%ds score=%.1f\n",
             p_world->low_evnum, p_world->high_evnum,
             p_world->low_evtime, p_world->high_evtime,
             p_world->low_detnum, p_world->high_detnum, (int) t1,
             p_world->world_score);
    }
    

    /* move the window forward */
    p_world->low_evtime += p_world->step;

    /* write out any inferred events */
    write_events(p_netmodel, p_world);
    
  } while (p_world->high_evtime < p_netmodel->end_time);


  if (p_world->verbose)
  {
    for (i=0; i<p_world->high_evnum; i++)
    {
      Event_t * p_event;
    
      p_event = p_world->pp_events[i];

      printf("orid %d - score %.1f computed score %.1f\n", p_event->orid, 
             p_event->evscore, score_event(p_netmodel, p_event));
    }
    printf("World Score %.1f\n", p_world->world_score);
  }
}

PyObject * py_infer(NetModel_t * p_netmodel, PyObject * args)
{
  World_t * p_world;
  int runid;
  int numsamples;
  int window;
  int step;
  int verbose;
  PyObject * write_events_cb;

  PyObject * retobj;
  PyObject * eventsobj;
  PyObject * evdetlistobj;
  
  if (!PyArg_ParseTuple(args, "iiiiiO", &runid, &numsamples, &window, &step,
                        &verbose, &write_events_cb))
    return NULL;

  /* allocate the world and initialize the user arguments */
  p_world = alloc_world(p_netmodel);
  p_world->runid = runid;
  p_world->numsamples = numsamples;
  p_world->window = window;
  p_world->step = step;
  p_world->verbose = verbose;
  p_world->write_events_cb = write_events_cb;
  
  infer(p_netmodel, p_world);

  /* convert the world to python structures */
  convert_events_to_pyobj(p_netmodel->p_earth,
                          (const Event_t **)p_world->pp_events,
                          p_world->high_evnum, &eventsobj, &evdetlistobj);
  
  free_world(p_world);

  retobj = Py_BuildValue("(OO)", eventsobj, evdetlistobj);

  /* BuildValue increments the ref count so we need to decrement our ref */
  Py_DECREF(eventsobj);
  Py_DECREF(evdetlistobj);
  
  return retobj;  
}
