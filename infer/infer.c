#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "../netvisa.h"

#define DEBUG
/*
#define VERBOSE
*/

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

#define FIXUP_EVDEPTH(p_event)                                          \
  do {                                                                  \
    if ((p_event)->evdepth < MIN_DEPTH)                                 \
      (p_event)->evdepth = 2 * MIN_DEPTH - (p_event)->evdepth;          \
    else if ((p_event)->evdepth > MAX_DEPTH)                            \
      (p_event)->evdepth = 2 * MAX_DEPTH - (p_event)->evdepth;          \
  } while(0)

#define FIXUP_EVTIME(p_world, p_event)                                  \
  do {                                                                  \
    if ((p_event)->evtime < (p_world)->low_evtime)                      \
      (p_event)->evtime = 2 * (p_world)->low_evtime - (p_event)->evtime; \
    else if ((p_event)->evtime > (p_world)->high_evtime)                \
      (p_event)->evtime = 2 * (p_world)->high_evtime - (p_event)->evtime; \
  } while(0)

#define FIXUP_EVMAG(p_event)                                          \
  do {                                                                  \
    if ((p_event)->evmag < MIN_MAGNITUDE)                               \
      (p_event)->evmag = 2 * MIN_MAGNITUDE - (p_event)->evmag;          \
    else if ((p_event)->evmag > MAX_MAGNITUDE)                          \
      (p_event)->evmag = 2 * MAX_MAGNITUDE - (p_event)->evmag;          \
  } while(0)

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

  /* Cached proposed events */
  Event_t ** pp_prop_events;
  int num_prop_events;
  double max_prop_evtime;

  /* static entries */
  int runid;
  int numsamples;
  int birthsteps;
  int window;
  int step;
  int numthreads;
  PyObject * propose_eventobj;
  int verbose;
  PyObject * write_events_cb;

} World_t;

static void insert_event(NetModel_t * p_netmodel,
                         World_t * p_world, Event_t * p_event)
{
  int numsites;
  int numtimedefphases;
  int j;
  int evnum;
  
  numsites = EarthModel_NumSites(p_netmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_netmodel->p_earth);

  /* assign an origin-id to the event */
  p_event->orid = p_world->ev_orid_sequence ++; 

  /* initialize the number of detections to 0 for all phases */
  for (j = 0; j < numsites * numtimedefphases; j ++)
    p_event->p_num_dets[j] = 0;
  
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

static int propose_from_eventobj(NetModel_t * p_netmodel,
                                 Event_t ** p_p_events,
                                 int low_evtime, int high_evtime,
                                 PyArrayObject * propose_eventobj)
{
  int numevents;
  Py_ssize_t i;

  numevents = 0;
  
  for(i=0; i<propose_eventobj->dimensions[0]; i++)
  {
    double evtime;
    evtime = ARRAY2(propose_eventobj, i, EV_TIME_COL);
    
    if ((evtime >= low_evtime) && (evtime < high_evtime))
    {
      Event_t * p_event;

      p_event = alloc_event(p_netmodel);
      
      p_event->evlon = ARRAY2(propose_eventobj, i, EV_LON_COL);
      p_event->evlat = ARRAY2(propose_eventobj, i, EV_LAT_COL);
      p_event->evdepth = ARRAY2(propose_eventobj, i, EV_DEPTH_COL);
      p_event->evtime = evtime;
      p_event->evmag = ARRAY2(propose_eventobj, i, EV_MB_COL);
     
      p_p_events[numevents ++] = p_event;
    }
  }

  return numevents;
}


/* add events using the propose_invert proposer */
static void add_propose_invert_events(NetModel_t * p_netmodel,
                                      World_t * p_world)
{
  Event_t * pp_events[1000];     /* assume at most 1000 events init */
  int numevents;
  int i;
  int saved_num_prop_events;
  time_t t1;
  
  /* first time init */
  if (p_world->max_prop_evtime < p_world->low_evtime)
    p_world->max_prop_evtime = p_world->low_evtime;

  saved_num_prop_events = p_world->num_prop_events;
  
  t1 = time(NULL);

  if (p_world->propose_eventobj != Py_None)
    numevents = propose_from_eventobj(p_netmodel, pp_events,
                                      p_world->max_prop_evtime, 
                                      p_world->high_evtime,
                                  (PyArrayObject * )p_world->propose_eventobj);
  else
  {
    /* since the inverts doesn't create secondaries it doesn't make sense to
     * enable the secondary model for it. */
    p_netmodel->enable_sec_arr = 0;
    numevents = propose_invert_step(p_netmodel, pp_events,
                                    p_world->max_prop_evtime,
                                    p_world->high_evtime,
                                    p_world->low_detnum,
                                    p_world->high_detnum,
                                    2.5, p_world->birthsteps,
                                    p_world->numthreads);
    p_netmodel->enable_sec_arr = 1;
  }
  
  t1 = time(NULL) - t1;
  
  assert(numevents < 1000);

  /* cache all the newly proposed events */
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event = alloc_event(p_netmodel);
    copy_event(p_netmodel, p_event, pp_events[i]);
    p_world->pp_prop_events[p_world->num_prop_events ++] = p_event;
  }
  /* update the max time of the cached proposed events */
  p_world->max_prop_evtime = p_world->high_evtime;
  
  /* extend the proposed events with previously cached events, note
   * that the cached events are sorted across windows but not within a window */
  for (i=saved_num_prop_events-1;
       i>=0 && (p_world->pp_prop_events[i]->evtime 
                > (p_world->low_evtime - p_world->window));
       i--)
  {
    if (p_world->pp_prop_events[i]->evtime > p_world->low_evtime)
    {
      Event_t * p_event = alloc_event(p_netmodel);
      copy_event(p_netmodel, p_event, p_world->pp_prop_events[i]);
      pp_events[numevents ++] = p_event;
    }
  }

  if (p_world->verbose)
  {
    printf("initial window: %d events ela %ds\n", numevents, (int) t1);

    for (i=0; i<numevents; i++)
    {
      Event_t * p_event;
      
      p_event = pp_events[i];
      
      /* we are populating the orid here just for debugging the
       * proposed event and the score, insert_event below will populate the
       * real orid and clear out any detections */
      p_event->orid = p_world->ev_orid_sequence + i;

      printf("init+inv: ");
      print_event(p_event);
    }
  }

  for (i=0; i<numevents; i++)
    insert_event(p_netmodel, p_world, pp_events[i]);

}

/* returns the number of events with -ve scores that were removed from
 * the world */
static int remove_negative_events(World_t * p_world)
{
  int numdel;
  int j;
  
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

  return numdel;
}

static int remove_prim_negative_events(NetModel_t * p_netmodel,
                                       World_t * p_world)
{
  int numdel;
  int j;
  
  numdel = 0;
  
  for (j=p_world->low_evnum; j<p_world->high_evnum; j++)
  {
    Event_t * p_old_event;
    double prim_score;
    
    p_old_event = p_world->pp_events[j];
        
    p_netmodel->enable_sec_arr = 0;
    prim_score = score_event(p_netmodel, p_old_event);
    p_netmodel->enable_sec_arr = 1;
    
    if (prim_score < 0)
    {
      if (p_world->verbose)
      {
        printf("death (prim-score %.1f): ", prim_score);
        print_event(p_old_event);
      }
      delete_event(p_world, p_old_event);
      free_event(p_old_event);
      numdel ++;
    }
  }

  return numdel;
}


void resort_event(World_t * p_world, int curr_evnum)
{
  while(1)
  {
    /* check the event left of the current */
    if (((curr_evnum-1)>=0) && 
        (p_world->pp_events[curr_evnum-1]->evtime 
         > p_world->pp_events[curr_evnum]->evtime))
    {
      Event_t * temp;
      /* swap with the event to the left */
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


static void change_events(NetModel_t * p_netmodel, World_t * p_world,
                          int numchoices)
{
  int evnum;
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum ++)
  {
    Event_t * p_event;
    double old_score;

    p_event = p_world->pp_events[evnum];
    
    if (p_event->evtime < p_world->low_evtime)
      continue;

    do 
    {
      Event_t curr_event;
      int choice;
      
      old_score = p_event->evscore;

      curr_event = *p_event;        /* start from the current event */
    
#define UPDATE_BEST                                                     \
      do {                                                              \
        curr_event.evscore = score_event(p_netmodel, &curr_event);      \
        if (curr_event.evscore > p_event->evscore)                      \
          *p_event = curr_event;                                        \
      } while (0)

      /* first try to change all the dimensions simultaneously */
      for (choice = 0; choice < numchoices; choice ++)
      {
        curr_event.evlon = p_event->evlon + RAND_UNIFORM(-2, 2);
        FIXUP_EVLON(&curr_event);
        curr_event.evlat = p_event->evlat + RAND_UNIFORM(-2, 2);
        FIXUP_EVLAT(&curr_event);
        curr_event.evdepth = p_event->evdepth + RAND_UNIFORM(-100, 100);
        FIXUP_EVDEPTH(&curr_event);
        curr_event.evtime = p_event->evtime + RAND_UNIFORM(-5, 5);
        FIXUP_EVTIME(p_world, &curr_event);
        curr_event.evmag = p_event->evmag + RAND_UNIFORM(-2, 2);
        FIXUP_EVMAG(&curr_event);
      
        UPDATE_BEST;
      }
    
      /* change the depth and time simultaneously to try and account for
       * depth-time ambiguity */
      for (choice=0; choice < numchoices; choice ++)
      {
        curr_event.evdepth = RAND_UNIFORM(MIN_DEPTH, MAX_DEPTH);
        /* for each 10 km of depth the event time moves forward by 1 second */
        curr_event.evtime = p_event->evtime \
          + (curr_event.evdepth - p_event->evdepth)/10;
        curr_event.evtime += RAND_UNIFORM(-5, 5);
        FIXUP_EVTIME(p_world, &curr_event);
      
        UPDATE_BEST;
      }
    
      /* find the best longitude */
      for (choice = 0; choice < numchoices; choice ++)
      {
        curr_event.evlon = p_event->evlon + RAND_UNIFORM(-2, 2);
        FIXUP_EVLON(&curr_event);
        UPDATE_BEST;
      }

      /* find the best latitude */
      for (choice = 0; choice < numchoices; choice ++)
      {
        curr_event.evlat = p_event->evlat + RAND_UNIFORM(-2, 2);
        FIXUP_EVLAT(&curr_event);
        UPDATE_BEST;
      }

      /* find the best depth */
      for (choice = 0; choice < numchoices; choice ++)
      {
        curr_event.evdepth = p_event->evdepth + RAND_UNIFORM(-100, 100);
        FIXUP_EVDEPTH(&curr_event);
        UPDATE_BEST;
      }

      /* find the best time */
      for (choice = 0; choice < numchoices; choice ++)
      {
        curr_event.evtime = p_event->evtime + RAND_UNIFORM(-5, 5);
        FIXUP_EVTIME(p_world, &curr_event);
        UPDATE_BEST;
      }

      /* find the best magnitude */
      for (choice = 0; choice < numchoices; choice ++)
      {
        curr_event.evmag = p_event->evmag + RAND_UNIFORM(-2, 2);
        FIXUP_EVMAG(&curr_event);
        UPDATE_BEST;
      }

      /* did we change something? */
      if (p_event->evscore > old_score)
      {
#ifdef VERBOSE
        printf("change_events: orid %d score %.1f -> %.1f\n", p_event->orid,
               old_score, p_event->evscore);
#endif
      
        p_world->world_score += p_event->evscore - old_score;

        resort_event(p_world, evnum);
      }
    } while (p_event->evscore > (old_score + .001));
  }
}

/* disassociate the detection from any event-phase it might be associated to 
 * and adjust the world and event score accordingly 
 * returns the previous association of this detection if any otherwise -1 
 * note: in the case of a chain of detections all detections following this
 * are also disassociated */
static void disassoc_detection(NetModel_t * p_netmodel, World_t * p_world,
                               int detnum, int * prev_evnum,
                               int * prev_phaseid, int * prev_pos)
{
  EarthModel_t * p_earth = p_netmodel->p_earth;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  Detection_t * p_detection = p_netmodel->p_detections + detnum;
  int siteid = p_detection->site_det;
  double dettime = p_detection->time_det;
  Event_t * p_event;
  int evnum, phaseid, pos;
  
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum++)
  {
    p_event = p_world->pp_events[evnum];
    
    /* if the event couldn't possibly have caused the detection then
     * don't bother with it */
    if ((p_event->evtime > dettime) 
        || (p_event->evtime < (dettime-MAX_TRAVEL_TIME)))
      continue;

    for (phaseid=0; phaseid < numtimedefphases; phaseid ++)
    {
      int numdet = p_event->p_num_dets[siteid * numtimedefphases + phaseid];
      
      for (pos=0; pos<numdet; pos++)
      {
        if (p_event->p_all_detids[siteid * numtimedefphases * MAX_PHASE_DET
                                  + phaseid * MAX_PHASE_DET + pos] == detnum)
        {
          goto detnum_found;
        }
      }
    }
  }

  /* the detnum is not associated to anything ! */
  *prev_evnum = *prev_phaseid = *prev_pos = -1;
  return;

detnum_found:
  *prev_evnum = evnum;
  *prev_phaseid = phaseid;
  *prev_pos = pos;

  {
    double old_score, new_score;
    double distance, pred_az;
    
    distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                                siteid);
    
    pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                        p_event->evlat, siteid);
    
    /* the old score of this event before any change */
    score_event_site_phase(p_netmodel, p_event,
                           siteid, phaseid,
                           distance, pred_az, &old_score);

    /* remove the detection and everything after it as well ! */
    p_event->p_num_dets[siteid * numtimedefphases + phaseid] = pos;
    
    /* the new score of this event */
    score_event_site_phase(p_netmodel, p_event,
                           siteid, phaseid,
                           distance, pred_az, &new_score);

    p_event->evscore += new_score - old_score;
    p_world->world_score += new_score - old_score;
  }
}


/* greedily find the best event-phase for a detection 
* returns 0 if not change else 1 */
static int change_one_detection(NetModel_t * p_netmodel, World_t * p_world,
                                int detnum, int maxscore_mode)
{
  EarthModel_t * p_earth = p_netmodel->p_earth;
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  Detection_t * p_detection = p_netmodel->p_detections + detnum;
  int siteid = p_detection->site_det;
  int evnum;
  int best_evnum;
  int best_phaseid;
  int best_pos;
  double best_score;
  double best_score_delta;
  int prev_evnum, prev_phaseid, prev_pos;
  
  best_evnum = best_phaseid = best_pos = -1;
  best_score = best_score_delta = 0;

  /* first disassociate the detection and determine the previous assoc of
   * of this detection (or -1 if not associated) */
  disassoc_detection(p_netmodel, p_world, detnum, &prev_evnum, &prev_phaseid,
                     &prev_pos);

  /* then try to find the best possible assoc that one can make */
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum++)
  {
    Event_t * p_event;
    int phaseid;
    double distance, pred_az;
    
    p_event = p_world->pp_events[evnum];
    
    /* if the event couldn't possibly have caused the detection then
     * don't bother with it */
    if ((p_event->evtime > p_detection->time_det) 
        || (p_event->evtime < (p_detection->time_det - MAX_TRAVEL_TIME)))
      continue;
    
    distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,siteid);
    
    pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                        p_event->evlat, siteid);
    
    /* try all possible phases of this event */
    for (phaseid=0; phaseid < numtimedefphases; phaseid ++)
    {
      int old_detnum, old_numdet;
      double old_score;
      double score;
      double score_delta;
      int poss;

      /* first compute the old score for this event-site-phase */
      poss = score_event_site_phase(p_netmodel, p_event, siteid, phaseid,
                                    distance, pred_az, &old_score);
      
      /* if the phase is not possible at this site then skip it */
      if (!poss)
        continue;
      
      /* save the old detnum and numdet */
      old_numdet = p_event->p_num_dets[siteid * numtimedefphases + phaseid];
      old_detnum = p_event->p_all_detids[siteid * numtimedefphases 
                                         * MAX_PHASE_DET
                                         + phaseid * MAX_PHASE_DET + 0];
      
      /* try adding to the head of the detection list */
      p_event->p_num_dets[siteid * numtimedefphases + phaseid] = 1;
      p_event->p_all_detids[siteid * numtimedefphases * MAX_PHASE_DET
                            + phaseid * MAX_PHASE_DET + 0] = detnum;
      

      poss = score_event_site_phase(p_netmodel, p_event, siteid, phaseid,
                                    distance, pred_az, &score);

      assert(poss);                          /* it should remain possible! */
      
      score_delta = score - old_score;

      /* restore the phase */
      p_event->p_num_dets[siteid * numtimedefphases + phaseid] = old_numdet;
      p_event->p_all_detids[siteid * numtimedefphases * MAX_PHASE_DET
                            + phaseid * MAX_PHASE_DET + 0] = old_detnum;
      
      /* did we improve the score ? */
      /* note that the .001 in maxscore mode is because floating point
       * addition/subtraction is not associative and we could get into loops
       * otherwise */
      if ((score > 0) && (score_delta > 0) 
          && ((best_evnum == -1)
              || ((maxscore_mode && ((p_event->evscore + score_delta)
                                     > (best_score + .001)))
                  || 
                  (!maxscore_mode && (score_delta > best_score_delta)))))
      {
        best_evnum = evnum;
        best_phaseid = phaseid;
        best_pos = 0;
        
        best_score = p_event->evscore + score_delta;
        best_score_delta = score_delta;
#ifdef VERBOSE
    printf("curr change_detections %d: orid %d site %d phase %d pos 0"
           " score %f -> %f best_score %f best_score_delta %f\n",
           detnum, p_event->orid, p_detection->site_det, best_phaseid,
           p_event->evscore, p_event->evscore + best_score_delta,
           best_score, best_score_delta);
#endif
      }
    }
  }

  /* now assign this detection to the best spot we found */
  if (best_evnum != -1)
  {
    Event_t * p_event;

    p_event = p_world->pp_events[best_evnum];
    
#ifdef VERBOSE
    printf("change_detections %d: orid %d site %d phase %d pos %d "
           "score %f -> %f best_score %f best_score_delta %f\n",
           detnum, p_event->orid, p_detection->site_det, best_phaseid,
           best_pos,
           p_event->evscore, p_event->evscore + best_score_delta,
           best_score, best_score_delta);
#endif
    p_event->p_num_dets[siteid * numtimedefphases+ best_phaseid] = best_pos + 1;
    p_event->p_all_detids[siteid * numtimedefphases * MAX_PHASE_DET
                          + best_phaseid * MAX_PHASE_DET + best_pos] = detnum;
    
    p_event->evscore += best_score_delta;
    p_world->world_score += best_score_delta;
  }

  /* did we change this detection ? */
  /* note: we are only interested in redoing the earlier detections if we
   * changed the head of a detlist, the rest of the detlist is always redone
   * in any case */
  if ((best_pos > 0) || (best_evnum == -1)
      || ((best_evnum == prev_evnum) && (best_phaseid == prev_phaseid)
          && (best_pos == prev_pos)))
    return 0;
  else
    return 1;
}

static void change_detections(NetModel_t * p_netmodel, World_t * p_world,
                              int maxscore_mode)
{
  int detnum;
  int cnt;
  int iters;
  
  /* keep changing the detections till something converges */
  iters = 0;
  do 
  {
    cnt = 0;
    iters ++;
    
    for (detnum = p_world->low_detnum; detnum < p_world->high_detnum;
         detnum ++)
    {
      cnt += change_one_detection(p_netmodel, p_world, detnum,
                                  maxscore_mode);
    }
#ifdef VERBOSE
    printf("change_detections_loop: %d\n", cnt);
#endif
  } while ((cnt > 0) && (iters < 20));
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
  p_world->maxevents = p_netmodel->numdetections + 1000;
  p_world->pp_events = (Event_t **) calloc(p_world->maxevents,
                                           sizeof(*p_world->pp_events));

  p_world->pp_prop_events = (Event_t **) calloc(p_world->maxevents,
                                             sizeof(*p_world->pp_prop_events));
  
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

  for (i=0; i<p_world->num_prop_events; i++)
  {
    Event_t * p_event;
    
    p_event = p_world->pp_prop_events[i];

    free_event(p_event);
  }

  free(p_world->pp_prop_events);

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

#ifdef NEVER
    /* add an initial set of events in the new window */
     add_propose_hough_events(p_netmodel, p_world);
    
    add_invert_events(p_netmodel, p_world);
#endif

    add_propose_invert_events(p_netmodel, p_world);

    /* change the detections to use these new events */
    change_detections(p_netmodel, p_world, 0);

    /* keep track of whether or not we have wrapped around inverting
     * detections this will trigger further inverts to perturb around
     * the inverted location */
    p_world->inv_detnum = 0;
    p_world->inv_detnum_wrap = 0;

    t1 = time(NULL);

    for (i=0; i<p_world->numsamples; i++)
    {
      int numdel;
      double old_score;
#ifdef NEVER
      Event_t * p_new_event;

      old_score = p_world->world_score;
      
      /* birth move */
      p_new_event = add_event(p_netmodel, p_world);

      change_detections(p_netmodel, p_world, 0);

      if (p_world->world_score < old_score)
      {
        delete_event(p_world, p_new_event);
        free_event(p_new_event);
        change_detections(p_netmodel, p_world, 0);
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
      
#endif /* NEVER */

      old_score = p_world->world_score;
      
      /* remove some obvious events */
      numdel = remove_negative_events(p_world);
      
      if (numdel > 0)
      {
        change_detections(p_netmodel, p_world, 0);
      }
      
      if (p_world->world_score < (old_score - 1e-6))
      {
        printf("after death: world score has gone down by %.3f -> %.3f\n", 
               old_score - p_world->world_score, p_world->world_score);
      }
      
      change_events(p_netmodel, p_world, 10);

      change_detections(p_netmodel, p_world, 0);
    };

    /* only remove events if numsamples > 0. This allows a
     * numsamples=0 run to save all the proposed events */
    if (p_world->numsamples)
    {
      int numdel;
      
      if (remove_negative_events(p_world))
        change_detections(p_netmodel, p_world, 0);

      /* now for one round try to delete each event and see if that improves
       * the world score, but we only want to delete events with 
       * time > low_evtime, otherwise we might not be able to put 
       * their detections back */
      for (i=p_world->low_evnum; i<p_world->high_evnum; i++)
      {
        Event_t * p_old_event;
        double old_score;
      
        p_old_event = p_world->pp_events[i];

        if (p_old_event->evtime <= p_world->low_evtime)
          continue;
        
        old_score = p_world->world_score;
      
        delete_event(p_world, p_old_event);
      
        change_detections(p_netmodel, p_world, 0);
      
        if (p_world->world_score < old_score)
        {
          insert_event(p_netmodel, p_world, p_old_event);
          change_detections(p_netmodel, p_world, 0);
        }
        else
        {
          if (p_world->verbose)
          {
            printf("kill: ");
            print_event(p_old_event);
          }
          free_event(p_old_event);
        }
      }

#ifdef MAXSCORE_MODE
      /* now try and change the detections to go to the event with the
       * highest score rather than the best place for the
       * detection. This might get rid of some spurious events */
      change_detections(p_netmodel, p_world, 1);
      
      numdel = remove_negative_events(p_world);
      if ((numdel > 0) && p_world->verbose)
      {
        printf("maxscore mode killed %d events\n", numdel);
      }
      
      /* restore the detections in normal mode */
      change_detections(p_netmodel, p_world, 0);
      
      /* if we freed up some detections then we need to
       * relocate/reassociate again */
      if (numdel > 0)
      {
        change_events(p_netmodel, p_world, 10);
        change_detections(p_netmodel, p_world, 0);
      }
#endif

      /* finally, we will check if all these events are justified by
       * their primary detections alone. If not kill those events */
      numdel = remove_prim_negative_events(p_netmodel, p_world);
      if ((numdel > 0) && p_world->verbose)
      {
        printf("primary negative mode killed %d events\n", numdel);
      }
      
      /* if we freed up some detections then we need to
       * relocate/reassociate again */
      if (numdel > 0)
      {
        change_detections(p_netmodel, p_world, 0);
        change_events(p_netmodel, p_world, 10);
        change_detections(p_netmodel, p_world, 0);
      }

    }
    
/*
    change_events(p_netmodel, p_world, 2);
    
    change_detections(p_netmodel, p_world, 0);
*/
    t1 = time(NULL) - t1;
    
    if (p_world->verbose)
    {
      printf("evnum %d-%d evtime %.0f-%.0f detnum %d-%d ela %ds score=%.1f\n",
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
  int birthsteps;
  int window;
  int step;
  int numthreads;
  int verbose;
  PyObject * propose_eventobj;
  PyObject * write_events_cb;

  PyObject * retobj;
  PyObject * eventsobj;
  PyObject * evdetlistobj;
  
  if (!PyArg_ParseTuple(args, "iiiiiiOiO", &runid, &numsamples, 
                        &birthsteps,
                        &window, &step, &numthreads,
                        &propose_eventobj,
                        &verbose, &write_events_cb))
    return NULL;

  /* allocate the world and initialize the user arguments */
  p_world = alloc_world(p_netmodel);
  p_world->runid = runid;
  p_world->numsamples = numsamples;
  p_world->birthsteps = birthsteps;
  p_world->window = window;
  p_world->step = step;
  p_world->numthreads = numthreads;
  p_world->propose_eventobj = propose_eventobj;
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
