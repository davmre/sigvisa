#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "../sigvisa.h"

#define DEBUG
/*#define DEBUG2*/

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


static void insert_event(SigModel_t * p_sigmodel,
                         World_t * p_world, Event_t * p_event)
{
  int numsites;
  int numtimedefphases;
  int j;
  int evnum;
  
  EarthModel_t * p_earth= p_sigmodel->p_earth;

  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  /* assign an origin-id to the event */
  p_event->orid = p_world->ev_orid_sequence ++; 

  
  // TODO: figure out how world scoring works with SIGVISA
    /* initialize the number of detections to 0 for all phases */
  /*  if(p_netmodel != NULL) {
    for (j = 0; j < numsites * numtimedefphases; j ++)
      p_event->p_num_dets[j] = 0;
    p_event->evscore = score_event(p_netmodel, p_event);
    p_world->world_score += p_event->evscore;
  }*/
  
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

static int propose_from_eventobj(SigModel_t * p_sigmodel, 
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

      p_event = ALLOC_EVENT(p_sigmodel);
      
      p_event->evlon = ARRAY2(propose_eventobj, i, EV_LON_COL);
      p_event->evlat = ARRAY2(propose_eventobj, i, EV_LAT_COL);
      p_event->evdepth = ARRAY2(propose_eventobj, i, EV_DEPTH_COL);
      p_event->evtime = evtime;
      p_event->evmag = ARRAY2(propose_eventobj, i, EV_MB_COL);
     
      if (p_sigmodel) {
	initialize_mean_arrivals(p_sigmodel, p_event);
      }

      p_p_events[numevents ++] = p_event;
    }
  }

  return numevents;
}


/* add events using the propose_invert proposer */
static void add_propose_invert_events(SigModel_t * p_sigmodel,
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

  if (p_world->propose_eventobj != Py_None) {
    numevents = propose_from_eventobj(p_sigmodel, pp_events,
                                      p_world->max_prop_evtime, 
                                      p_world->high_evtime,
                                  (PyArrayObject * )p_world->propose_eventobj);
    LogDebug("propose_from_eventobj proposed %d events", numevents);
  }
  else
  {
    numevents = propose_invert_step(p_sigmodel, pp_events,
                                    p_world->max_prop_evtime,
                                    p_world->high_evtime,
                                    p_world->low_detnum,
                                    p_world->high_detnum,
                                    2.5, p_world->birthsteps,
                                    p_world->numthreads,
				    p_world->runid,
				    p_world->log_segment_cb);
    LogDebug("propose_invert_step proposed %d events", numevents);
  }

  
  t1 = time(NULL) - t1;
  
  assert(numevents < 1000);

  /* cache all the newly proposed events */
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event = ALLOC_EVENT(p_sigmodel);
    COPY_EVENT(p_sigmodel, p_event, pp_events[i]);
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
      Event_t * p_event = ALLOC_EVENT(p_sigmodel);
      COPY_EVENT(p_sigmodel, p_event, p_world->pp_prop_events[i]);
      pp_events[numevents ++] = p_event;
    }
  }

  if (p_world->verbose)
  {
    LogInfo("initial window: %d events ela %ds", numevents, (int) t1);

    for (i=0; i<numevents; i++)
    {
      Event_t * p_event;
      
      p_event = pp_events[i];
      
      /* we are populating the orid here just for debugging the
       * proposed event and the score, insert_event below will populate the
       * real orid and clear out any detections */
      p_event->orid = p_world->ev_orid_sequence + i;

      char * es = event_str(p_event);
      LogInfo("init+inv: %s", es);
      free(es);
    }
  }

  for (i=0; i<numevents; i++)
    insert_event(p_sigmodel, p_world, pp_events[i]);

}



/* propose a dummy event, for debugging purposes */
static void add_dummy_event(SigModel_t * p_sigmodel, World_t * p_world)
{
  Event_t * pp_events[1000];     /* assume at most 1000 events init */
  int numevents, i;
 
  /* first time init */
  if (p_world->max_prop_evtime < p_world->low_evtime)
    p_world->max_prop_evtime = p_world->low_evtime;

  numevents = 1;
  pp_events[0] = ALLOC_EVENT(p_sigmodel)

  Event_t * p_event = pp_events[0];
  
  // 5288718, args -k 12.8 -r 0.5 --siteids=2,91,109 --det-propose
  p_event->evlon = -178.69;
  p_event->evlat = -33.5;
  p_event->evdepth = 33;
  p_event->evtime = 1237726470.9;
  p_event->evmag = 4.39;
  

  //5295646, args -k 1.75 -r 0.5 --siteids=2,23,86,91,109 --det-propose 
  /*p_event->evlon = -174.66811;
  p_event->evlat = -22.93;
  p_event->evdepth = 0;
  p_event->evtime = 1237686882.4;
  p_event->evmag = 3.94;*/

  // 5297348, args -k 14.75 -r 0.5 --siteids=45,47 --det-propose 
  /*p_event->evlon = 128.496;
  p_event->evlat = 26.097;
  p_event->evdepth = 0;
  p_event->evtime = 1237733144.38;
  p_event->evmag = 4.43;*/


  /* cache all the newly proposed events */
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event = ALLOC_EVENT(p_sigmodel);
    COPY_EVENT(p_sigmodel, p_event, pp_events[i]);
    p_world->pp_prop_events[p_world->num_prop_events ++] = p_event;
  }
  /* update the max time of the cached proposed events */
  p_world->max_prop_evtime = p_world->high_evtime;
  
  for (i=0; i<numevents; i++)
    insert_event(p_sigmodel, p_world, pp_events[i]);

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

int list_other_events(World_t * p_world, int event, const Event_t *** ppp_other_events) {

  int n = p_world->high_evnum - p_world->low_evnum - 1;
  *ppp_other_events = (const Event_t **) malloc(n * sizeof(const Event_t *));
  
  int k=0;
  for (int evnum=p_world->low_evnum; evnum < p_world->high_evnum; ++evnum) {
    if (evnum != event) {
      assert(k < n);
      (*ppp_other_events)[k++] = p_world->pp_events[evnum];
    }
  }
  return n;
}

static void change_events(SigModel_t * p_sigmodel, 
			  World_t * p_world,
                          int numchoices)
{

  int evnum;
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum ++)
  {
    Event_t * p_event;
    Event_t curr_event;
    Event_t best_event;
    double best_prior_score;
    double curr_prior_score;
    int choice;
    
    p_event = p_world->pp_events[evnum];
    
    const Event_t ** pp_other_events;
    int num_other_events = -1;
    if (p_sigmodel != NULL) {
       num_other_events = list_other_events(p_world, evnum, &pp_other_events);
    }

    if (p_event->evtime < p_world->low_evtime)
      continue;

    /* the existing event is initially the best event */
    best_event = *p_event;
    best_prior_score = score_event_prior(p_sigmodel, &best_event);
    LogTrace("init update_events with best_event score %lf", best_event.evscore);

#define UPDATE_BEST                                                 \
    do {                                                            \
      curr_prior_score = score_event_prior(p_sigmodel, &curr_event);	\
      if (curr_prior_score > best_prior_score){			\
	LogTrace("    updating best event to %s", event_str(&curr_event)); \
        best_event = curr_event;					\
      }	else { 								\
	LogTrace("not updating, score %lf vs %lf for %s", best_prior_score, curr_prior_score, event_str(&curr_event)); \
      }									\
    } while (0)								

    /* first try to change all the dimensions simultaneously */
    curr_event = best_event;
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
    
    
    /* find the best longitude */
    curr_event = best_event;
    
    for (choice = 0; choice < numchoices; choice ++)
    {
      curr_event.evlon = p_event->evlon + RAND_UNIFORM(-2, 2);
      FIXUP_EVLON(&curr_event);
      UPDATE_BEST;
    }

    /* find the best latitude */
    curr_event = best_event;

    for (choice = 0; choice < numchoices; choice ++)
    {
      curr_event.evlat = p_event->evlat + RAND_UNIFORM(-2, 2);
      FIXUP_EVLAT(&curr_event);
      UPDATE_BEST;
    }

    /* find the best depth */
    curr_event = best_event;

    for (choice = 0; choice < numchoices; choice ++)
    {
      curr_event.evdepth = p_event->evdepth + RAND_UNIFORM(-100, 100);
      FIXUP_EVDEPTH(&curr_event);
      UPDATE_BEST;
    }

    /* find the best time */
    curr_event = best_event;

    for (choice = 0; choice < numchoices; choice ++)
    {
      curr_event.evtime = p_event->evtime + RAND_UNIFORM(-5, 5);
      FIXUP_EVTIME(p_world, &curr_event);
      UPDATE_BEST;
    }

    /* find the best magnitude */
    curr_event = best_event;

    for (choice = 0; choice < numchoices; choice ++)
    {
      curr_event.evmag = p_event->evmag + RAND_UNIFORM(-2, 2);
      FIXUP_EVMAG(&curr_event);
      UPDATE_BEST;
    }

    score_event_sig(p_sigmodel, &best_event, num_other_events, pp_other_events);
    /* set the event to the best so far */
    if (best_event.evscore > p_event->evscore)
    {

      LogTrace("change_events: orid %d score %.1f -> %.1f", p_event->orid,
             p_event->evscore, best_event.evscore);

      
      p_world->world_score += best_event.evscore - p_event->evscore;
      *p_event = best_event;

      resort_event(p_world, evnum);
    }

    
    if (p_sigmodel != NULL) {
      free(pp_other_events);
    }

  }
}



static void change_arrivals(SigModel_t * p_sigmodel, World_t * p_world) {

  // to a first approximation: loop over events, for each event, optimize its arrivals
  // note that this is optimizing event scores instead of world scores

  for (int evnum=p_world->low_evnum; evnum < p_world->high_evnum; ++evnum) {
    Event_t * p_event = p_world->pp_events[evnum];
    const Event_t ** pp_other_events;
    int num_other_events = list_other_events(p_world, evnum, &pp_other_events);
    optimize_arrivals(p_sigmodel, p_event, num_other_events, pp_other_events);
    free(pp_other_events);
  }

}

static void write_events(SigModel_t * p_sigmodel, World_t * p_world)
{
  PyObject * retval;
  double maxtime;
  PyObject * eventsobj;
  PyObject * evdetarrlistobj;
  int i;
  int numevents;
  
  double end_time = p_sigmodel->end_time;
  double start_time = p_sigmodel->start_time;

  EarthModel_t * p_earth = p_sigmodel->p_earth;

  if (p_world->high_evtime < end_time)
    maxtime = MAX(p_world->low_evtime - MAX_TRAVEL_TIME,
                  start_time);
  else
    maxtime = p_world->high_evtime;
  
  /* count the number of events */
  numevents = 0;
  for (i=p_world->write_evnum;
       (i<p_world->high_evnum) && p_world->pp_events[i]->evtime < maxtime;
       i++)
    numevents ++;

  LogInfo("writing %d events", numevents);
  convert_events_arrs_to_pyobj(p_sigmodel, p_earth, 
			       (const Event_t **) (p_world->pp_events 
						   + p_world->write_evnum),
			       numevents, &eventsobj, &evdetarrlistobj);
  if (p_world->verbose)
  {
    for (i=0; i<numevents; i++)
    {
      Event_t * p_event;
      
      p_event = p_world->pp_events[p_world->write_evnum + i];
      
      char * es = event_str(p_event);
      LogInfo("Write: %s", es);
      free(es);
    }
  }
  
  retval = PyObject_CallFunction(p_world->write_events_cb, "OOOOid", 
				 p_sigmodel, 
				 p_earth, 
				 eventsobj, evdetarrlistobj,
				 p_world->runid, maxtime);

  p_world->write_evnum += numevents;
  
  Py_DECREF(eventsobj);
  Py_DECREF(evdetarrlistobj);
  
  if (!retval) {
    LogError("Warning: can't write objects");
    CHECK_ERROR
  }
  else
    Py_DECREF(retval);
}



static World_t * alloc_world_sigvisa(SigModel_t * p_sigmodel)
{
  World_t * p_world;
  
  /* assume that we won't have more than MAX_EVENT_RATE events per second */
  p_world = (World_t *) calloc(1, sizeof(*p_world));

  p_world->maxevents = (p_sigmodel->end_time - p_sigmodel->start_time)*MAX_EVENT_RATE + 1000;
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



static void log_events(World_t * p_world) {
  LogDebug("events:");
  for(int i=p_world->low_evnum; i < p_world->high_evnum; ++i) {
    LogDebug(event_str(p_world->pp_events[i]));
  }
}

static void infer_sig(SigModel_t * p_sigmodel, World_t * p_world)
{

  int numsites = EarthModel_NumSites(p_sigmodel->p_earth);
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);


  int i;
  time_t t1;

  /* initialize the window */
  p_world->low_evnum = 0;
  p_world->high_evnum = 0;
  p_world->low_evtime = p_sigmodel->start_time;
  p_world->high_evtime = 0;

  do 
  {
    /* initialize high_evtime */
    p_world->high_evtime = MIN(p_world->low_evtime + p_world->window,
                               p_sigmodel->end_time);

    /* initialize low_evnum */
    for ( ; p_world->low_evnum < p_world->high_evnum; p_world->low_evnum ++)
    {
      Event_t * p_event;
      
      p_event = p_world->pp_events[p_world->low_evnum];

      if (p_event->evtime >= (p_world->low_evtime - MAX_TRAVEL_TIME))
        break;
    }

    /* initialize low_detnum */
    for ( ; p_world->low_detnum < p_sigmodel->numdetections;
          p_world->low_detnum ++)
    {
      Detection_t * p_det;
      
      p_det = p_sigmodel->p_detections + p_world->low_detnum;
      
      if (p_det->time_det >= p_world->low_evtime)
        break;
    }

    /* initialize high_detnum */
    for( ; p_world->high_detnum < p_sigmodel->numdetections;
         p_world->high_detnum ++)
    {
      Detection_t * p_det;
      
      p_det = p_sigmodel->p_detections + p_world->high_detnum;
      
      if (p_det->time_det >= (p_world->high_evtime + MAX_TRAVEL_TIME))
        break;
    }

    //LogInfo("adding initial event proposals");
    
    add_propose_invert_events(p_sigmodel, p_world);
    //LogDebug("logging segment proposals");
    //log_segments(p_sigmodel, p_world);
    /*    for (int i=0; i < numsites; ++i) {
      if (i != 2 && i != 91 && i != 109) continue;
      for (int j=0; j < numtimedefphases; ++j) {
	printf("arrival at station %d phase %d ", i, j);
	print_arrival(p_world->pp_events[0]->p_arrivals + (i-1)*numtimedefphases + j);
      }
    }
    log_segments(p_sigmodel, p_world);
    return;*/
//score_event_sig(p_sigmodel, p_world->pp_events[0], 0, NULL);
    
    for (int i=0; i < numsites; ++i) {
      for (int j=0; j < numtimedefphases; ++j) {
	Arrival_t * p_arr = p_world->pp_events[0]->p_arrivals + (i-1)*numtimedefphases + j;
	if (p_arr->amp == 0) continue;
	printf("arrival at station %d phase %d ", i, j);
	print_arrival(p_arr);
      }
    }

    /*for (int i=0; i < numsites; ++i) {
      for (int j=0; j < numtimedefphases; ++j) {
	Arrival_t * p_arr = p_world->pp_events[0]->p_arrivals + (i-1)*numtimedefphases + j;
	if (p_arr->amp == 0) continue;
	p_arr->time = 1237733169.4053;
	p_arr->amp = 506.4235;
	p_arr->azi = 164.7380;
	p_arr->slo = 33.0914;
	p_arr->phase = j;
	p_arr->siteid = i;
      }
    }

    double score1 = score_event_sta_sig(p_sigmodel, p_world->pp_events[0], 48, 0, NULL);
	 

    for (int i=0; i < numsites; ++i) {
      for (int j=0; j < numtimedefphases; ++j) {
	Arrival_t * p_arr = p_world->pp_events[0]->p_arrivals + (i-1)*numtimedefphases + j;
	if (p_arr->amp == 0) continue;
	printf("arrival at station %d phase %d ", i, j);
	print_arrival(p_arr);
	p_arr->time = 1237733171.4053;
	p_arr->amp = 337.6157;
	p_arr->azi = 180.0000;
	p_arr->slo = 33.0914;
	p_arr->phase = j;
	p_arr->siteid = i;
      }
    }

    double score2 = score_event_sta_sig(p_sigmodel, p_world->pp_events[0], 48, 0, NULL);

    LogInfo("score1 %lf score2 %lf", score1, score2);
    return;*/

    /* change the arrivals to use these new events */
    LogInfo("changing arrivals");
    change_arrivals(p_sigmodel, p_world);

    for (int i=0; i < numsites; ++i) {
      for (int j=0; j < numtimedefphases; ++j) {
	Arrival_t * p_arr = p_world->pp_events[0]->p_arrivals + (i-1)*numtimedefphases + j;
	if (p_arr->amp == 0) continue;
	printf("arrival at station %d phase %d ", i, j);
	print_arrival(p_arr);
      }
    }
    log_events(p_world);
    //LogDebug("logging arrivals");
    //log_segments(p_sigmodel, p_world);

    /* keep track of whether or not we have wrapped around inverting
     * detections this will trigger further inverts to perturb around
     * the inverted location */
    p_world->inv_detnum = 0;
    p_world->inv_detnum_wrap = 0;

    t1 = time(NULL);

    for (i=0; i<p_world->numsamples; i++)
    {
      int numdel=0;
      double old_score;

      old_score = p_world->world_score;
      
      /* remove some obvious events */
            numdel = remove_negative_events(p_world);
      
      if (numdel > 0)
      {
	LogInfo("deleted %d events, changing arrivals...", numdel);
        change_arrivals(p_sigmodel, p_world);
      }
      
      if (p_world->world_score < (old_score - 1e-6))
      {
        LogInfo("after death: world score has gone down by %.3f -> %.3f", 
               old_score - p_world->world_score, p_world->world_score);
      }

      LogInfo("changing events...");
      change_events(p_sigmodel, p_world, 10);

      LogInfo("changing arrivals...");
      change_arrivals(p_sigmodel, p_world);
      log_events(p_world);
    };
    
    /* only remove negative events if numsamples > 0. This allows a
     * numsamples=0 run to save all the proposed events */
    if (p_world->numsamples)
      remove_negative_events(p_world);
    
/*
    change_events(p_sigmodel, p_world, 2);
    
    change_detections(p_sigmodel, p_world);
*/
    t1 = time(NULL) - t1;
    
    //    if (p_world->verbose)
    //{
      LogInfo("evnum %d-%d evtime %.0f-%.0f detnum %d-%d ela %ds score=%.1f",
             p_world->low_evnum, p_world->high_evnum,
             p_world->low_evtime, p_world->high_evtime,
             p_world->low_detnum, p_world->high_detnum, (int) t1,
             p_world->world_score);
      //}
    

    /* move the window forward */
    p_world->low_evtime += p_world->step;

    /* write out any inferred events */

    //LogDebug("logging segments");
    //log_segments(p_sigmodel, p_world);
    LogInfo("writing events");
    write_events(p_sigmodel, p_world);
    
  } while (p_world->high_evtime < p_sigmodel->end_time);


  if (p_world->verbose)
  {
    for (i=0; i<p_world->high_evnum; i++)
    {
      Event_t * p_event;
    
      p_event = p_world->pp_events[i];

      LogDebug("orid %d - score %.1f", p_event->orid, 
             p_event->evscore);
    }
    LogDebug("World Score %.1f", p_world->world_score);
  }
}



PyObject * py_infer_sig(SigModel_t * p_sigmodel, PyObject * args)
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
  PyObject * log_segment_cb;

  PyObject * retobj;
  PyObject * eventsobj;
  PyObject * evarrlistobj;
  
  if (!PyArg_ParseTuple(args, "iiiiiiOiOO", &runid, &numsamples, 
                        &birthsteps,
                        &window, &step, &numthreads,
                        &propose_eventobj,
                        &verbose, &write_events_cb, &log_segment_cb))
    return NULL;

  /* allocate the world and initialize the user arguments */
  p_world = alloc_world_sigvisa(p_sigmodel);
  p_world->runid = runid;
  p_world->numsamples = numsamples;
  p_world->birthsteps = birthsteps;
  p_world->window = window;
  p_world->step = step;
  p_world->numthreads = numthreads;
  p_world->propose_eventobj = propose_eventobj;
  p_world->verbose = verbose;
  p_world->write_events_cb = write_events_cb;
  p_world->log_segment_cb = log_segment_cb;
  
  // TODO: write inference
  LogDebug("created world, calling inference...");
  infer_sig(p_sigmodel, p_world);

  /* convert the world to python structures */
  convert_events_arrs_to_pyobj(p_sigmodel, p_sigmodel->p_earth,
			       (const Event_t **)p_world->pp_events,
			       p_world->high_evnum, &eventsobj, &evarrlistobj);
  
  free_world(p_world);

  retobj = Py_BuildValue("(OO)", eventsobj, evarrlistobj);

  /* BuildValue increments the ref count so we need to decrement our ref */
  Py_DECREF(eventsobj);
  Py_DECREF(evarrlistobj);
  
  return retobj;  
}
