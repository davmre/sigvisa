#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "../sigvisa.h"
#include "../netvisa.h"


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


static void insert_event(NetModel_t * p_netmodel,
			 SigModel_t * p_sigmodel,
                         World_t * p_world, Event_t * p_event)
{
  int numsites;
  int numtimedefphases;
  int j;
  int evnum;
  
  assert(p_netmodel == NULL || p_sigmodel == NULL);
  
  EarthModel_t * p_earth;
  if (p_netmodel != NULL) 
    p_earth  = p_netmodel->p_earth;
  else
    p_earth  = p_sigmodel->p_earth;

  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  /* assign an origin-id to the event */
  p_event->orid = p_world->ev_orid_sequence ++; 

  
  // TODO: figure out how world scoring works with SIGVISA
  if(p_netmodel != NULL) {
    /* initialize the number of detections to 0 for all phases */
    for (j = 0; j < numsites * numtimedefphases; j ++)
      p_event->p_num_dets[j] = 0;
    p_event->evscore = score_event(p_netmodel, p_event);
    p_world->world_score += p_event->evscore;
  }
  
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
				 SigModel_t * p_sigmodel, 
                                 Event_t ** p_p_events,
                                 int low_evtime, int high_evtime,
                                 PyArrayObject * propose_eventobj)
{
  int numevents;
  Py_ssize_t i;

  assert(p_netmodel == NULL || p_sigmodel == NULL);

  numevents = 0;
  
  for(i=0; i<propose_eventobj->dimensions[0]; i++)
  {
    double evtime;
    evtime = ARRAY2(propose_eventobj, i, EV_TIME_COL);
    
    if ((evtime >= low_evtime) && (evtime < high_evtime))
    {
      Event_t * p_event;

      p_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
      
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
static void add_propose_invert_events(NetModel_t * p_netmodel, SigModel_t * p_sigmodel,
                                      World_t * p_world)
{
  Event_t * pp_events[1000];     /* assume at most 1000 events init */
  int numevents;
  int i;
  int saved_num_prop_events;
  time_t t1;
 
  assert(p_netmodel == NULL || p_sigmodel == NULL);
 
  /* first time init */
  if (p_world->max_prop_evtime < p_world->low_evtime)
    p_world->max_prop_evtime = p_world->low_evtime;

  saved_num_prop_events = p_world->num_prop_events;
  
  t1 = time(NULL);

  if (p_world->propose_eventobj != Py_None) {
    numevents = propose_from_eventobj(p_netmodel, p_sigmodel, pp_events,
                                      p_world->max_prop_evtime, 
                                      p_world->high_evtime,
                                  (PyArrayObject * )p_world->propose_eventobj);
    printf("propose_from_eventobj proposed %d events\n", numevents);
  }
  else
  {
    numevents = propose_invert_step(p_netmodel, p_sigmodel, pp_events,
                                    p_world->max_prop_evtime,
                                    p_world->high_evtime,
                                    p_world->low_detnum,
                                    p_world->high_detnum,
                                    2.5, p_world->birthsteps,
                                    p_world->numthreads,
				    p_world->runid,
				    p_world->log_segment_cb);
    printf("propose_invert_step proposed %d events\n", numevents);
  }

  
  t1 = time(NULL) - t1;
  
  assert(numevents < 1000);

  /* cache all the newly proposed events */
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
    COPY_EVENT(p_netmodel, p_sigmodel, p_event, pp_events[i]);
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
      Event_t * p_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
      COPY_EVENT(p_netmodel, p_sigmodel, p_event, p_world->pp_prop_events[i]);
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
    insert_event(p_netmodel, p_sigmodel, p_world, pp_events[i]);

}



/* propose a dummy event, for debugging purposes */
static void add_dummy_event(NetModel_t * p_netmodel, SigModel_t * p_sigmodel, World_t * p_world)
{
  Event_t * pp_events[1000];     /* assume at most 1000 events init */
  int numevents, i;
 
  assert(p_netmodel == NULL || p_sigmodel == NULL);
 
  /* first time init */
  if (p_world->max_prop_evtime < p_world->low_evtime)
    p_world->max_prop_evtime = p_world->low_evtime;

  numevents = 1;
  pp_events[0] = ALLOC_EVENT(p_netmodel, p_sigmodel)

  Event_t * p_event = pp_events[0];
  p_event->evlon = -178.69;
  p_event->evlat = -33.5;
  p_event->evdepth = 33;
  p_event->evtime = 1237726470.9;
  p_event->evmag = 4.39;
  
  /* cache all the newly proposed events */
  for (i=0; i<numevents; i++)
  {
    Event_t * p_event = ALLOC_EVENT(p_netmodel, p_sigmodel);
    COPY_EVENT(p_netmodel, p_sigmodel, p_event, pp_events[i]);
    p_world->pp_prop_events[p_world->num_prop_events ++] = p_event;
  }
  /* update the max time of the cached proposed events */
  p_world->max_prop_evtime = p_world->high_evtime;
  
  for (i=0; i<numevents; i++)
    insert_event(p_netmodel, p_sigmodel, p_world, pp_events[i]);

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

static void change_events(NetModel_t * p_netmodel, SigModel_t * p_sigmodel, 
			  World_t * p_world,
                          int numchoices)
{

  assert(p_netmodel == NULL || p_sigmodel == NULL);

  int evnum;
  for (evnum = p_world->low_evnum; evnum < p_world->high_evnum; evnum ++)
  {
    Event_t * p_event;
    Event_t curr_event;
    Event_t best_event;
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
    //printf("init update_events with best_event score %lf\n", best_event.evscore);

#define UPDATE_BEST                                                 \
    do {                                                            \
      if (p_netmodel != NULL )                                      \
	curr_event.evscore = score_event(p_netmodel, &curr_event);  \
      else                                                          \
	score_event_prior(p_sigmodel, &curr_event); \
      if (curr_event.evscore > best_event.evscore){			\
        best_event = curr_event;					\
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

#ifdef DEBUG2
      printf("change_events: orid %d score %.1f -> %.1f\n", p_event->orid,
             p_event->evscore, best_event.evscore);
#endif
      
      p_world->world_score += best_event.evscore - p_event->evscore;
      *p_event = best_event;

      resort_event(p_world, evnum);
    }

    
    if (p_sigmodel != NULL) {
      free(pp_other_events);
    }

  }
}


/* greedily find the best event-phase for a detection */
static void change_one_detection(NetModel_t * p_netmodel, World_t * p_world,
  int detnum) {
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
                                        p_detection->site_det-1);
      
    for (phaseid=0; phaseid < MAX_PHASE(numtimedefphases); phaseid ++)
    {
      int old_detnum;
      double score;
      double not_det_prob;
      double score_delta;
      int poss;

      if (p_event->p_num_dets[(p_detection->site_det-1) * numtimedefphases
                              + phaseid] > 0)
        old_detnum = p_event->p_all_detids[(p_detection->site_det-1) 
                                           * numtimedefphases * MAX_PHASE_DET
                                           + phaseid * MAX_PHASE_DET + 0];
      else
        old_detnum = -1;

      /* change the detection of this phase and measure the score */
      p_event->p_num_dets[(p_detection->site_det-1) * numtimedefphases
                          + phaseid] = 1;
      p_event->p_all_detids[(p_detection->site_det -1)
                            * numtimedefphases * MAX_PHASE_DET
                            + phaseid * MAX_PHASE_DET + 0] = detnum;


      poss = score_event_site_phase(p_netmodel, p_event,
                                    p_detection->site_det, phaseid,
                                    distance, pred_az, &score);

      /* now, compute the probability that this phase was not detected
       * at this site */
      p_event->p_num_dets[p_detection->site_det * numtimedefphases
                          + phaseid] = 0;
      
      poss = score_event_site_phase(p_netmodel, p_event,
                                    p_detection->site_det, phaseid,
                                    distance, pred_az, &not_det_prob);
      
      score -= not_det_prob;

      /* if we have come across the current event/phase of the detection
       * then we need to leave the event phase as noise for now */
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
        
        p_event->p_num_dets[(p_detection->site_det-1) * numtimedefphases
                            + phaseid] = 1;
        
        p_event->p_all_detids[(p_detection->site_det -1)
                              * numtimedefphases * MAX_PHASE_DET
                              + phaseid * MAX_PHASE_DET + 0] = old_detnum;
        
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
    
    if (p_event->p_num_dets[(p_detection->site_det-1) * numtimedefphases
                            + best_phaseid] > 0)
      replaced_detnum = p_event->p_all_detids[(p_detection->site_det -1)
                                              * numtimedefphases * MAX_PHASE_DET
                                              + best_phaseid 
                                              * MAX_PHASE_DET + 0];
    else
      replaced_detnum = -1;

    p_event->p_num_dets[(p_detection->site_det-1) * numtimedefphases
                        + best_phaseid] = 1;
    p_event->p_all_detids[(p_detection->site_det -1)
                          * numtimedefphases * MAX_PHASE_DET
                          + best_phaseid 
                          * MAX_PHASE_DET + 0] = detnum;
    
    p_event->evscore += best_score_delta;
    p_world->world_score += best_score_delta;
  }

  if (replaced_detnum != -1)
    change_one_detection(p_netmodel, p_world, replaced_detnum);
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
    pp_saved_events[i] = alloc_event_net(p_netmodel);
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

static void write_events(NetModel_t * p_netmodel, SigModel_t * p_sigmodel, World_t * p_world)
{
  PyObject * retval;
  double maxtime;
  PyObject * eventsobj;
  PyObject * evdetarrlistobj;
  int i;
  int numevents;
  
  assert(p_netmodel == NULL || p_sigmodel == NULL);

  double end_time = (p_netmodel != NULL) ? 
    p_netmodel->end_time : p_sigmodel->end_time;
  double start_time = (p_netmodel != NULL) ? 
    p_netmodel->start_time : p_sigmodel->start_time;

  EarthModel_t * p_earth = (p_netmodel != NULL) ? p_netmodel->p_earth : p_sigmodel->p_earth;

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

  printf("writing %d events\n", numevents);
  if (p_netmodel != NULL) {
    convert_events_dets_to_pyobj(p_earth, 
			    (const Event_t **) (p_world->pp_events 
						+ p_world->write_evnum),
			    numevents, &eventsobj, &evdetarrlistobj);
  } else {
    convert_events_arrs_to_pyobj(p_sigmodel, p_earth, 
				 (const Event_t **) (p_world->pp_events 
						     + p_world->write_evnum),
				 numevents, &eventsobj, &evdetarrlistobj);
  }
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
  
  if (p_netmodel != NULL) {
    retval = PyObject_CallFunction(p_world->write_events_cb, "OOOOid", 
				   p_netmodel, 
				   p_earth, 
				   eventsobj, evdetarrlistobj,
				   p_world->runid, maxtime);
  } else {
    retval = PyObject_CallFunction(p_world->write_events_cb, "OOOOid", 
				   p_sigmodel, 
				   p_earth, 
				   eventsobj, evdetarrlistobj,
				   p_world->runid, maxtime);
  }

  p_world->write_evnum += numevents;
  
  Py_DECREF(eventsobj);
  Py_DECREF(evdetarrlistobj);
  
  if (!retval) {
    printf("Warning: can't write objects\n");
    CHECK_ERROR
  }
  else
    Py_DECREF(retval);
}


void log_segments_events(SigModel_t * p_sigmodel, PyObject * log_segment_cb, int numevents, const Event_t ** pp_events, double max_start_time, PyObject *py_text)
{
  PyObject * retval;
  PyObject * eventsobj;
  PyObject * evarrlistobj;
  int i;
  
  if(py_text == NULL) {
    py_text = Py_BuildValue("s", "");
  }

  convert_events_arrs_to_pyobj(p_sigmodel, p_sigmodel->p_earth, 
			       pp_events,numevents, 
			       &eventsobj, &evarrlistobj);
  
  for (i = 0; i < p_sigmodel->numsegments; ++i) {
    printf("logging segment %d\n", i);

    ChannelBundle_t * p_real_segment = p_sigmodel->p_segments + i;
    if (p_real_segment->start_time > max_start_time) {
      continue;
    }

    ChannelBundle_t * p_pred_segment = calloc(1, sizeof(ChannelBundle_t));
    memcpy(p_pred_segment, p_real_segment, sizeof(ChannelBundle_t));

    SignalPrior_ThreeAxisEnvelope(&p_sigmodel->sig_prior,
				  p_sigmodel->p_earth,
				  numevents,
				  pp_events,
				  p_pred_segment,
				  NULL);

    printf("real st %lf pred st %lf real sig st %lf %lf %lf\n", p_real_segment->start_time, p_pred_segment->start_time, p_real_segment->p_channels[0]->start_time, p_real_segment->p_channels[1]->start_time, p_real_segment->p_channels[2]->start_time);
    printf("real len %ld pred len %ld real sig len %ld %ld %ld\n", p_real_segment->len, p_pred_segment->len, p_real_segment->p_channels[0]->len, p_real_segment->p_channels[1]->len, p_real_segment->p_channels[2]->len);
    printf("real hz %lf pred hz %lf real sig hz %lf %lf %lf\n", p_real_segment->hz, p_pred_segment->hz, p_real_segment->p_channels[0]->hz, p_real_segment->p_channels[1]->hz, p_real_segment->p_channels[2]->hz);

    PyObject * real_trace, * pred_trace;
    real_trace = channel_bundle_to_trace_bundle(p_real_segment);
    pred_trace = channel_bundle_to_trace_bundle(p_pred_segment);
    printf("calling log_segment\n");
    retval = PyObject_CallFunction(log_segment_cb, "OOOOO", 
				   eventsobj, evarrlistobj,
				   real_trace,
				   pred_trace,
				   py_text);

    if (!retval) {
      printf("log_segment_cb call failed!\n");
      CHECK_ERROR;
    } else {
      Py_DECREF(retval);
    }

    Py_DECREF(real_trace);
    Py_DECREF(pred_trace);
  }
 
  Py_DECREF(eventsobj);
  Py_DECREF(evarrlistobj);
  
}

static void log_segments(SigModel_t * p_sigmodel, World_t * p_world)
{
  int numevents, i;
  double maxtime;
  double end_time = p_sigmodel->end_time;
  double start_time = p_sigmodel->start_time;

  if (p_world->high_evtime < end_time)
    maxtime = MAX(p_world->low_evtime - MAX_TRAVEL_TIME,
                  start_time);
  else
    maxtime = p_world->high_evtime;

  numevents = 0;
  for (i=p_world->write_evnum;
       (i<p_world->high_evnum) && p_world->pp_events[i]->evtime < maxtime;
       i++)
    numevents ++;

  log_segments_events(p_sigmodel, p_world->log_segment_cb, numevents, (const Event_t **) p_world->pp_events + p_world->write_evnum, maxtime + MAX_TRAVEL_TIME, NULL);

  p_world->write_evnum += numevents;

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

static void infer(NetModel_t * p_netmodel, World_t * p_world)
{
  int i;
  time_t t1;

  // initialize random seed, for deterministic testing
  srand ( 1 );

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


    add_propose_invert_events(p_netmodel, NULL, p_world);

    /* change the detections to use these new events */
    change_detections(p_netmodel, p_world);

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

      old_score = p_world->world_score;
      
      /* remove some obvious events */
      numdel = remove_negative_events(p_world);
      
      if (numdel > 0)
      {
        change_detections(p_netmodel, p_world);
      }
      
      if (p_world->world_score < (old_score - 1e-6))
      {
        printf("after death: world score has gone down by %.3f -> %.3f\n", 
               old_score - p_world->world_score, p_world->world_score);
      }
      
      change_events(p_netmodel, NULL, p_world, 10);

      change_detections(p_netmodel, p_world);
    };
    
    /* only remove negative events if numsamples > 0. This allows a
     * numsamples=0 run to save all the proposed events */
    if (p_world->numsamples)
      remove_negative_events(p_world);
    
/*
    change_events(p_netmodel, p_world, 2);
    
    change_detections(p_netmodel, p_world);
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
    write_events(p_netmodel, NULL, p_world);
    
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
  convert_events_dets_to_pyobj(p_netmodel->p_earth,
                          (const Event_t **)p_world->pp_events,
                          p_world->high_evnum, &eventsobj, &evdetlistobj);
  
  free_world(p_world);

  retobj = Py_BuildValue("(OO)", eventsobj, evdetlistobj);

  /* BuildValue increments the ref count so we need to decrement our ref */
  Py_DECREF(eventsobj);
  Py_DECREF(evdetlistobj);
  
  return retobj;  
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

    printf("adding initial event proposals\n");
    
    //add_propose_invert_events(NULL, p_sigmodel, p_world);
    add_dummy_event(NULL, p_sigmodel, p_world);
    initialize_mean_arrivals(p_sigmodel, p_world->pp_events[0]);
    Event_t * p_event = p_world->pp_events[0];
    for(int i=1; i <= numsites; ++i) {
      for(int j=0; j < numtimedefphases; ++j) {
	Arrival_t * p_arr = p_event->p_arrivals + (i-1)*numtimedefphases+j;
	if (p_arr && p_arr->time) {
	  printf(" arrival at station %d phase %d: ", i, j);
	  print_arrival(p_arr);
	}
      }
    }
    log_segments(p_sigmodel, p_world);
    return;
    //score_event_sig(p_sigmodel, p_world->pp_events[0], 0, NULL);

    printf("changing arrivals\n");
    /* change the arrivals to use these new events */
    change_arrivals(p_sigmodel, p_world);

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

      old_score = p_world->world_score;
      
      /* remove some obvious events */
      numdel = remove_negative_events(p_world);
      
      if (numdel > 0)
      {
	printf("deleted %d events, changing arrivals...\n", numdel);
        change_arrivals(p_sigmodel, p_world);
      }
      
      if (p_world->world_score < (old_score - 1e-6))
      {
        printf("after death: world score has gone down by %.3f -> %.3f\n", 
               old_score - p_world->world_score, p_world->world_score);
      }

      printf("events[0] is ");
      print_event(p_world->pp_events[0]);
      printf("changing events...\n");
      change_events(NULL, p_sigmodel, p_world, 10);

      printf("events[0] is ");
      print_event(p_world->pp_events[0]);

      printf("changing arrivals...\n");
      change_arrivals(p_sigmodel, p_world);

      printf("events[0] is ");
      print_event(p_world->pp_events[0]);
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
      printf("evnum %d-%d evtime %.0f-%.0f detnum %d-%d ela %ds score=%.1f\n",
             p_world->low_evnum, p_world->high_evnum,
             p_world->low_evtime, p_world->high_evtime,
             p_world->low_detnum, p_world->high_detnum, (int) t1,
             p_world->world_score);
      //}
    

    /* move the window forward */
    p_world->low_evtime += p_world->step;

    /* write out any inferred events */

    printf("logging segments\n");
    //log_segments(p_sigmodel, p_world);
    printf("writing events\n");
    write_events(NULL, p_sigmodel, p_world);
    
  } while (p_world->high_evtime < p_sigmodel->end_time);


  if (p_world->verbose)
  {
    for (i=0; i<p_world->high_evnum; i++)
    {
      Event_t * p_event;
    
      p_event = p_world->pp_events[i];

      printf("orid %d - score %.1f\n", p_event->orid, 
             p_event->evscore);
    }
    printf("World Score %.1f\n", p_world->world_score);
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
  printf("created world, calling inference...\n");
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
