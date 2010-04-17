#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "../netvisa.h"

#define INFER_WINDOW_SIZE 1800
#define INFER_WINDOW_STEP  300

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
    
    if (p_world->inv_detnum < p_world->low_detnum)
      p_world->inv_detnum = p_world->low_detnum;
    else if (p_world->inv_detnum >= p_world->high_detnum)
    {
      p_world->inv_detnum = p_world->low_detnum;
      p_world->inv_detnum_wrap = 1;
    }
    
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

#define STEP_MIN  1
#define STEP_MAX  2
#define STEP_PLUS 1

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
  
  for (detnum = p_world->low_detnum; detnum < p_world->high_detnum; detnum ++)
  {
    change_one_detection(p_netmodel, p_world, detnum);
  }
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
    
    i = 0;
    do
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
      else
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
          printf("death: ");
          print_event(p_old_event);
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
      
      change_events(p_netmodel, p_world);

      change_detections(p_netmodel, p_world);

      if (p_world->inv_detnum_wrap)
      {
        p_world->inv_detnum_wrap = 0;
        i ++;
      }
      
    } while (i < numsamples);

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
