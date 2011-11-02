#include "../sigvisa.h"


typedef struct World_t
{

  /* are we doing inference from signal data (SIGVISA)? */
  int sigvisa;

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
  
  /* NETVISA ONLY: this is the earliest detection which could have
   * affected a new event  */
  int low_detnum; 

  /* NETVISA ONLY: detections above this will not be looked at since
   * they couldn't have been caused by any event currently being
   * hypothesized high_detnum are all detections below high_evtime +
   * MAX_TRAVEL_TIME */
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

  int inv_detnum;           /* NETVISA ONLY: detection number to be
			       inverted next */
  int inv_detnum_wrap;          /* NETVISA ONLY: wrap around inverting
				   detections */
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
  PyObject * traces;
  int numsamples;
  int birthsteps;
  int window;
  int step;
  int numthreads;
  PyObject * propose_eventobj;
  int verbose;
  PyObject * write_events_cb;
  PyObject * log_segment_cb;
} World_t;

PyObject * py_infer(NetModel_t * p_netmodel, PyObject * args);
PyObject * py_infer_sig(SigModel_t * p_sigmodel, PyObject * args);

void log_segments_events(SigModel_t * p_sigmodel, PyObject * log_segment_cb, int runid, int numevents, const Event_t ** pp_events, double max_start_time);
