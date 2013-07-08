#ifndef SIGVISA_NETWORK_INCLUDE
#define SIGVISA_NETWORK_INCLUDE

#include <Python.h>
#include <math.h>
#include <float.h>
#include "uthash.h"

#undef NDEBUG
#include <assert.h>


typedef struct Detection_t
{
  int site_det;
  int arid_det;
  double time_det;
  double deltim_det;
  double azi_det;
  double delaz_det;
  double slo_det;
  double delslo_det;
  double snr_det;
  int phase_det;
  double amp_det;
  double per_det;

  int prev_det;               /* the previous detection or -1 */
  double logprob_det;   /* the log probability of either being noise or coda */

  int sigvisa_fake;

} Detection_t;

typedef struct Arrival_t {

  int phaseid;
  int siteid;

  double time;
  double azi;
  double slo;

  double amp;
  double peak_time;
  double peak_amp;
  double peak_decay;
  double coda_decay;

  double dist;
  double score;

} Arrival_t;

typedef struct Event_t
{
  double evlon;
  double evlat;
  double evdepth;
  double evtime;
  double evmag;


  int * p_num_dets; /* number of secondary detections, 0 to MAX_PHASE_DET */

  /* array of numsites * numtimedefphases * MAX_PHASE_DET */
  int * p_all_detids;       /* all detids, first one is the primary */

  /* array of numsites * numtimedefphases */
  Arrival_t * p_arrivals;

  int orid;
  double evscore;
} Event_t;

typedef struct Site_t
{
  char sta[7];
  double sitelon;
  double sitelat;
  double siteelev;
  int    siteisarr;
  double ontime;
  double offtime;
  int ref_siteid;
  struct Site_t * previous;
  UT_hash_handle hh;
} Site_t;

/* event array columns */
#define EV_LON_COL   0
#define EV_LAT_COL   1
#define EV_DEPTH_COL 2
#define EV_TIME_COL  3
#define EV_MB_COL    4
#define EV_ORID_COL  5
#define EV_EVID_COL  6
#define EV_NUM_COLS  7

/* site array columns */
#define SITE_LON_COL      0
#define SITE_LAT_COL      1
#define SITE_ELEV_COL     2
#define SITE_ISARR_COL    3        /* is the site an array station? */
#define SITE_ONTIME_COL   4
#define SITE_OFFTIME_COL  5
#define SITE_REF_SITEID_COL 6
#define SITE_NUM_COLS     7

/* detections array columns */
#define DET_SITE_COL    0
#define DET_ARID_COL    1
#define DET_TIME_COL    2
#define DET_DELTIM_COL  3
#define DET_AZI_COL     4
#define DET_DELAZ_COL   5
#define DET_SLO_COL     6
#define DET_DELSLO_COL  7
#define DET_SNR_COL     8
#define DET_PHASE_COL   9
#define DET_AMP_COL    10
#define DET_PER_COL    11
#define DET_NUM_COLS   12


#define PHASENAME_MAXLEN 6


#define MIN_MAGNITUDE   ((double) 2.0)
#define MAX_MAGNITUDE   ((double) 8.0)

#define MIN_DEPTH       ((double) 0.0)
#define MAX_DEPTH       ((double) 700.0)

#define MIN_SLOWNESS    ((double) 0.0)
#define MAX_SLOWNESS    ((double) 400.0)

#define LOGPROB_UNIFORM_SLOWNESS (- log(MAX_SLOWNESS - MIN_SLOWNESS))

#define MIN_AZIMUTH     ((double) -180)
#define MAX_AZIMUTH     ((double) +180)

#define LOGPROB_UNIFORM_AZIMUTH (- log(MAX_AZIMUTH - MIN_AZIMUTH))



/* maximum time taken by any phase */
#define MAX_TRAVEL_TIME ((double) 2000.0)

/* the spacing between secondary detections */
#define SECDET_INTERVAL ((double) 5.0)

/* maximum number of primary + secondary detections for any phase */
#define MAX_PHASE_DET  15

/* max time residual should not be needed in a proper model but due to
 * unexplained extremely high variance in some of the travel time residual
 * we need to put this in place for now */
#define MAX_TIME_RESIDUAL ((double) 6.0)

/* DELTA_TIME and DELTA_DIST are used in evaluating the answer */
#define DELTA_TIME 50                        /* in seconds */
#define DELTA_DIST 5                         /* in degrees */

#define MAX_EVENT_RATE 1

#define UPTIME_QUANT     3600                /* 1 hour */

#define MAX_PHASE(ntdp) ntdp

#define USE_PHASE(phase) (phase == 0)

#define IS_S_PHASE(phase) (phase == 3 || phase == 4 || phase == 10)
#define IS_P_PHASE(phase) (phase == 0 || phase == 1 || phase == 2 || phase == 5 || phase == 6 || phase == 7 || phase == 9 || phase == 11 || phase == 12)

#include "sigmodel.h"


//#define ALLOC_EVENT(net, sig) (net != NULL) ? alloc_event_net(net) : alloc_event_sig(sig);
//#define COPY_EVENT(net,sig, a, b) (net != NULL) ? copy_event_net(net, a, b) : copy_event_sig(sig, a, b);

#endif // SIGVISA_INCLUDE
