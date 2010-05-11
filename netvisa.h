#include <Python.h>
/* we don't want _import_array defined in all files except the main
 * module of netvisa */
#if !defined NETVISA_MAIN_MODULE
  #define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

typedef struct Event_t
{
  double evlon;
  double evlat;
  double evdepth;
  double evtime;
  double evmag;

  /* array of numsites x numtimedefphases */
  int * p_detids;                            /* detection numbers or -1 */

  int orid;
  double evscore;
} Event_t;

typedef struct Site_t
{
  double sitelon;
  double sitelat;
  double siteelev;
  int    siteisarr;
} Site_t;

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
} Detection_t;


#include "priors/NumEventPrior.h"
#include "priors/EventLocationPrior.h"
#include "priors/EventMagPrior.h"
#include "priors/EventDetectionPrior.h"
#include "priors/ArrivalTimePrior.h"
#include "priors/NumFalseDetPrior.h"
#include "priors/ArrivalAzimuthPrior.h"
#include "priors/ArrivalSlownessPrior.h"
#include "priors/ArrivalPhasePrior.h"
#include "priors/ArrivalSNRPrior.h"
#include "priors/EarthModel.h"
#include "priors/Laplace.h"
#include "priors/Poisson.h"
#include "priors/Gaussian.h"
#include "priors/LogNormal.h"

typedef struct NetModel_t
{
  PyObject_HEAD

  double start_time;
  double end_time;
  
  int numdetections;
  struct Detection_t * p_detections;

  int numsites;
  int numtime;       /* number of quantized time slots in p_site_up */
  
  int * p_site_up;                           /* numsites x numtime */

  EarthModel_t * p_earth;
  
  NumEventPrior_t num_event_prior;
  EventLocationPrior_t event_location_prior;
  EventMagPrior_t event_mag_prior;
  EventDetectionPrior_t event_det_prior;
  ArrivalTimePrior_t arr_time_prior;
  NumFalseDetPrior_t num_falsedet_prior;
  ArrivalAzimuthPrior_t arr_az_prior;
  ArrivalSlownessPrior_t arr_slo_prior;
  ArrivalPhasePrior_t arr_phase_prior;
  ArrivalSNRPrior_t arr_snr_prior;
  
} NetModel_t;

#include "priors/score.h"
#include "infer/infer.h"
#include "infer/propose.h"

#define UPTIME_QUANT     3600                /* 1 hour */

#define NetModel_IsSiteUp(p_netmodel, siteid, arrtime)\
  ((arrtime >= (p_netmodel)->start_time) && (arrtime < (p_netmodel)->end_time)\
   && (p_netmodel)->p_site_up[(siteid) * (p_netmodel)->numtime\
   + ((int) floor(((arrtime) - (p_netmodel)->start_time) / UPTIME_QUANT))])

#define PI                 ((double) 3.1415926535897931)

#define LAT2Z(lat) (sin((lat) * PI / 180.0))
#define Z2LAT(z) (asin(z) * 180.0 / PI)

/* event array columns */
#define EV_LON_COL   0
#define EV_LAT_COL   1
#define EV_DEPTH_COL 2
#define EV_TIME_COL  3
#define EV_MB_COL    4
#define EV_ORID_COL  5
#define EV_NUM_COLS  6

/* site array columns */
#define SITE_LON_COL      0
#define SITE_LAT_COL      1 
#define SITE_ELEV_COL     2
#define SITE_ISARR_COL    3        /* is the site an array station? */
#define SITE_NUM_COLS     4

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

#define Event2R3Vector(event, vector) do {\
(vector)[0] = (event)->evlon; (vector)[1] = (event)->evlat;\
(vector)[2] = (event)->evdepth;} while(0)

#define ARRAY2(arr,i,j) (*((double *)PyArray_GETPTR2(arr,i,j)))
#define ARRAY1(arr,i) (*((double *)PyArray_GETPTR1(arr,i)))

#define BOOLARRAY2(arr,i,j) (*((npy_bool *)PyArray_GETPTR2(arr,i,j)))
#define BOOLARRAY1(arr,i) (*((npy_bool *)PyArray_GETPTR1(arr,i)))

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

#define PHASENAME_MAXLEN 6

/* maximum time taken by any phase */
#define MAX_TRAVEL_TIME ((double) 2000.0)

/* max time residual should not be needed in a proper model but due to
 * unexplained extremely high variance in some of the travel time residual
 * we need to put this in place for now */
#define MAX_TIME_RESIDUAL ((double) 6.0)

#define MIN(a,b) ((a) <= (b) ? (a) : (b))
#define MAX(a,b) ((a) >= (b) ? (a) : (b))

/* RAND_DOUBLE -> random number between 0 and 1 (exclusive) */
#define RAND_DOUBLE ( ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0) )
/* RAND_UNIFORM(a,b) -> random value between a and b */
#define RAND_UNIFORM(a,b) (((double) (a)) + ((double) ((b)-(a))) * RAND_DOUBLE)

#define PI                     ((double) 3.1415926535897931)
#define DEG2RAD                ((double) (PI / 180))
#define RAD2DEG                ((double) (180 / PI))
#define AVG_EARTH_RADIUS_KM    ((double) 6371) /* when modeled as a sphere */

/* DELTA_TIME and DELTA_DIST are used in evaluating the answer */
#define DELTA_TIME 50                        /* in seconds */
#define DELTA_DIST 5                         /* in degrees */


Event_t * alloc_event(NetModel_t * p_netmodel);
void free_event(Event_t * p_event);
void copy_event(NetModel_t * p_netmodel, Event_t * p_tgt_event,
                const Event_t * p_src_event);
void print_event(const Event_t * p_event);

void convert_events_to_pyobj(const EarthModel_t * p_earth,
                             const Event_t ** pp_events, int numevents,
                             PyObject ** pp_eventsobj,
                             PyObject ** pp_evdetlistobj);
