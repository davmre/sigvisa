#ifndef SIGVISA_INCLUDE
#define SIGVISA_INCLUDE

#include <Python.h>
#include <math.h>
#include <float.h>
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
  
  int sigvisa_fake;

} Detection_t;

typedef struct Arrival_t {

  double time;
  double amp;
  double azi;
  double slo;

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
  double sitelon;
  double sitelat;
  double siteelev;
  int    siteisarr;
} Site_t;

/* event array columns */
#define EV_LON_COL   0
#define EV_LAT_COL   1
#define EV_DEPTH_COL 2
#define EV_TIME_COL  3
#define EV_MB_COL    4
#define EV_ORID_COL  5
#define EV_SCORE_COL  6
#define EV_NUM_COLS  7

/* site array columns */
#define SITE_LON_COL      0
#define SITE_LAT_COL      1 
#define SITE_ELEV_COL     2
#define SITE_ISARR_COL    3        /* is the site an array station? */
#define SITE_NUM_COLS     4

#include "netvisa.h"

/* define signal channels */
#define NUM_CHANS   4
#define CHAN_BHE    0
#define CHAN_BHN    1
#define CHAN_BHZ    2
#define CHAN_OTHER  3

#define CHECK_ERROR if(PyErr_Occurred()) { PyErr_Print(); exit(1); }
#define CHECK_PTR(p) if (p == NULL) { printf("null pointer detected!"); exit(1);}


typedef struct Signal_t
{

  long len;
  double * p_data;
  PyArrayObject * py_array;  /*  we're forced to keep the Python
				 object around so that we can DECREF
				 it when finished */
  double start_time;
  double hz;

  int siteid;
  int chan;
  int chanid;
  
} Signal_t;

typedef struct ChannelBundle_t {
  long len;

  double start_time;
  double hz;

  int siteid;

  Signal_t * p_channels[NUM_CHANS];
} ChannelBundle_t;

double ChannelBundle_EndTime(ChannelBundle_t * b);

#include "priors/ArrivalTimeJointPrior.h"
#include "priors/SignalPrior.h"

int print_signal(Signal_t * signal);
Signal_t * alloc_signal(ChannelBundle_t * p_segment);

typedef struct SigModel_t
{
  PyObject_HEAD

  double start_time;
  double end_time;

  int numsegments;
  ChannelBundle_t * p_segments;
  ChannelBundle_t * p_wave_segments;

  EarthModel_t * p_earth;

  long numdetections;
  Detection_t * p_detections;

  SignalPrior_t sig_prior;

  NumEventPrior_t num_event_prior;
  EventLocationPrior_t event_location_prior;
  EventMagPrior_t event_mag_prior;

  ArrivalTimeJointPrior_t arr_time_joint_prior;
  ArrivalAzimuthPrior_t arr_az_prior;
  ArrivalSlownessPrior_t arr_slo_prior;
  ArrivalAmplitudePrior_t arr_amp_prior;
  /*
  NumSecDetPrior_t num_secdet_prior;
  EventDetectionPrior_t event_det_prior;
  ArrivalTimePrior_t arr_time_prior;
  NumFalseDetPrior_t num_falsedet_prior;

  ArrivalPhasePrior_t arr_phase_prior;
  ArrivalSNRPrior_t arr_snr_prior;
*/
} SigModel_t;

int have_signal(SigModel_t * p_sigmodel, int site, double start_time, double end_time);


#include "priors/score_sig.h"

#include "priors/score.h"
#include "infer/infer.h"
#include "infer/propose.h"


#define Event2R3Vector(event, vector) do {\
(vector)[0] = (event)->evlon; (vector)[1] = (event)->evlat;\
(vector)[2] = (event)->evdepth;} while(0)


#define PI                     ((double) 3.1415926535897931)
#define DEG2RAD                ((double) (PI / 180))
#define RAD2DEG                ((double) (180 / PI))
#define AVG_EARTH_RADIUS_KM    ((double) 6371) /* when modeled as a sphere */

#define LAT2Z(lat) (sin((lat) * PI / 180.0))
#define Z2LAT(z) (asin(z) * 180.0 / PI)

#define ARRAY3(arr,i,j,k) (*((double *)PyArray_GETPTR3(arr,i,j,k)))
#define ARRAY2(arr,i,j) (*((double *)PyArray_GETPTR2(arr,i,j)))
#define ARRAY1(arr,i) (*((double *)PyArray_GETPTR1(arr,i)))

#define BOOLARRAY2(arr,i,j) (*((npy_bool *)PyArray_GETPTR2(arr,i,j)))
#define BOOLARRAY1(arr,i) (*((npy_bool *)PyArray_GETPTR1(arr,i)))

#define UPDATE_AND_VERIFY(a, b) if (*a == -1) *a = b; else assert(*a == b);

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


#define MIN(a,b) ((a) <= (b) ? (a) : (b))
#define MAX(a,b) ((a) >= (b) ? (a) : (b))

/* RAND_DOUBLE -> random number between 0 and 1 (exclusive) */
#define RAND_DOUBLE ( ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0) )
/* RAND_UNIFORM(a,b) -> random value between a and b */
#define RAND_UNIFORM(a,b) (((double) (a)) + ((double) ((b)-(a))) * RAND_DOUBLE)

#define SPHERE2X(azi, incl) sin(DEG2RAD * incl)*cos(DEG2RAD * azi)
#define SPHERE2Y(azi, incl) sin(DEG2RAD * incl)*sin(DEG2RAD * azi)
#define SPHERE2Z(azi, incl) cos(DEG2RAD * incl)

/* DELTA_TIME and DELTA_DIST are used in evaluating the answer */
#define DELTA_TIME 50                        /* in seconds */
#define DELTA_DIST 5                         /* in degrees */

#define MAX_AMP 10000.0
#define LOG_MAX_AMP 9.2103

#define MAX_EVENT_RATE 1

#define MAX_ENVELOPE_LENGTH 50  /* TODO: find a number that is not
				  made up and actually makes sense */

#define MAX_PHASE(ntdp) ntdp

Event_t * alloc_event_sig(SigModel_t * p_sigmodel);
void free_event(Event_t * p_event);
void free_events(int numevents, Event_t * p_events);
void copy_event_sig(SigModel_t * p_sigmodel, Event_t * p_tgt_event,
                const Event_t * p_src_event);
void print_event(const Event_t * p_event);

void print_arrival(const Arrival_t * p_arr);


#define ALLOC_EVENT(net, sig) (net != NULL) ? alloc_event_net(net) : alloc_event_sig(sig);
#define COPY_EVENT(net,sig, a, b) (net != NULL) ? copy_event_net(net, a, b) : copy_event_sig(sig, a, b);

void convert_events_dets_to_pyobj(const EarthModel_t * p_earth,
                             const Event_t ** pp_events, int numevents,
                             PyObject ** pp_eventsobj,
                             PyObject ** pp_evdetlistobj);

void convert_events_arrs_to_pyobj(SigModel_t * p_sigmodel, 
				  const EarthModel_t * p_earth,
				  const Event_t ** pp_events, int numevents,
				  PyObject ** pp_eventsobj,
				  PyObject ** pp_evarrlistobj);

PyObject * channel_bundle_to_trace_bundle(ChannelBundle_t * p_segment);
#endif // SIGVISA_INCLUDE
