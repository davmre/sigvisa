#ifndef NETVISA_INCLUDE
#define NETVISA_INCLUDE

#include <Python.h>
/* we don't want _import_array defined in all files except the main
 * module of netvisa */
#if !defined NETVISA_MAIN_MODULE
  #define NO_IMPORT_ARRAY
#endif


#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"
#include "priors/NumEventPrior.h"
#include "priors/EventLocationPrior.h"
#include "priors/EventMagPrior.h"
#include "priors/EventDetectionPrior.h"
#include "priors/ArrivalTimePrior.h"
#include "priors/ArrivalAzimuthPrior.h"
#include "priors/ArrivalSlownessPrior.h"
#include "priors/ArrivalPhasePrior.h"
#include "priors/ArrivalAmplitudePrior.h"
#include "priors/EarthModel.h"
#include "priors/Laplace.h"
#include "priors/Poisson.h"
#include "priors/Gaussian.h"
#include "priors/LogNormal.h"
#include "priors/Gamma.h"

#include "priors/SecDetPrior.h"

#include "priors/NumFalseDetPrior.h"
#include "priors/ArrivalSNRPrior.h"

typedef struct NetModel_t
{

  // test 123
  PyObject_HEAD

  double start_time;
  double end_time;
  
  int numdetections;
  struct Detection_t * p_detections;

  int numsites;
  int numtime;       /* number of quantized time slots in p_site_up */
  
  int * p_site_up;                           /* numsites x numtime */

  EarthModel_t * p_earth;
  
  SecDetPrior_t secdet_prior;
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
  ArrivalAmplitudePrior_t arr_amp_prior;
  
} NetModel_t;

#define UPTIME_QUANT     3600                /* 1 hour */

#define NetModel_IsSiteUp(p_netmodel, siteid, arrtime)\
  ((arrtime >= (p_netmodel)->start_time) && (arrtime < (p_netmodel)->end_time)\
   && (p_netmodel)->p_site_up[(siteid) * (p_netmodel)->numtime\
   + ((int) floor(((arrtime) - (p_netmodel)->start_time) / UPTIME_QUANT))])



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

/* the spacing between secondary detections */
#define SECDET_INTERVAL ((double) 5.0)

/* maximum number of primary + secondary detections for any phase */
#define MAX_PHASE_DET  15

/* max time residual should not be needed in a proper model but due to
 * unexplained extremely high variance in some of the travel time residual
 * we need to put this in place for now */
#define MAX_TIME_RESIDUAL ((double) 6.0)


Event_t * alloc_event_net(NetModel_t * p_netmodel);
void copy_event_net(NetModel_t * p_netmodel, Event_t * p_tgt_event,
                const Event_t * p_src_event);
void print_event_detections(EarthModel_t * p_earth, const Event_t * p_event);

#endif // NETVISA_INCLUDE
