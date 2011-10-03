#include <Python.h>
#include "numpy/arrayobject.h"
#include "netvisa.h"


typedef struct SigModel_t
{
  PyObject_HEAD

  double start_time;
  double end_time;

  PyObject * siteid; /* dictionary mapping station strings to integer site ids */
  
  EarthModel_t * p_earth;
  NetModel_t * p_netmodel;
  

  StationNoisePrior sta_noise_prior;
  EnvelopePrior env_prior;
  /*
  NumSecDetPrior_t num_secdet_prior;
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
  ArrivalAmplitudePrior_t arr_amp_prior;*/
  
} SigModel_t;

#define MAX_EVENT_RATE 1
