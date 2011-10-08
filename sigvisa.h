#ifndef SIGVISA_INCLUDE
#define SIGVISA_INCLUDE

#include <Python.h>

#include "netvisa.h"

/* define signal channels */
#define CHAN_BHE    0
#define CHAN_BHN    1
#define CHAN_BHZ    2
#define CHAN_OTHER  3

#define CHECK_ERROR if(PyErr_Occurred()) { PyErr_Print(); exit(1); }

typedef struct Signal_t
{

  int len;
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

#include "priors/ArrivalTimeJointPrior.h"
#include "priors/SignalPrior.h"

int print_signal(Signal_t * signal);

typedef struct SigModel_t
{
  PyObject_HEAD

  double start_time;
  double end_time;

  int numsignals;
  Signal_t * p_signals;

  EarthModel_t * p_earth;

  //StationNoisePrior sta_noise_prior;
  //EnvelopePrior env_prior;
  SignalPrior_t sig_prior;

  NumEventPrior_t num_event_prior;
  EventLocationPrior_t event_location_prior;
  EventMagPrior_t event_mag_prior;

  ArrivalTimeJointPrior_t arr_time_joint_prior;

  /*
  NumSecDetPrior_t num_secdet_prior;
  EventDetectionPrior_t event_det_prior;
  ArrivalTimePrior_t arr_time_prior;
  NumFalseDetPrior_t num_falsedet_prior;
  ArrivalAzimuthPrior_t arr_az_prior;
  ArrivalSlownessPrior_t arr_slo_prior;
  ArrivalPhasePrior_t arr_phase_prior;
  ArrivalSNRPrior_t arr_snr_prior;
  ArrivalAmplitudePrior_t arr_amp_prior;*/
} SigModel_t;

#include "priors/score_sig.h"

#include "priors/score.h"
#include "infer/infer.h"
#include "infer/propose.h"

#define MAX_EVENT_RATE 1

#endif // SIGVISA_INCLUDE
