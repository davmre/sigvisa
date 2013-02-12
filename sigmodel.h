#ifndef SIGMODEL_INCLUDE
#define SIGMODEL_INCLUDE

#include <Python.h>
#include <math.h>
#include <float.h>

#undef NDEBUG
#include <assert.h>

#include "priors/EarthModel.h"

#include "priors/NumEventPrior.h"
#include "priors/EventLocationPrior.h"
#include "priors/EventMagPrior.h"
#include "priors/ArrivalTimeJointPrior.h"
#include "priors/ArrivalAzimuthPrior.h"
#include "priors/ArrivalSlownessPrior.h"
#include "priors/ArrivalAmplitudePrior.h"

#include "priors/Laplace.h"
#include "priors/Poisson.h"
#include "priors/Gaussian.h"
#include "priors/LogNormal.h"
#include "priors/Gamma.h"

typedef struct SigModel_t
{
  PyObject_HEAD

  EarthModel_t * p_earth;

  NumEventPrior_t num_event_prior;
  EventLocationPrior_t event_location_prior;
  EventMagPrior_t event_mag_prior;

  ArrivalTimeJointPrior_t arr_time_joint_prior;
  ArrivalAzimuthPrior_t arr_az_prior;
  ArrivalSlownessPrior_t arr_slo_prior;
  ArrivalAmplitudePrior_t arr_amp_prior;

} SigModel_t;

int have_signal(SigModel_t * p_sigmodel, int site, double start_time, double end_time);

#endif // SIGMODEL_INCLUDE
