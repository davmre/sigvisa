#ifndef SIGMODEL_INCLUDE
#define SIGMODEL_INCLUDE

#include <Python.h>
#include <math.h>
#include <float.h>

#undef NDEBUG
#include <assert.h>

#include "priors/EarthModel.h"


#include "priors/EventLocationPrior.h"

#include "priors/Laplace.h"
#include "priors/Poisson.h"
#include "priors/Gaussian.h"
#include "priors/LogNormal.h"
#include "priors/Gamma.h"

typedef struct SigModel_t
{
  PyObject_HEAD

  EarthModel_t * p_earth;

  EventLocationPrior_t event_location_prior;

} SigModel_t;

int have_signal(SigModel_t * p_sigmodel, int site, double start_time, double end_time);

#endif // SIGMODEL_INCLUDE
