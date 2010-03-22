#include <Python.h>
#include "numpy/arrayobject.h"

#include "priors/NumEventPrior.h"
#include "priors/EventLocationPrior.h"

typedef struct NetModel_t
{
  PyObject_HEAD

  NumEventPrior_t num_event_prior;
  EventLocationPrior_t event_location_prior;
  
} NetModel_t;

#include "priors/score.h"

/* event array columns */
#define EV_LON_COL   0
#define EV_LAT_COL   1
#define EV_DEPTH_COL 2
#define EV_TIME_COL  3
#define EV_MB_COL    4
#define EV_NUM_COLS  5

