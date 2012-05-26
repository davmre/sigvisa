#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../sigvisa.h"
#include "SignalModelUtil.h"


PyObject * py_gen_logenvelope(PyObject * self, PyObject * args) {

  double start_time, end_time, noise_floor, srate;
  int siteid;
  PyObject * py_phaseids_list;
  PyObject * py_params_array;
    // def generate_trace(self, start_time, end_time, siteid, noise_floor, srate, phaseids, params):
  if (!PyArg_ParseTuple(args, "ddnddO!O!", &start_time, &end_time, &siteid, &noise_floor, &srate, &PyList_Type, &py_phaseids_list, &PyArray_Type, &py_params_array))
      return NULL;

  Trace_t trace;

  long len = (end_time-start_time)*srate;
  trace.len=len;
  trace.start_time=start_time;
  trace.hz=srate;
  trace.noise_floor=noise_floor;
  trace.siteid=siteid;

  int n = PyList_Size(py_phaseids_list);
  Arrival_t * arrs = calloc(n, sizeof(Arrival_t));
  for(int i=0; i < n; ++i) {
    arrs[i].time = ARRAY2(py_params_array, i, ARR_TIME_PARAM);
    arrs[i].peak_time = ARRAY2(py_params_array, i, PEAK_OFFSET_PARAM) + arrs[i].time;
    arrs[i].peak_amp = ARRAY2(py_params_array, i, PEAK_HEIGHT_PARAM);
    arrs[i].peak_decay = ARRAY2(py_params_array, i, PEAK_DECAY_PARAM);
    arrs[i].amp = ARRAY2(py_params_array, i, CODA_HEIGHT_PARAM);
    arrs[i].coda_decay = ARRAY2(py_params_array, i, CODA_DECAY_PARAM);
  }

  generate_log_envelope(n, arrs, &trace);

  PyObject * tr = build_trace(&trace);
  free(arrs);
  return tr;
}
