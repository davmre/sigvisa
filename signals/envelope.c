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


void convert_arrivals(PyObject * py_phaseids_list, PyObject * py_params_array, int *n, Arrival_t ** p_arrs, Arrival_t *** pp_arrs) {

  *n = PyList_Size(py_phaseids_list);
  *p_arrs = calloc(*n, sizeof(Arrival_t));
  *pp_arrs = calloc(*n, sizeof(Arrival_t *));
  for(int i=0; i < *n; ++i) {
    (*p_arrs)[i].time = ARRAY2(py_params_array, i, ARR_TIME_PARAM);
    (*p_arrs)[i].peak_time = ARRAY2(py_params_array, i, PEAK_OFFSET_PARAM) + (*p_arrs)[i].time;
    (*p_arrs)[i].peak_amp = ARRAY2(py_params_array, i, PEAK_HEIGHT_PARAM);
    (*p_arrs)[i].peak_decay = ARRAY2(py_params_array, i, PEAK_DECAY_PARAM);
    (*p_arrs)[i].amp = ARRAY2(py_params_array, i, CODA_HEIGHT_PARAM);
    (*p_arrs)[i].coda_decay = ARRAY2(py_params_array, i, CODA_DECAY_PARAM);

    (*pp_arrs)[i] = (*p_arrs)+i;
  }

}

PyObject * py_trace_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

  double start_time, end_time, noise_floor, srate;
  int siteid;
  PyObject * py_trace;
  PyObject * py_phaseids_list;
  PyObject * py_params_array;
    // def generate_trace(self, start_time, end_time, siteid, noise_floor, srate, phaseids, params):
  if (!PyArg_ParseTuple(args, "OO!O!", &py_trace, &PyList_Type, &py_phaseids_list, &PyArray_Type, &py_params_array))
      return NULL;

  Trace_t * p_trace;
  trace_to_signal(py_trace, &p_trace);
  
  Segment_t segment;
  segment.len = p_trace->len;
  segment.start_time = p_trace->start_time;
  segment.hz = p_trace->hz;
  segment.siteid = p_trace->siteid;
  Channel_t channel;
  segment.channels[p_trace->chan] = &channel;
  channel.len = p_trace->len;
  channel.start_time = p_trace->start_time;
  channel.hz = p_trace->hz;
  channel.siteid = p_trace->siteid;
  channel.chan = p_trace->chan;

  // TODO: fix this once we start to handle bands properly
  //  channel.p_bands[p_trace->band] = p_trace;
  channel.p_bands[BB_ENVELOPE] = p_trace;
  
  int n;
  Arrival_t * p_arrs;
  Arrival_t ** pp_arrs;
  convert_arrivals(py_phaseids_list, py_params_array, &n, &p_arrs, &pp_arrs);

  double ll = Spectral_Envelope_Model_Likelihood(p_sigmodel, n, pp_arrs);

  free(pp_arrs);
  free(arrs);
  free(p_trace);
  return Py_BuildValue("d", ll);
}

PyObject * py_segment_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

  double start_time, end_time, noise_floor, srate;
  int siteid;
  PyObject * py_segment;
  PyObject * py_phaseids_list;
  PyObject * py_params_array;
    // def generate_trace(self, start_time, end_time, siteid, noise_floor, srate, phaseids, params):
  if (!PyArg_ParseTuple(args, "OO!O!", &py_segment, &PyList_Type, &py_phaseids_list, &PyArray_Type, &py_params_array))
      return NULL;

  Segment_t segment;
  py_segment_to_c_segment(py_segment, &segment);

  int n;
  Arrival_t * p_arrs;
  Arrival_t ** pp_arrs;
  convert_arrivals(py_phaseids_list, py_params_array, &n, &p_arrs, &pp_arrs);

  double ll = Spectral_Envelope_Model_Likelihood(p_sigmodel, &segment, n, pp_arrs);

  free(pp_arrs);
  free(arrs);
  return Py_BuildValue("d", ll);
}

PyObject * py_gen_logenvelope(SigModel_t * self, PyObject * args) {

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
