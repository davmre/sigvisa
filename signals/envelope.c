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



int add_phase_arrival_model2(Arrival_t * p_arrival, Trace_t * p_trace) {

  double * d = p_trace->p_data;

  long arrival_offset_idx = (p_arrival->time - p_trace->start_time) * p_trace->hz;
  long peak_offset_idx = (p_arrival->peak_time - p_trace->start_time) * p_trace->hz;

  for (long t=arrival_offset_idx; t < p_trace->len; ++t) {
    d[t] = LOGSUM(d[t], p_arrival->peak_amp - pow((t - peak_offset_idx)/(p_trace->hz), 2) * pow(p_arrival->peak_decay, 2));
  }
  for (long t=peak_offset_idx; t < p_trace->len; ++t) {
    double t_off = (t - peak_offset_idx)/p_trace->hz;
    double signal = p_arrival->amp + p_arrival->coda_decay*t_off;
    d[t] = LOGSUM(d[t], signal);
  }
}



int add_phase_arrival(Arrival_t * p_arrival, Trace_t * p_trace) {

  double * d = p_trace->p_data;

  long arrival_offset_idx = (p_arrival->time - p_trace->start_time) * p_trace->hz;
  long peak_offset_idx = (p_arrival->peak_time - p_trace->start_time) * p_trace->hz;

  double real_peak_height = LOGSUM(p_trace->noise_floor, p_arrival->peak_amp);
  double onset_slope = (real_peak_height-p_trace->noise_floor)/(peak_offset_idx - arrival_offset_idx);
  double alpha = pow(exp(p_arrival->amp) / (exp(p_arrival->peak_amp) - exp(p_arrival->amp)), p_arrival->peak_decay );

  for (long t=MAX(0, arrival_offset_idx); t < peak_offset_idx; ++t) {
    d[t] = LOGSUM(d[t], (t - arrival_offset_idx) * onset_slope);
  }
  for (long t=MAX(0, peak_offset_idx); t < p_trace->len; ++t) {
    double t_off = (t - peak_offset_idx)/p_trace->hz;
    double signal = p_arrival->amp + log(1+ 1.0/(pow(alpha + t_off, p_arrival->peak_decay))) + p_arrival->coda_decay*t_off;
    d[t] = LOGSUM(d[t], signal);
  }
 }

int gen_logenvelope(int n, Arrival_t * p_arrivals, Trace_t * p_trace) {

  p_trace->p_data = (double*) calloc(p_trace->len, sizeof(double));

  if (p_trace->p_data == NULL) {
    printf("error allocating memory for Trace in gen_logenvelope, len %ld\n", p_trace->len);
    exit(-1);
  }

  for (int t=0; t < p_trace->len; ++t) {
    p_trace->p_data[t] = p_trace->noise_floor;
  }

  for(int i=0; i < n; ++i) {
    add_phase_arrival(p_arrivals + i, p_trace);
  }

}


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

  trace.py_array = NULL;
  trace.p_data = (double*) calloc(trace.len, sizeof(double));
  if (trace.p_data == NULL) {
    printf("error allocating memory for Trace in gen_logenvelope, len %ld\n", trace.len);
    exit(-1);
  }
  for (long t=0; t < len; ++t) {
    trace.p_data[t] = noise_floor;
  }

  Arrival_t arr;
  int n = PyList_Size(py_phaseids_list);
  for(int i=0; i < n; ++i) {
    arr.time = ARRAY2(py_params_array, i, ARR_TIME_PARAM);
    arr.peak_time = ARRAY2(py_params_array, i, PEAK_OFFSET_PARAM) + arr.time;
    arr.peak_amp = ARRAY2(py_params_array, i, PEAK_HEIGHT_PARAM);
    arr.peak_decay = ARRAY2(py_params_array, i, PEAK_DECAY_PARAM);
    arr.amp = ARRAY2(py_params_array, i, CODA_HEIGHT_PARAM);
    arr.coda_decay = ARRAY2(py_params_array, i, CODA_DECAY_PARAM);

    // printf("%d %f %f %f %f %f %f\n", i, arr.time, arr.peak_time, arr.peak_amp, arr.peak_decay, arr.amp, arr.coda_decay);

    add_phase_arrival(&arr, &trace);
  }

  PyObject * tr = build_trace(&trace);
  return tr;
}
