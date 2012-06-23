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
  int n2 = PyArray_DIM(py_params_array, 0);

  int nparams = PyArray_DIM(py_params_array, 1);
  if ((n2 != *n) || (nparams != 6)) {
    LogError("convert_arrivals passed invalid parameter array: %d params!\n", nparams);
    exit(EXIT_FAILURE);
  }

  *p_arrs = calloc(*n, sizeof(Arrival_t));
  *pp_arrs = calloc(*n, sizeof(Arrival_t *));

    for(int i=0; i < *n; ++i) {
      (*p_arrs)[i].time = ARRAY2(py_params_array, i, ARR_TIME_PARAM);
      (*p_arrs)[i].peak_time = ARRAY2(py_params_array, i, PEAK_OFFSET_PARAM) + (*p_arrs)[i].time;
      (*p_arrs)[i].peak_amp = ARRAY2(py_params_array, i, PEAK_HEIGHT_PARAM);
      (*p_arrs)[i].peak_decay = ARRAY2(py_params_array, i, PEAK_DECAY_PARAM);
      (*p_arrs)[i].amp = ARRAY2(py_params_array, i, CODA_HEIGHT_PARAM);
      (*p_arrs)[i].coda_decay = ARRAY2(py_params_array, i, CODA_DECAY_PARAM);

      (*p_arrs)[i].phaseid = PyInt_AsLong(PyList_GET_ITEM(py_phaseids_list, i));
      (*pp_arrs)[i] = (*p_arrs)+i;

      printf("converted %f %f %f %f %f %f %d \n", (*p_arrs)[i].time, (*p_arrs)[i].peak_time, (*p_arrs)[i].peak_amp, (*p_arrs)[i].peak_decay, (*p_arrs)[i].amp, (*p_arrs)[i].coda_decay, (*p_arrs)[i].phaseid);
    }

}

void ARProcess_py_to_c(double noise_mean, double noise_variance, PyArrayObject * py_coeffs, ARProcess_t * p_ar_process) {
  npy_intp n = PyArray_DIM(py_coeffs, 0);
  p_ar_process->order = n;
  p_ar_process->mean = noise_mean;
  p_ar_process->sigma2 = noise_variance;
  p_ar_process->coeffs = calloc(n, sizeof(double));
  for (int i=0; i < n; ++i) {
    p_ar_process->coeffs[i] = ARRAY1(py_coeffs, i);
  }
}

PyObject * py_set_noise_process(SigModel_t * p_sigmodel, PyObject * args) {
  int siteid, band, chan;
  double noise_mean, noise_variance;
  PyArrayObject * py_coeffs;
  if (!PyArg_ParseTuple(args, "iiiddO!", &siteid, &band, &chan, &noise_mean, &noise_variance, &PyArray_Type, &py_coeffs))
      return NULL;

  printf("set noise process for channel %d band %d\n", chan, band);

  ARProcess_t * p_ar = &(((Spectral_Envelope_Model_t * )(p_sigmodel->signal_model.pv_params))->p_stations + siteid-1)->bands[band].channel_noise_models[chan];
  if (p_ar->coeffs != NULL) {
    free(p_ar->coeffs);
    p_ar->coeffs = NULL;
  }
  ARProcess_py_to_c(noise_mean, noise_variance, py_coeffs, p_ar);
  return Py_BuildValue("d", 1);
}

PyObject * py_set_wiggle_process(SigModel_t * p_sigmodel, PyObject * args) {
  int siteid, band, chan, phaseid;
  double noise_mean, noise_variance;
  PyArrayObject * py_coeffs;
  if (!PyArg_ParseTuple(args, "iiiiddO!", &siteid, &band, &chan, &phaseid, &noise_mean, &noise_variance, &PyArray_Type, &py_coeffs))
      return NULL;

  ARProcess_t * p_ar = &(((Spectral_Envelope_Model_t * )(p_sigmodel->signal_model.pv_params))->p_stations + siteid-1)->bands[band].wiggle_model[chan][phaseid-1];
  if (p_ar->coeffs != NULL) {
    free(p_ar->coeffs);
    p_ar->coeffs = NULL;
  }
  ARProcess_py_to_c(noise_mean, noise_variance, py_coeffs, p_ar);
  return Py_BuildValue("d", 1);
}


PyObject * py_trace_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

  PyObject * py_trace;
  PyObject * py_phaseids_list;
  PyObject * py_params_array;
  if (!PyArg_ParseTuple(args, "OO!O!", &py_trace, &PyList_Type, &py_phaseids_list, &PyArray_Type, &py_params_array))
      return NULL;

  Trace_t * p_trace;
  trace_to_signal(py_trace, &p_trace);

  Segment_t segment;
  init_dummy_segment(p_trace, &segment);

  int n;
  Arrival_t * p_arrs;
  Arrival_t ** pp_arrs;
  convert_arrivals(py_phaseids_list, py_params_array, &n, &p_arrs, &pp_arrs);

  SignalModel_t * p_model = &p_sigmodel->signal_model;
  double ll = p_model->likelihood(p_sigmodel, &segment, n, (const Arrival_t **)pp_arrs);

  free(pp_arrs);
  free(p_arrs);
  free_segment_inner(&segment);
  return Py_BuildValue("d", ll);
}

PyObject * py_segment_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

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

  SignalModel_t * p_model = &p_sigmodel->signal_model;
  double ll = p_model->likelihood(p_sigmodel, &segment, n, (const Arrival_t **)pp_arrs);

  free(pp_arrs);
  free(p_arrs);
  return Py_BuildValue("d", ll);
}

PyObject * py_gen_segment(SigModel_t * self, PyObject * args, int sample) {
  int canary = 0;
  double start_time, end_time, srate;
  int siteid;
  PyObject * py_phaseids_list;
  PyObject * py_params_array;

  if (!PyArg_ParseTuple(args, "ddndO!O!", &start_time, &end_time, &siteid, &srate, &PyList_Type, &py_phaseids_list, &PyArray_Type, &py_params_array))
      return NULL;

  Segment_t segment;

  long len = (end_time-start_time)*srate;
  segment.len=len;
  segment.start_time=start_time;
  segment.hz=srate;
  segment.siteid=siteid;
  alloc_segment_inner(&segment);


  int n;
  Arrival_t * p_arrs;
  Arrival_t ** pp_arrs;
  convert_arrivals(py_phaseids_list, py_params_array, &n, &p_arrs, &pp_arrs);

  Spectral_Envelope_Model_Sample(self, &segment, n, (const Arrival_t **)pp_arrs, sample, sample);

  PyObject * py_seg = c_segment_to_py_segment(&segment);
  free(p_arrs);
  free(pp_arrs);
  free_segment_inner(&segment);


  return py_seg;
}

PyObject * py_gen_logenvelope_segment(SigModel_t * self, PyObject * args) {
  int a = 0;

  PyObject * s = py_gen_segment(self, args, FALSE);

  return s;
}

PyObject * py_sample_segment(SigModel_t * self, PyObject * args) {
  PyObject * s =  py_gen_segment(self, args, TRUE);
  return s;
}

PyObject * py_gen_trace(SigModel_t * self, PyObject * args, int sample) {
  int canary = 0;
  double start_time, end_time, srate;
  int siteid, chan, band;
  PyObject * py_phaseids_list;
  PyObject * py_params_array;

  if (!PyArg_ParseTuple(args, "ddiiidO!O!", &start_time, &end_time, &siteid, &band, &chan, &srate, &PyList_Type, &py_phaseids_list, &PyArray_Type, &py_params_array))
      return NULL;

  Trace_t * p_trace = alloc_trace();

  long len = (end_time-start_time)*srate;
  p_trace->p_data = calloc(len, sizeof(double));
  p_trace->len=len;
  p_trace->start_time=start_time;
  p_trace->hz=srate;
  p_trace->siteid=siteid;
  p_trace->chan=chan;
  p_trace->band=band;

  int n;
  Arrival_t * p_arrs;
  Arrival_t ** pp_arrs;
  convert_arrivals(py_phaseids_list, py_params_array, &n, &p_arrs, &pp_arrs);

  Spectral_Envelope_Model_Sample_Trace(self, p_trace, n, (const Arrival_t **)pp_arrs, sample, sample);

  PyObject * py_trace = build_trace(p_trace);
  free(p_arrs);
  free(pp_arrs);
  free_trace(p_trace);

  return py_trace;
}

PyObject * py_gen_logenvelope_trace(SigModel_t * self, PyObject * args) {
  PyObject * s = py_gen_trace(self, args, FALSE);
  return s;
}

PyObject * py_sample_trace(SigModel_t * self, PyObject * args) {
  PyObject * s =  py_gen_trace(self, args, TRUE);
  return s;
}
