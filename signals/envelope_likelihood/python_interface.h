#ifndef SIGNALS_PYTHON_INTERFACE_H
#define SIGNALS_PYTHON_INTERFACE_H

#include <Python.h>
#include "../../sigmodel.h"

PyObject * py_set_noise_process(SigModel_t * p_sigmodel, PyObject * args);
PyObject * py_set_wiggle_process(SigModel_t * p_sigmodel, PyObject * args);

PyObject * py_trace_likelihood(SigModel_t * p_sigmodel, PyObject * args);
PyObject * py_segment_likelihood(SigModel_t * p_sigmodel, PyObject * args);

PyObject * py_gen_logenvelope_segment(SigModel_t * self, PyObject * args);
PyObject * py_sample_segment(SigModel_t * self, PyObject * args);
PyObject * py_gen_logenvelope_trace(SigModel_t * self, PyObject * args);
PyObject * py_sample_trace(SigModel_t * self, PyObject * args);

#endif
