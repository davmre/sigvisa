#include <math.h>
#include <stdlib.h>

#define NETVISA_MAIN_MODULE
#include "sigvisa.h"

static int py_sig_model_init(SigModel_t *self, PyObject *args);
static void py_sig_model_dealloc(SigModel_t * self);
static PyObject * py_set_signals(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_get_signals(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_set_fake_detections(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_synthesize_signals(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_synthesize_signals_det(SigModel_t *p_sigmodel, PyObject *args);
PyObject * py_event_likelihood(SigModel_t * p_sigmodel, PyObject * args);

static PyObject * py_canonical_channel_num(PyObject * self, PyObject * args);

extern PyTypeObject py_EarthModel;

static PyMethodDef SigModel_methods[] = {
  {"set_signals", (PyCFunction)py_set_signals, METH_VARARGS,
   "set_signals(traces) "
   "-> num_translated\n"},
  {"synthesize_signals", (PyCFunction)py_synthesize_signals, METH_VARARGS,
   "synthesize_signals(evlist, stalist, start_time, end_time, hz, samplePerturb, sampleNoise) "
   "-> success\n"},
  {"synthesize_signals_det", (PyCFunction)py_synthesize_signals_det, METH_VARARGS,
   "synthesize_signals(stalist, start_time, end_time, hz, samplePerturb, sampleNoise) "
   "-> success\n"},
  {"get_signals", (PyCFunction)py_get_signals, METH_VARARGS,
   "get_signals() "
   "-> signals\n"},
  {"set_fake_detections", (PyCFunction)py_set_fake_detections, METH_VARARGS,
   "set_fake_detections(fake_detections) "
   "-> num_translated\n"},
  {"arrival_likelihood", (PyCFunction)py_arr_likelihood, METH_VARARGS,
   "arrival_likelihood(arrtime, arramp, arrazi, arrslo, arrphase, arrsiteid, write_log)"
   "-> log likelihood\n"},
  {"set_signal_params", (PyCFunction)py_set_params, METH_VARARGS,
   "set_signal_params(siteid, param_dict)"
   "-> success\n"},
  {"set_all_signal_params", (PyCFunction)py_set_all_params, METH_VARARGS,
   "set_all_signal_params(siteid_dict)"
   "-> success\n"},
  {"event_likelihood", (PyCFunction)py_event_likelihood, METH_VARARGS,
   "event_likelihood(time, lon, lat, depth, mb)"
   "-> log likelihood\n"},
  {"detection_likelihood", (PyCFunction)py_det_likelihood, METH_VARARGS,
   "detection_likelihood(write_log)"
   "-> log likelihood\n"},
  {"infer", (PyCFunction)py_infer_sig, METH_VARARGS,
   "infer(runid, numsamples, birthsteps, window, step, threads, propose_events, verbose,"
   "write_cb, log_cb)\n -> events, ev_detlist"},/*
  {"propose", (PyCFunction)py_propose, METH_VARARGS,
   "propose(time_low, time_high, det_low, det_high, degree_step, num_step)\n"
   " -> events, ev_detlist"},*/
  {NULL}  /* Sentinel */
};

static PyTypeObject py_sig_model = {
    PyObject_HEAD_INIT(NULL)
    0,                                       /*ob_size*/
    "sigvisa.SigModel",                      /*tp_name*/
    sizeof(SigModel_t),                      /*tp_basicsize*/
    0,                                       /*tp_itemsize*/
    (destructor)py_sig_model_dealloc,        /*tp_dealloc*/
    0,                                       /*tp_print*/
    0,                                       /*tp_getattr*/
    0,                                       /*tp_setattr*/
    0,                                       /*tp_compare*/
    0,                                       /*tp_repr*/
    0,                                       /*tp_as_number*/
    0,                                       /*tp_as_sequence*/
    0,                                       /*tp_as_mapping*/
    0,                                       /*tp_hash */
    0,                                       /*tp_call*/
    0,                                       /*tp_str*/
    0,                                       /*tp_getattro*/
    0,                                       /*tp_setattro*/
    0,                                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                      /*tp_flags*/
    "Signal Processing Model",               /* tp_doc */
    0,                                       /* tp_traverse */
    0,                                       /* tp_clear */
    0,                                       /* tp_richcompare */
    0,                                       /* tp_weaklistoffset */
    0,                                       /* tp_iter */
    0,                                       /* tp_iternext */
    SigModel_methods,                        /* tp_methods */
    0,                                       /* tp_members */
    0,                                       /* tp_getset */
    0,                                       /* tp_base */
    0,                                       /* tp_dict */
    0,                                       /* tp_descr_get */
    0,                                       /* tp_descr_set */
    0,                                       /* tp_dictoffset */
    (initproc)py_sig_model_init,             /* tp_init */
    0,                                       /* tp_alloc */
    0,                                       /* tp_new */
};

static int py_sig_model_init(SigModel_t *self, PyObject *args)
{
  EarthModel_t * p_earth;
  //PyObject * siteid;
  double start_time;
  double end_time;
  int ar_perturbation;
  const char * signal_model_name;
  const char * numevent_fname;
  const char * evloc_fname;
  const char * evmag_fname;
  const char * arrtime_fname;
  const char * arrazi_fname;
  const char * arrslo_fname;
  const char * arramp_fname;
  PyObject * log_trace_cb;

  if (!PyArg_ParseTuple(args, "OddssssssssO", &p_earth, &start_time, &end_time, &signal_model_name, &numevent_fname, &evloc_fname, &evmag_fname, &arrtime_fname, &arrazi_fname, &arrslo_fname, &arramp_fname, &log_trace_cb)) {
    LogError("can't parse input args");
    CHECK_ERROR;
    return -1;
  }
  if (end_time <= start_time)
  {
    LogError("invalid end time");
    PyErr_SetString(PyExc_ValueError,
                    "SigModel: error: end_time <= start_time");
    return -1;
  }

  LogTrace("init sigmodel...");
  // TODO: check that siteid array is good

  self->start_time = start_time;
  self->end_time = end_time;

  self->p_earth = p_earth;
  Py_INCREF((PyObject *)self->p_earth);

  NumEventPrior_Init_Params(&self->num_event_prior, numevent_fname);
  EventLocationPrior_Init_Params(&self->event_location_prior, evloc_fname);
  EventMagPrior_Init_Params(&self->event_mag_prior, 1, evmag_fname);
  ArrivalTimeJointPrior_Init_Params(&self->arr_time_joint_prior, arrtime_fname);
  ArrivalAzimuthPrior_Init_Params(&self->arr_az_prior, arrazi_fname);
  ArrivalSlownessPrior_Init_Params(&self->arr_slo_prior, arrslo_fname);
  ArrivalAmplitudePrior_Init_Params(&self->arr_amp_prior, arramp_fname);

  int numsites = EarthModel_NumSites(p_earth);
  init_signal_model(&(self->signal_model), signal_model_name, numsites);
  self->log_trace_cb = log_trace_cb;
  Py_INCREF(log_trace_cb);

  return 0;
}

static PyObject * traceClass_obj;

static PyMethodDef sigvisaMethods[] = {
  {"canonical_channel_num", (PyCFunction)py_canonical_channel_num, METH_VARARGS,
   "canonical_channel_num(chan_name) "
   "-> channel_num\n"},
  {"srand", py_srand, METH_VARARGS,
    "srand(seed) : sets the random number generator seed"},
  {NULL, NULL}
};


void initsigvisa(void)
{
  PyObject * m;

  py_sig_model.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_sig_model) < 0)
    return;


  m = Py_InitModule3("sigvisa", sigvisaMethods,
                     "Signal-Based Vertically Integrated Seismological Processing");
  import_array();/* Must be present for NumPy. Called first after above line.*/

  Py_INCREF(&py_sig_model);
  PyModule_AddObject(m, "SigModel", (PyObject *)&py_sig_model);

  PyObject * pmod = PyImport_ImportModule("obspy.core");
  if (!pmod) {
    LogFatal("cannot load module obspy.core!\n");
    exit(-1);
  }
  traceClass_obj = PyObject_GetAttrString(pmod, "Trace");
  Py_DECREF(pmod);

  InitLogger(LogToConsole, stdout);

}

PyObject * py_srand(PyObject * self, PyObject * args)
{
  int seed;

  if (!PyArg_ParseTuple(args, "i", &seed))
    return NULL;

  printf("set seed %i\n", seed);

  srand(seed);

  Py_INCREF(Py_None);

  return Py_None;
}

void segment_dealloc(Segment_t * p_segment) {
  for(int j=0; j < NUM_CHANS; ++j) {
    Channel_t *channel = p_segment->p_channels[j];
    if (channel != NULL) {
      for (int band = 0; band < NUM_BANDS; ++band) {
	if (channel->p_bands[band] != NULL) {
	  if (channel->p_bands[band]->py_data != NULL) {
	    Py_DECREF(channel->p_bands[band]->py_data);
	  }
	  free(channel->p_bands[band]);
	}
      }
      free(p_segment->p_channels[j]);
    }
  }
}

static void py_sig_model_dealloc(SigModel_t * self)
{
  if (self->p_earth)
  {
    Py_DECREF((PyObject *)self->p_earth);
    self->p_earth = NULL;
  }

  if (self->log_trace_cb)
  {
    Py_DECREF((PyObject *)self->log_trace_cb);
    self->log_trace_cb = NULL;
  }


  if(self->p_segments) {
    for (int i=0; i < self->numsegments; ++i) {
      segment_dealloc(self->p_segments + i);
    }
    free(self->p_segments);
  }


  EventLocationPrior_UnInit(&self->event_location_prior);
  ArrivalTimeJointPrior_UnInit(&self->arr_time_joint_prior);

  uninit_signal_model(&self->signal_model);

}

static PyArrayObject * convert_arrtimes(SigModel_t *p_sigmodel, PyObject * p_arrtime_array) {
  /*
     Returns a 3D array indexed by [siteid][eventid][phaseid]

   */
  PyArrayObject * array = (PyArrayObject *) PyArray_ContiguousFromAny(p_arrtime_array,
							 NPY_DOUBLE, 3, 3);
  if (array == NULL) {
    LogFatal("arrtimes: argument cannot be interpreted as a " \
	    "three-dimensional numpy array\n");
    exit(1);
  }

 if (array->dimensions[0] != p_sigmodel->p_earth->numsites) {
   LogFatal("arrtimes: first dimension %d does not match the number of sites (%d)\n", (int) (array->dimensions[0]), p_sigmodel->p_earth->numsites);
    exit(1);
 }


 // TODO: make phases work
 /*if (array->dimensions[2] != p_sigmodel->p_earth->numphases) {
  fprintf(stderr, "arrtimes: third dimension %d does not match the number of phases (%d)\n", (int) (array->dimensions[2]), p_sigmodel->p_earth->numphases);
    exit(1);
    }*/

 return array;

}


static Event_t ** indirect_event_list(int numevents, Event_t * evlist) {

  Event_t ** ilist = calloc(numevents, sizeof(Event_t *));

  for (int i=0; i < numevents; ++i) {
    ilist[i] = &evlist[i];
  }
  return ilist;
}

static void convert_tuple_int(PyObject * tuple,
				    int * p_len, int ** pp_ints)
{
  Py_ssize_t len;
  int * p_ints;
  Py_ssize_t i;

  len = PyTuple_Size(tuple);

  p_ints = (int *)calloc(len, sizeof(*p_ints));

  for(i=0; i<len; i++)
  {
    p_ints[i] = PyInt_AsLong(PyTuple_GetItem(tuple, i));
  }

  *p_len = len;
  *pp_ints = p_ints;
}


static void convert_eventobj_no_det(PyArrayObject * p_events_arrobj,
				    int * p_numevents, Event_t ** p_p_events)
{
  Py_ssize_t numevents;
  Event_t * p_events;
  Py_ssize_t i;

  numevents = p_events_arrobj->dimensions[0];

  p_events = (Event_t *)calloc(numevents, sizeof(*p_events));

  for(i=0; i<numevents; i++)
  {
    Event_t * p_event;

    p_event = p_events + i;

    p_event->evlon = ARRAY2(p_events_arrobj, i, EV_LON_COL);
    p_event->evlat = ARRAY2(p_events_arrobj, i, EV_LAT_COL);
    p_event->evdepth = ARRAY2(p_events_arrobj, i, EV_DEPTH_COL);
    p_event->evtime = ARRAY2(p_events_arrobj, i, EV_TIME_COL);
    p_event->evmag = ARRAY2(p_events_arrobj, i, EV_MB_COL);

  }

  *p_numevents = numevents;
  *p_p_events = p_events;
}


 int canonical_band_num(char* band_str) {
   int result = -1;
   if (strcmp("narrow_logenvelope_0.50_0.70", band_str) == 0) {
     result = NARROW_05_07;
   } else if (strcmp("narrow_logenvelope_0.70_1.00", band_str) == 0) {
     result = NARROW_07_10;
   } else if (strcmp("narrow_logenvelope_1.00_1.50", band_str) == 0) {
     result = NARROW_10_15;
   } else if (strcmp("narrow_logenvelope_1.50_2.00", band_str) == 0) {
     result = NARROW_15_20;
   } else if (strcmp("narrow_logenvelope_2.00_3.00", band_str) == 0) {
     result = NARROW_20_30;
   } else if (strcmp("narrow_logenvelope_3.00_4.00", band_str) == 0) {
     result = NARROW_30_40;
   } else if (strcmp("narrow_logenvelope_4.00_6.00", band_str) == 0) {
     result = NARROW_40_60;
   } else if (strcmp("narrow_logenvelope_6.00_8.00", band_str) == 0) {
     result = NARROW_60_80;
   } else if (strcmp("broadband", band_str) == 0) {
     result = BROADBAND;
   } else if (strcmp("broadband_envelope", band_str) == 0) {
     result = BB_ENVELOPE;
   } else {
     LogError("unrecognized band name %s", band_str);
     exit(EXIT_FAILURE);
   }

   return result;
 }


PyObject * canonical_band_name(int num) {

  switch(num) {
  case NARROW_05_07:
    return PyString_FromString("narrow_logenvelope_0.50_0.70");
 case NARROW_07_10:
    return PyString_FromString("narrow_logenvelope_0.70_1.00");
  case NARROW_10_15:
    return PyString_FromString("narrow_logenvelope_1.00_1.50");
  case NARROW_15_20:
    return PyString_FromString("narrow_logenvelope_1.50_2.00");
  case NARROW_20_30:
    return PyString_FromString("narrow_logenvelope_2.00_3.00");
  case NARROW_30_40:
    return PyString_FromString("narrow_logenvelope_3.00_4.0");
  case NARROW_40_60:
    return PyString_FromString("narrow_logenvelope_4.00_6.00");
  case NARROW_60_80:
    return PyString_FromString("narrow_logenvelope_6.00_8.00");
  case BROADBAND:
    return PyString_FromString("broadband");
  case BB_ENVELOPE:
    return PyString_FromString("broadband_envelope");
  }

  LogError("invalid frequency band num %d", num);
  exit(EXIT_FAILURE);
}

PyObject* canonical_channel_name(int num) {
   switch (num) {
   case CHAN_BHZ:
     return PyString_FromString("BHZ");
   case CHAN_BHE:
     return PyString_FromString("BHE");
   case CHAN_BHN:
     return PyString_FromString("BHN");
   case CHAN_HORIZ_AVG:
     return PyString_FromString("horiz_avg");
   case CHAN_OTHER:
   default:
     return PyString_FromString("");
   }
 }


 int canonical_channel_num(char* chan_str) {
   int result = -1;
   if (strcmp("BHZ", chan_str) == 0) {
     result = CHAN_BHZ;
   } else if (strcmp("BHE", chan_str) == 0 || strcmp("BH1", chan_str) == 0) {
     result = CHAN_BHE;
   } else if (strcmp("BHN", chan_str) == 0 || strcmp("BH2", chan_str) == 0) {
     result = CHAN_BHN;
   } else if (strcmp("horiz_avg", chan_str) == 0) {
     result = CHAN_HORIZ_AVG;
   } else {
     result = CHAN_OTHER;
   }
   return result;
 }


static PyObject * py_canonical_channel_num(PyObject * self, PyObject * args) {
  PyObject * py_chan_str;
  if (!PyArg_ParseTuple(args, "O!", &PyString_Type, &py_chan_str))
    return NULL;

   int result = canonical_channel_num(PyString_AsString(py_chan_str));
   return Py_BuildValue("n", result);
 }





/* return the score of an event, using optimal arrival times */
PyObject * py_event_score(SigModel_t * p_sigmodel, PyObject * args) {

  LogInfo ("hello!");
  double time, lon, lat, depth, mb;
  if (!PyArg_ParseTuple(args, "ddddd", &time, &lon, &lat, &depth, &mb))
    return NULL;

  Event_t * p_event = ALLOC_EVENT(NULL, p_sigmodel);

    p_event->evtime = time;
    p_event->evlon = lon;
    p_event->evlat = lat;
    p_event->evdepth = depth;
    p_event->evmag = mb;

    LogInfo("optimizing...");
    optimize_arrivals(p_sigmodel, p_event, 0, NULL);
    LogInfo("scoring...");
    score_event_sig(p_sigmodel, p_event, 0, NULL);
    LogInfo("score %lf",p_event->evscore);
    double score = p_event->evscore;

    free(p_event);

    return Py_BuildValue("d", score);
}


int pydict_get_double(PyObject * py_dict, char * key, double *value) {
  int retcode = 0;
  PyObject * py_value = PyDict_GetItemString(stats, py_key); CHECK_ERROR;
  if (py_value != NULL) {
    *value = PyFloat_AsDouble(py_value);
    retcode = 1;
  }
  return retcode;
}

int pydict_get_int(PyObject * py_dict, char * key, long * value) {
  int retcode = 0;
  PyObject * py_value = PyDict_GetItemString(stats, py_key); CHECK_ERROR;
  if (py_value != NULL) {
    *value = PyInt_AsLong(py_value);
    retcode = 1;
  }
  return retcode;
}

int pydict_get_string(PyObject * py_dict, char * key, char ** value) {
  int retcode = 0;
  PyObject * py_value = PyDict_GetItemString(stats, py_key); CHECK_ERROR;
  if (py_value != NULL) {
    *value = PyString_AsString(py_value);
    retcode = 1;
  }
  return retcode;
}


/*
****************************************************************
py_set_signals and associated helper methods for converting Python
signal structures into C:
****************************************************************
*/

int trace_to_signal(PyObject * py_trace, Trace_t ** pp_trace) {

  // a trace object contains two members: data, a numpy array, and stats, a python dict.
  PyObject * py_data =  PyObject_GetAttrString(py_trace, "data");
  if (py_data == NULL) {
    LogFatal("error: py_data is null!\n");
    exit(1);
  }

  /*  fprintf(stdout, "py_data is an array with depth %d and dimension %d x %d\n",
	  ((PyArrayObject *)py_data)->nd,
	  ((PyArrayObject *)py_data)->dimensions[0],
	  ((PyArrayObject *)py_data)->dimensions[1]); */


  *pp_trace = calloc(sizeof(Trace_t), 1);

  *py_array = (PyArrayObject *) PyArray_ContiguousFromAny(py_data, NPY_DOUBLE, 1,2);
  Py_INCREF(*py_array);
  Py_DECREF(py_data);
  CHECK_ERROR;
  (*pp_trace)->p_data = (double *) (*py_array)->data;
  (*pp_trace)->py_data = py_array;
  (*pp_trace)->len = PyArray_SIZE(*py_array);

  PyObject * stats = PyObject_GetAttrString(py_trace, "stats");

  pydict_get_double(stats, "starttime_unix", & (*pp_trace)->start_time);
  pydict_get_double(stats, "sampling_rate", & (*pp_trace)->hz);
  pydict_get_int(stats, "siteid", & (*pp_trace)->siteid);

  char * chan_str;
  pydict_get_string(stats, "channel", &chan_str);
  & (*pp_trace)->chan = canonical_channel_num(chan_str)

  pydict_get_double(stats, "noise_floor", & (*pp_trace)->noise_floor);
  pydict_get_double(stats, "p_time", & (*pp_trace)->p_time);
  pydict_get_double(stats, "s_time", & (*pp_trace)->s_time);
  pydict_get_int(stats, "p_phaseid", & (*pp_trace)->p_phaseid);
  pydict_get_int(stats, "s_phaseid", & (*pp_trace)->s_phaseid);

  return 0;
}

int py_segment_to_c_segment(PyObject * py_segment, Segment_t * p_segment) {
  if(!PyDict_Check(py_segment)) {
    LogFatal("py_segment_to_c_segment: expected Python list!\n");
    exit(1);
  }

  p_segment->len = -1;
  p_segment->start_time = -1;
  p_segment->hz = -1;
  p_segment->siteid = -1;

  int i;

  PyObject *py_key, *py_value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(py_segment, &pos, &py_key, &py_value)) {

    int chan = canonical_channel_num(PyString_AsString(py_key)); CHECK_ERROR;
    PyObject * py_band_traces =  py_value;

    Channel_t * p_channel = calloc(1, sizeof(Channel_t));
    p_channel->len = -1;
    p_channel->start_time = -1;
    p_channel->hz = -1;
    p_channel->siteid = -1;
    p_channel->chan = chan;

    PyObject *py_key2, *py_value2;
    Py_ssize_t pos2 = 0;
    while (PyDict_Next(py_band_traces, &pos2, &py_key2, &py_value2)) {

      int band = canonical_band_num(PyString_AsString(py_key2));

      Trace_t * p_trace;

      trace_to_signal(py_value2, &p_trace);
      p_trace->band = band;
      p_channel->p_bands[band] = p_trace;

      /* ensure that all band traces for a given channel have the same attributes */
      UPDATE_AND_VERIFY(&p_channel->len, p_trace->len);
      UPDATE_AND_VERIFY(&p_channel->start_time, p_trace->start_time);
      UPDATE_AND_VERIFY(&p_channel->hz, p_trace->hz);
      UPDATE_AND_VERIFY(&p_channel->siteid, p_trace->siteid);
      UPDATE_AND_VERIFY(&p_channel->chan, p_trace->chan);
    }
    p_segment->p_channels[p_channel->chan] = p_channel;

    /* ensure that all channels within a segment have the same attributes */
    UPDATE_AND_VERIFY(&p_segment->len, p_channel->len);
    UPDATE_AND_VERIFY(&p_segment->start_time, p_channel->start_time);
    UPDATE_AND_VERIFY(&p_segment->hz, p_channel->hz);
    UPDATE_AND_VERIFY(&p_segment->siteid, p_channel->siteid);
  }

  return pos;
}

int py_segments_to_c_segments(SigModel_t * p_sigmodel, PyObject * py_segments, Segment_t ** pp_segments) {

  if(!PyList_Check(py_segments)) {
    LogFatal("trace_bundles_to_signal_bundles: expected Python list!\n");
    exit(1);
  }

  int n = PyList_Size(py_segments);
  (*pp_segments) = calloc(n, sizeof(Segment_t));

  int idx = 0;
  for (int i=0; i < n; ++i) {
    PyObject * py_segment = PyList_GetItem(py_segments, i);

    Segment_t * new_segment = calloc(1, sizeof(Segment_t));
    py_segment_to_c_segment(py_segment, new_segment);

    int skip = 0;
    for(int chan=0; chan < NUM_CHANS; ++chan) {
      if (new_segment->p_channels[chan] != NULL) {
	if (!(*p_sigmodel->signal_model.has_model)(p_sigmodel, new_segment->siteid , chan)) {
	  LogInfo("no signal model available for siteid %d chan %d, skipping segment...", new_segment->siteid, chan);
	  skip = 1;
	  break;
	}
      }
    }

    if (!skip) {
      memcpy((*pp_segments) + idx++, new_segment, sizeof(Segment_t));

      // don't do a deep dealloc because we want to retain the segment data
      free(new_segment);
    } else {
      segment_dealloc(new_segment);
    }
  }

  *pp_segments = realloc(*pp_segments, idx*sizeof(Segment_t));
  return idx;
}

static PyObject * py_set_signals(SigModel_t *p_sigmodel, PyObject *args) {
  PyObject * p_tracelist_obj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_tracelist_obj))
    return NULL;

  if (p_sigmodel->numsegments != 0 && p_sigmodel->p_segments != NULL) {

    for (int i=0; i < p_sigmodel->numsegments; ++i) {
      segment_dealloc(p_sigmodel->p_segments + i);
    }
    free(p_sigmodel->p_segments);
  }

  int n = py_segments_to_c_segments(p_sigmodel, p_tracelist_obj, &p_sigmodel->p_segments);
  p_sigmodel->numsegments = n;

  return Py_BuildValue("n", n);
}



/*
****************************************************************
py_get_signals and associated helper methods for converting C
signal structures into Python:
****************************************************************
*/

PyObject * build_trace(long len, double * p_data, PyArrayObject * py_array, double start_time, double hz, int siteid, int chan) {

  // a trace object contains two members: data, a numpy array, and stats, a python dict.
   npy_intp dims[1];
   dims[0] = len;

   PyObject * py_data;
   if (py_array == NULL) {
     py_data = (PyObject *)PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, p_data);
     Py_DECREF(py_data);
   } else {
     py_data = (PyObject *)py_array;
   }

   PyObject * py_stats = PyDict_New();

   PyObject * key = PyString_FromString("npts_processed");
   PyObject * value = Py_BuildValue("n", len);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("starttime_unix");
   value = Py_BuildValue("d", start_time);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("sampling_rate");
   value = Py_BuildValue("d", hz);
   assert(key != NULL && value != NULL);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("siteid");
   value = Py_BuildValue("n", siteid);
   assert(key != NULL && value != NULL);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("channel");
   value = canonical_channel_name(chan);
   assert(key != NULL && value != NULL);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   PyObject * args = Py_BuildValue("OO", py_data, py_stats);
   PyObject * trace =   PyEval_CallObject(traceClass_obj, args);

  return trace;
}


void channel_to_band_dict(Channel_t * p_channel, PyObject ** py_band_dict) {

  *py_band_dict = PyDict_New();
  for(int j=0; j < NUM_BANDS; ++j) {

    if (p_channel->p_bands[j] != NULL) {
      PyObject * py_band_trace = build_trace(p_channel->len,  p_channel->p_bands[j], p_channel->py_bands[j], p_channel->start_time, p_channel->hz, p_channel->siteid, p_channel->chan);

      PyDict_SetItem(*py_band_dict, canonical_band_name(j), py_band_trace);
    }
  }
}


PyObject * c_segment_to_py_segment(Segment_t * p_segment) {

  PyObject * py_chan_dict = PyDict_New();
  for(int j=0; j < NUM_CHANS; ++j) {
    Channel_t *p_channel = (p_segment)->p_channels[j];
    if (p_channel != NULL) {
      PyObject * py_band_dict;
      channel_to_band_dict(p_channel, &py_band_dict);
      PyDict_SetItem(py_chan_dict, canonical_channel_name(j), py_band_dict);
    }
  }
  return py_chan_dict;
}


static PyObject * py_get_signals(SigModel_t *p_sigmodel, PyObject *args) {
  //PyObject * p_tracelist_obj;
  //  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_tracelist_obj))
  //  return NULL;

  int numsegments = p_sigmodel->numsegments;

  PyObject * py_segments = PyList_New(0);
  for (int i=0; i < numsegments; ++i) {
    PyObject * py_segment = c_segment_to_py_segment(p_sigmodel->p_segments + i);
    PyList_Append(py_segments, py_segment);
  }

  return py_segments;
}

int convert_fake_detections(PyObject * det_list, Detection_t ** pp_detections) {

  if(!PyList_Check(det_list)) {
    LogFatal("convert_fake_detections: expected Python list!\n");
    exit(1);
  }

  int n = PyList_Size(det_list);
  (*pp_detections) = calloc(n, sizeof(Detection_t));

  int i;
  for (i=0; i < n; ++i) {
    PyObject * p_fakedet = PyList_GetItem(det_list, i);

    if (!PyTuple_Check(p_fakedet)) {
      LogFatal("convert_fake_detections: expected Python tuple!\n");
      exit(EXIT_FAILURE);
    }

    int n = PyTuple_Size(p_fakedet);
    if (n != 7) {
      LogFatal("convert_fake_detections: expected tuple of size 7, found size %d", n);
      exit(EXIT_FAILURE);
    }

    (*pp_detections)[i].arid_det = (int)PyInt_AsLong(PyTuple_GetItem(p_fakedet, 0));
    (*pp_detections)[i].site_det = (int)PyInt_AsLong(PyTuple_GetItem(p_fakedet, 1));
    (*pp_detections)[i].time_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 2));
    (*pp_detections)[i].amp_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 3));
    (*pp_detections)[i].azi_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 4));
    (*pp_detections)[i].slo_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 5));
    (*pp_detections)[i].phase_det = (int)PyInt_AsLong(PyTuple_GetItem(p_fakedet, 6));

    CHECK_ERROR;

    (*pp_detections)[i].sigvisa_fake = 1;
  }

  return i;
}

static PyObject * py_set_fake_detections(SigModel_t *p_sigmodel, PyObject *args) {
  PyObject * p_detlist_obj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_detlist_obj))
    return NULL;


  int n = convert_fake_detections(p_detlist_obj, &p_sigmodel->p_detections);
  p_sigmodel->numdetections = n;

  return Py_BuildValue("n", n);
}


void synthesize_signals_dets(SigModel_t *p_sigmodel, int numsiteids, int * p_siteids, double start_time, double end_time, double hz, int samplePerturb, int sampleNoise) {

  EarthModel_t * p_earth = p_sigmodel->p_earth;


  p_sigmodel->numsegments = numsiteids;
  p_sigmodel->p_segments = calloc(numsiteids, sizeof(Segment_t));

  for (int i=0; i < numsiteids; ++i) {
    int siteid = p_siteids[i];

    Segment_t * p_segment = p_sigmodel->p_segments + i;

    p_segment->start_time = start_time;
    p_segment->hz = hz;
    p_segment->siteid = siteid;
    p_segment->len = (int) (end_time - start_time) * hz;

    int num_arrivals;
    Arrival_t ** pp_arrivals;

    det_arrivals((void *)p_sigmodel, p_segment, &num_arrivals, &pp_arrivals);
    (*p_sigmodel->signal_model.sample)(p_sigmodel->signal_model.pv_params,
				       p_sigmodel->p_earth,
				       p_segment,
				       num_arrivals, (const Arrival_t **)pp_arrivals,
				       samplePerturb, sampleNoise);
    for (int i=0; i < num_arrivals; ++i) {
      free(pp_arrivals[i]);
    }
    free(pp_arrivals);

    LogTrace("generated segment at siteid %d w/ length %ld = (%lf - %lf) * %lf\n", siteid, p_segment->len, end_time, start_time, hz);

  }
}

void synthesize_signals(SigModel_t *p_sigmodel, int numevents, Event_t ** pp_events, int numsiteids, int * p_siteids, double start_time, double end_time, double hz, int samplePerturb, int sampleNoise) {

  EarthModel_t * p_earth = p_sigmodel->p_earth;

  int numsites =  EarthModel_NumSites(p_sigmodel->p_earth);
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);


  p_sigmodel->numsegments = numsiteids;
  p_sigmodel->p_segments = calloc(numsiteids, sizeof(Segment_t));


  for (int i=0; i < numevents; ++i) {
    Event_t * p_event = pp_events[i];

    p_event->p_arrivals = calloc(numsites*numtimedefphases, sizeof(Arrival_t));
    initialize_mean_arrivals(p_sigmodel,  p_event);
  }


  for (int i=0; i < numsiteids; ++i) {
    int siteid = p_siteids[i];
    LogInfo("synthing for %d", siteid);


    Segment_t * p_segment = p_sigmodel->p_segments + i;

    p_segment->start_time = start_time;
    p_segment->hz = hz;
    p_segment->siteid = siteid;
    p_segment->len = (int) (end_time - start_time) * hz;

    int num_arrivals;
    Arrival_t ** pp_arrivals;
    arrival_list(p_sigmodel->p_earth, p_segment->siteid, p_segment->start_time, Segment_EndTime(p_segment), numevents, (const Event_t **)pp_events, &num_arrivals, &pp_arrivals);

    LogInfo("siteid %d time %lf end %lf events %d", p_segment->siteid, p_segment->start_time, Segment_EndTime(p_segment), numevents);

    LogInfo("sampling siteid %d with %d arrivals", siteid, num_arrivals);
    (*p_sigmodel->signal_model.sample)(p_sigmodel->signal_model.pv_params,
				       p_sigmodel->p_earth,
				       p_segment,
				       num_arrivals, (const Arrival_t **)pp_arrivals,
				       samplePerturb, sampleNoise);

    /*for (int i=0; i < num_arrivals; ++i) {
      free(pp_arrivals[i]);
    }
    free(pp_arrivals);*/

    LogTrace("generated segment at siteid %d w/ length %ld = (%lf - %lf) * %lf\n", siteid, p_segment->len, end_time, start_time, hz);

  }
}

Channel_t * alloc_channel(Segment_t * p_segment) {
  Channel_t * p_channel = (Channel_t *) calloc(1, sizeof(Channel_t));
  p_channel->start_time = p_segment->start_time;
  p_channel->len = p_segment->len;
  p_channel->hz = p_segment->hz;
  p_channel->siteid = p_segment->siteid;
  return p_channel;
}

static PyObject * py_synthesize_signals(SigModel_t *p_sigmodel, PyObject *args) {
  PyArrayObject * p_evlist_obj;
  PyObject * p_stalist_obj;
  double stime, etime, hz;
  int samplePerturb, sampleNoise;
  if (!PyArg_ParseTuple(args, "O!O!dddii", &PyArray_Type, &p_evlist_obj, &PyTuple_Type, &p_stalist_obj, &stime, &etime, &hz, &samplePerturb, &sampleNoise))
    return NULL;

  int numevents;
  Event_t * p_events;
  convert_eventobj_no_det(p_evlist_obj, &numevents, &p_events);
  Event_t ** pp_events = indirect_event_list(numevents, p_events);

  int numsiteids;
  int * p_siteids;

  convert_tuple_int(p_stalist_obj, &numsiteids, &p_siteids);

  synthesize_signals(p_sigmodel, numevents, pp_events, numsiteids, p_siteids, stime, etime, hz, samplePerturb, sampleNoise);

  free(p_siteids);
  free(pp_events);
  free(p_events);

  return Py_BuildValue("n", 0);
}


static PyObject * py_synthesize_signals_det(SigModel_t *p_sigmodel, PyObject *args) {
  PyArrayObject * p_evlist_obj;
  PyObject * p_stalist_obj;
  double stime, etime, hz;
  int samplePerturb, sampleNoise;

  if (!PyArg_ParseTuple(args, "O!dddii", &PyTuple_Type, &p_stalist_obj, &stime, &etime, &hz,&samplePerturb, &sampleNoise))
    return NULL;

  int numsiteids;
  int * p_siteids;

  convert_tuple_int(p_stalist_obj, &numsiteids, &p_siteids);

  synthesize_signals_dets(p_sigmodel, numsiteids, p_siteids, stime, etime, hz, samplePerturb, sampleNoise);

  free(p_siteids);


  return Py_BuildValue("n", 0);
}


void convert_event_to_pyobj(const Event_t * p_event, PyObject * p_eventsobj, int i) {
  ARRAY2(p_eventsobj, i, EV_LON_COL) = p_event->evlon;
  ARRAY2(p_eventsobj, i, EV_LAT_COL) = p_event->evlat;
  ARRAY2(p_eventsobj, i, EV_DEPTH_COL) = p_event->evdepth;
  ARRAY2(p_eventsobj, i, EV_TIME_COL) = p_event->evtime;
  ARRAY2(p_eventsobj, i, EV_MB_COL) = p_event->evmag;
  ARRAY2(p_eventsobj, i, EV_ORID_COL) = (double) p_event->orid;
  ARRAY2(p_eventsobj, i, EV_SCORE_COL) = p_event->evscore;
}

void convert_events_dets_to_pyobj(const EarthModel_t * p_earth,
                             const Event_t ** pp_events, int numevents,
                             PyObject ** pp_eventsobj,
                             PyObject ** pp_evdetlistobj)
{
  PyObject * p_eventsobj;
  PyObject * p_evdetlistobj;
  npy_intp dims[2];
  int i;

  /* create an array of events */
  dims[0] = numevents;
  dims[1] = EV_NUM_COLS;
  p_eventsobj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

  /* and a list of event detections */
  p_evdetlistobj = PyList_New(0);

  for (i=0; i<numevents; i++)
  {
    PyObject * p_detlistobj;
    const Event_t * p_event;
    int numsites;
    int numtimedefphases;
    int siteid;
    int phaseid;

    p_event = pp_events[i];

    /* store the current event in its row */
    convert_event_to_pyobj(p_event, p_eventsobj, i);

    p_detlistobj = PyList_New(0);

    assert(p_event->p_num_dets != NULL);

    /* copy over the (phaseid, detnum) of the event */
    numsites = EarthModel_NumSites(p_earth);
    numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);


      for (siteid = 0; siteid < numsites; siteid ++) {
	for (phaseid = 0; phaseid < MAX_PHASE(numtimedefphases); phaseid ++)  {
	  if (!USE_PHASE(phaseid)) continue;
	  int numdet;

	  numdet = p_event->p_num_dets[(siteid-1) * numtimedefphases + phaseid];

	  if (numdet > 0) {
	    int pos;
	    PyObject * p_phase_det_obj;

	    /* first the phase and then the detnums */
	    p_phase_det_obj = PyTuple_New(numdet + 1);

	    /* tuple set_item steals a reference so we don't need to decr it */
	    PyTuple_SetItem(p_phase_det_obj, 0, Py_BuildValue("i", phaseid));

	    for (pos=0; pos<numdet; pos++) {
	      int detnum;
	      detnum = p_event->p_all_detids[(siteid-1) * numtimedefphases
					     * MAX_PHASE_DET
					     + phaseid * MAX_PHASE_DET + pos];

	      PyTuple_SetItem(p_phase_det_obj, pos+1,Py_BuildValue("i", detnum));
	    }

	    PyList_Append(p_detlistobj, p_phase_det_obj);
	    /* List Append increments the refcount so we need to
	     * decrement our ref */
	    Py_DECREF(p_phase_det_obj);
	  }
	}
      }


      PyList_Append(p_evdetlistobj, p_detlistobj);
      /* List Append increments the refcount so we need to decrement our ref */
      Py_DECREF(p_detlistobj);

  }

  *pp_eventsobj = p_eventsobj;
  *pp_evdetlistobj = p_evdetlistobj;
}



void convert_events_arrs_to_pyobj(SigModel_t * p_sigmodel,
				  const EarthModel_t * p_earth,
                             const Event_t ** pp_events, int numevents,
                             PyObject ** pp_eventsobj,
                             PyObject ** pp_evarrlistobj)
{
  PyObject * p_eventsobj;
  PyObject * p_evarrlistobj;
  npy_intp dims[2];
  int i;

  /* create an array of events */
  dims[0] = numevents;
  dims[1] = EV_NUM_COLS;
  p_eventsobj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

  /* and a list of event detections */
  p_evarrlistobj = PyList_New(0);

  for (i=0; i<numevents; i++)
  {
    PyObject * p_arrlistobj;
    const Event_t * p_event;
    int numsites;
    int numtimedefphases;
    int siteid;
    int phaseid;

    p_event = pp_events[i];

    char * estr = event_str(p_event);
    LogDebug(" converting to pyobj: event %s", estr);
    free(estr);

    /* store the current event in its row */
    convert_event_to_pyobj(p_event, p_eventsobj, i);

    p_arrlistobj = PyList_New(0);

    /* copy over the (phaseid, detnum) of the event */
    numsites = EarthModel_NumSites(p_earth);
    numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

    assert(p_event->p_arrivals != NULL);

    for (siteid = 0; siteid < numsites; siteid ++) {
      for (phaseid = 0; phaseid < MAX_PHASE(numtimedefphases); phaseid ++)  {
	if (!USE_PHASE(phaseid)) continue;
	Arrival_t * p_arr = p_event->p_arrivals
	  + (siteid-1)*numtimedefphases + phaseid;

	if (!have_signal(p_sigmodel, siteid, p_arr->time - 5, p_arr->time+MAX_ENVELOPE_LENGTH)) {
	  continue;
	}

	PyObject * p_phase_arr_obj;

	/* first the phase and then the detnums */
	p_phase_arr_obj = PyTuple_New(7);

	/* tuple set_item steals a reference so we don't need to decr it */
	PyTuple_SetItem(p_phase_arr_obj, 0, Py_BuildValue("i", i));
	PyTuple_SetItem(p_phase_arr_obj, 1, Py_BuildValue("i", siteid));
	PyTuple_SetItem(p_phase_arr_obj, 2, Py_BuildValue("i", phaseid));

	PyTuple_SetItem(p_phase_arr_obj, 3, Py_BuildValue("d", p_arr->time));
	PyTuple_SetItem(p_phase_arr_obj, 4, Py_BuildValue("d", p_arr->amp));
	PyTuple_SetItem(p_phase_arr_obj, 5, Py_BuildValue("d", p_arr->azi));
	PyTuple_SetItem(p_phase_arr_obj, 6, Py_BuildValue("d", p_arr->slo));

	PyList_Append(p_arrlistobj, p_phase_arr_obj);
	/* List Append increments the refcoqunt so we need to
	 * decrement our ref */
	Py_DECREF(p_phase_arr_obj);
      }
    }

    PyList_Append(p_evarrlistobj, p_arrlistobj);
    /* List Append increments the refcount so we need to decrement our ref */
    Py_DECREF(p_arrlistobj);
  }

  *pp_eventsobj = p_eventsobj;
  *pp_evarrlistobj = p_evarrlistobj;
}

Event_t * alloc_event_sig(SigModel_t * p_sigmodel)
{
  Event_t * p_event;
  int numsites;
  int numtimedefphases;

  p_event = (Event_t *) calloc(1, sizeof(*p_event));

  numsites = EarthModel_NumSites(p_sigmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);

  p_event->p_all_detids = NULL;
  p_event->p_num_dets = NULL;

  p_event->p_arrivals = (Arrival_t *) calloc(numsites * numtimedefphases,
                                       sizeof(*p_event->p_arrivals));
  for (int i=0; i < numsites; ++i) {
    for(int j=0; j < numtimedefphases; ++j) {
      (p_event->p_arrivals + i*numtimedefphases + j)->phase = j;
      (p_event->p_arrivals + i*numtimedefphases + j)->siteid = i+1;
    }
  }
  return p_event;
}

void free_event(Event_t * p_event)
{

  if (p_event->p_all_detids != NULL) {
    /* if a NETVISA event */
    free(p_event->p_all_detids);
    free(p_event->p_num_dets);
  }

  if (p_event->p_arrivals != NULL) {
    /* SIGVISA */
    free(p_event->p_arrivals);
  }
  free(p_event);
}

void copy_event_sig(SigModel_t * p_sigmodel, Event_t * p_tgt_event,
                const Event_t * p_src_event)
{
  Arrival_t * p_tgt_arrivals;
  int numsites;
  int numtimedefphases;

  numsites = EarthModel_NumSites(p_sigmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);

  /* save the arrivals pointer */
  p_tgt_arrivals = p_tgt_event->p_arrivals;

  /* copy the event */
  *p_tgt_event = *p_src_event;

  /* restore the arrivals pointer */
  p_tgt_event->p_arrivals = p_tgt_arrivals;

  /* copy the arrivals */
  memcpy(p_tgt_event->p_arrivals, p_src_event->p_arrivals,
         numsites * numtimedefphases * sizeof(*p_src_event->p_arrivals));
}

double Segment_EndTime(Segment_t * b) {
  return b->start_time + (b->len / b->hz);
}

int have_signal(SigModel_t * p_sigmodel, int site, double start_time, double end_time) {

  for (int i=0; i < p_sigmodel->numsegments; ++i) {

    Segment_t * p_seg = p_sigmodel->p_segments + i;

    double seg_start = p_seg->start_time;
    double seg_end = Segment_EndTime(p_seg);

    if (p_seg->siteid == site && seg_start <= end_time && seg_end >= start_time) {
      return 1;
    }

  }
  return 0;
}

PyObject * py_event_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

  LogInfo ("hello!");
  double time, lon, lat, depth, mb;
  if (!PyArg_ParseTuple(args, "ddddd", &time, &lon, &lat, &depth, &mb))
    return NULL;

  Event_t * p_event = ALLOC_EVENT(NULL, p_sigmodel);

    p_event->evtime = time;
    p_event->evlon = lon;
    p_event->evlat = lat;
    p_event->evdepth = depth;
    p_event->evmag = mb;

    LogInfo("optimizing...");
    optimize_arrivals(p_sigmodel, p_event, 0, NULL);
    LogInfo("scoring...");
    score_event_sig(p_sigmodel, p_event, 0, NULL);
    LogInfo("score %lf",p_event->evscore);
    double score = p_event->evscore;

    free(p_event);

    return Py_BuildValue("d", score);
}
