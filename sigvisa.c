#include <math.h>
#include <stdlib.h>

#define SIGVISA_MAIN_MODULE
#include "sigvisa.h"

static int py_sig_model_init(SigModel_t *self, PyObject *args);
static void py_sig_model_dealloc(SigModel_t * self);

PyObject * py_event_likelihood(SigModel_t * p_sigmodel, PyObject * args);


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
  {"trace_likelihood", (PyCFunction) py_trace_likelihood, METH_VARARGS,
   "trace_likelihood(trace, phaseids, params) -> ll\n" },
  {"segment_likelihood", (PyCFunction) py_segment_likelihood, METH_VARARGS,
   "segment_likelihood(segment, phaseids, params) -> ll\n" },
  {"generate_segment", (PyCFunction) py_gen_logenvelope_segment, METH_VARARGS,
   "generate_segment(start_time, end_time, siteid, srate, phaseids, params) -> segment\n" },
  {"sample_segment", (PyCFunction) py_sample_segment, METH_VARARGS,
   "sample_segment(start_time, end_time, siteid, srate, phaseids, params) -> segment\n" },
  {"get_signals", (PyCFunction)py_get_signals, METH_VARARGS,
   "get_signals() "
   "-> signals\n"},
  {"set_fake_detections", (PyCFunction)py_set_fake_detections, METH_VARARGS,
   "set_fake_detections(fake_detections) "
   "-> num_translated\n"},
  {"set_noise_process", (PyCFunction)py_set_noise_process, METH_VARARGS,
   "set_noise_process(siteid, band, chan, noise_mean, noise_variance, noise_coeffs)"
   "-> ??\n"},
  {"set_wiggle_process", (PyCFunction)py_set_wiggle_process, METH_VARARGS,
   "set_wiggle_process(siteid, band, noise_mean, noise_variance, noise_coeffs)"
   "-> ??\n"},
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

static PyMethodDef EarthModel_methods[] = {
    {"InRange", (PyCFunction)py_EarthModel_InRange, METH_VARARGS,
     "Inquire whether an event is within range of a site for a given phase\n"
     "InRange(lon, lat, depth, phaseid, siteid) -> True / False"
    },
    {"Delta", (PyCFunction)py_EarthModel_Delta, METH_VARARGS,
     "Compute the distance between an event location and a site\n"
     "Delta(lon, lat, siteid) -> distance in degrees"
    },
    {"ArrivalTime", (PyCFunction)py_EarthModel_ArrivalTime, METH_VARARGS,
     "Compute the arrival time of an event at a site\n"
     "ArrivalTime(evlon, evlat, evdepth, evtime, phaseid, siteid) -> arrtime"
    },
    {"ArrivalTimeCoord", (PyCFunction)py_EarthModel_ArrivalTime_Coord,
     METH_VARARGS,
     "Compute the arrival time of an event at a coordinate\n"
     "ArrivalTime(evlon, evlat, evdepth, evtime, phaseid, sitelon, sitelat,"
     "siteelev) -> arrtime"
    },
    {"ArrivalAzimuth", (PyCFunction)py_EarthModel_ArrivalAzimuth,
     METH_VARARGS,
     "Compute the arrival azimuth of an event at a site\n"
     "ArrivalAzimuth(evlon, evlat, siteid) -> seaz",
    },
    {"ArrivalIncidentAngle", (PyCFunction)py_EarthModel_ArrivalIncidentAngle,
     METH_VARARGS,
     "Compute the incident angle of an event at a site\n"
     "IncidentAngle(evlon, evlat, evdepth, int phaseid, siteid) -> iangle"
    },
    {"ArrivalSlowness", (PyCFunction)py_EarthModel_ArrivalSlowness,
     METH_VARARGS,
     "Compute the arrival slowness of an event at a site\n"
     "ArrivalSlowness(evlon, evlat, evdepth, int phaseid, siteid) -> slowness"
    },
    {"InvertDetection", (PyCFunction)py_EarthModel_InvertDetection,
     METH_VARARGS,
     "Invert a detection\n"
     "InvertDetection(siteid, azi, slo, time) -> (lon, lat, depth, time)",
    },
    {"IsTimeDefPhase", (PyCFunction)py_EarthModel_IsTimeDefPhase,
     METH_VARARGS,
     "Determine whether a phaseid is time-defining or not\n"
     "IsTimeDefPhase(phaseid) -> True or False",
    },
    {"NumPhases", (PyCFunction)py_EarthModel_NumPhases,
     METH_VARARGS, "NumPhases() -> number of phases",
    },
    {"NumTimeDefPhases", (PyCFunction)py_EarthModel_NumTimeDefPhases,
     METH_VARARGS, "NumTimeDefPhases() -> number of time-defining phases",
    },
    {"NumSites", (PyCFunction)py_EarthModel_NumSites,
     METH_VARARGS, "NumSites() -> number of sites",
    },
    {"DiffAzimuth", (PyCFunction)py_EarthModel_DiffAzimuth,
     METH_VARARGS,
     "Compute the angular difference between azi1 and azi2\n"
     " + ve implies azi2 is clockwise from azi1"
     "DiffAzimuth(azi1, azi2) -> azi2 - azi1",
    },
    {"PhaseName", (PyCFunction)py_EarthModel_PhaseName,
     METH_VARARGS, "PhaseName(phaseid) -> name of phase",
    },
    {"MaxTravelTime", (PyCFunction)py_EarthModel_MaxTravelTime,
     METH_VARARGS, "MaxTravelTime() -> time",
    },
    {"PhaseRange", (PyCFunction)py_EarthModel_PhaseRange,
     METH_VARARGS, "PhaseRange(phaseid) -> (min distance, max distance)",
    },
    {"QFVC", (PyCFunction)py_EarthModel_QFVC,
     METH_VARARGS, "QFVC(depth, distance) -> q-factor",
    },
    {"TravelTime", (PyCFunction)py_EarthModel_TravelTime, METH_VARARGS,
     "travel time for a give phase of an event a given depth and distance\n"
     "TravelTime(phaseid, depth, dist) -> travel time"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject py_EarthModel = {
    PyObject_HEAD_INIT(NULL)
    0,                                       /*ob_size*/
    "sigvisa.EarthModel",                    /*tp_name*/
    sizeof(EarthModel_t),                    /*tp_basicsize*/
    0,                                       /*tp_itemsize*/
    (destructor)py_EarthModel_UnInit,        /*tp_dealloc*/
    0,                                       /*tp_print*/
    0,                                       /*tp_getattr*/
    0,                                       /*tp_set<attr*/
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
    "Earth Model",                           /* tp_doc */
    0,                                       /* tp_traverse */
    0,                                       /* tp_clear */
    0,                                       /* tp_richcompare */
    0,                                       /* tp_weaklistoffset */
    0,                                       /* tp_iter */
    0,                                       /* tp_iternext */
    EarthModel_methods,                      /* tp_methods */
    0,                                       /* tp_members */
    0,                                       /* tp_getset */
    0,                                       /* tp_base */
    0,                                       /* tp_dict */
    0,                                       /* tp_descr_get */
    0,                                       /* tp_descr_set */
    0,                                       /* tp_dictoffset */
    (initproc)py_EarthModel_Init,            /* tp_init */
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
  init_signal_model(&(self->signal_model), (const char *)signal_model_name, numsites);
  self->log_trace_cb = log_trace_cb;
  Py_INCREF(log_trace_cb);

  return 0;
}



static PyMethodDef sigvisaMethods[] = {
  {"canonical_channel_num", (PyCFunction)py_canonical_channel_num, METH_VARARGS,
   "canonical_channel_num(chan_name) "
   "-> channel_num\n"},
  {"canonical_band_num", (PyCFunction)py_canonical_band_num, METH_VARARGS,
   "canonical_band_num(band_name) "
   "-> bandl_num\n"},
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

  py_EarthModel.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_EarthModel) < 0)
    return;


  m = Py_InitModule3("sigvisa", sigvisaMethods,
                     "Signal-Based Vertically Integrated Seismological Processing");
  import_array();/* Must be present for NumPy. Called first after above line.*/

  Py_INCREF(&py_sig_model);
  PyModule_AddObject(m, "SigModel", (PyObject *)&py_sig_model);

  Py_INCREF(&py_EarthModel);
  PyModule_AddObject(m, "EarthModel", (PyObject *)&py_EarthModel);

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
	  free_trace(channel->p_bands[band]);
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


void convert_tuple_int(PyObject * tuple,
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


/* return the score of an event, using optimal arrival times */
PyObject * py_event_score(SigModel_t * p_sigmodel, PyObject * args) {

  LogInfo ("hello!");
  double time, lon, lat, depth, mb;
  if (!PyArg_ParseTuple(args, "ddddd", &time, &lon, &lat, &depth, &mb))
    return NULL;

  Event_t * p_event = ALLOC_EVENT(p_sigmodel);

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
  PyObject * py_value = PyDict_GetItemString(py_dict, key); CHECK_ERROR;
  if (py_value != NULL) {
    *value = PyFloat_AsDouble(py_value);
    retcode = 1;
  }
  return retcode;
}

int pydict_get_int(PyObject * py_dict, char * key, long * value) {
  int retcode = 0;
  PyObject * py_value = PyDict_GetItemString(py_dict, key); CHECK_ERROR;
  if (py_value != NULL) {
    *value = PyInt_AsLong(py_value);
    retcode = 1;
  }
  return retcode;
}

int pydict_get_string(PyObject * py_dict, char * key, char ** value) {
  int retcode = 0;
  PyObject * py_value = PyDict_GetItemString(py_dict, key); CHECK_ERROR;
  if (py_value != NULL) {
    *value = PyString_AsString(py_value);
    retcode = 1;
  } else {
    *value = NULL;
  }
  return retcode;
}

PyObject * py_event_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

  LogInfo ("hello!");
  double time, lon, lat, depth, mb;
  if (!PyArg_ParseTuple(args, "ddddd", &time, &lon, &lat, &depth, &mb))
    return NULL;

  Event_t * p_event = ALLOC_EVENT(p_sigmodel);

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




/* Check to see if our sigmodel has a signal for a particular site at
   a particular time */ 
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
