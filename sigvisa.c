#include <math.h>
#include <stdlib.h>

#define NETVISA_MAIN_MODULE
#include "sigvisa.h"

static int py_sig_model_init(SigModel_t *self, PyObject *args);
static void py_sig_model_dealloc(SigModel_t * self);
static PyObject * py_set_signals(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_set_waves(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_get_signals(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_get_waves(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_set_fake_detections(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_synthesize_signals(SigModel_t *p_sigmodel, PyObject *args);
static PyObject * py_canonical_channel_num(SigModel_t * p_sigmodel, PyObject * args);
static PyObject * py_score_world(SigModel_t * p_sigmodel, PyObject * args);
PyObject * py_det_likelihood(SigModel_t * p_sigmodel, PyObject * args);

extern PyTypeObject py_EarthModel;

static PyMethodDef SigModel_methods[] = {
  {"set_signals", (PyCFunction)py_set_signals, METH_VARARGS,
   "set_signals(traces) "
   "-> num_translated\n"},
  {"set_waves", (PyCFunction)py_set_waves, METH_VARARGS,
   "set_waves(traces) "
   "-> num_translated\n"},
  {"synthesize_signals", (PyCFunction)py_synthesize_signals, METH_VARARGS,
   "synthesize_signals(evlist, stalist, start_time, end_time, hz) "
   "-> success\n"},
  {"get_signals", (PyCFunction)py_get_signals, METH_VARARGS,
   "get_signals() "
   "-> signals\n"},
  {"get_waves", (PyCFunction)py_get_waves, METH_VARARGS,
   "get_waves() "
   "-> signals\n"},
  {"set_fake_detections", (PyCFunction)py_set_fake_detections, METH_VARARGS,
   "set_fake_detections(fake_detections) "
   "-> num_translated\n"},
  {"canonical_channel_num", (PyCFunction)py_canonical_channel_num, METH_VARARGS,
   "canonical_channel_num(chan_name) "
   "-> channel_num\n"},
  {"score_world", (PyCFunction)py_score_world, METH_VARARGS,
   "score_world(events, arrtimes, verbose) "
   "-> log probability\n"},
  {"detection_likelihood", (PyCFunction)py_det_likelihood, METH_VARARGS,
   "detection_likelihood(env_height, env_decay, env_offset)"
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

  const char * numevent_fname;
  const char * evloc_fname;
  const char * evmag_fname;
  const char * arrtime_fname;
  const char * arrazi_fname;
  const char * arrslo_fname;
  const char * arramp_fname;
  const char * sig_fname;
  PyObject * log_trace_cb;

    if (!PyArg_ParseTuple(args, "OddissssssssO", &p_earth, &start_time, &end_time, &ar_perturbation, &numevent_fname, &evloc_fname, &evmag_fname, &arrtime_fname, &arrazi_fname, &arrslo_fname, &arramp_fname, &sig_fname, &log_trace_cb))
    return -1;
  
  if (end_time <= start_time)
  {
    PyErr_SetString(PyExc_ValueError, 
                    "SigModel: error: end_time <= start_time");
    return -1;
  }

  // TODO: check that siteid array is good

  self->start_time = start_time;
  self->end_time = end_time;

  self->p_earth = p_earth;
  Py_INCREF((PyObject *)self->p_earth);

  self->ar_perturbation = ar_perturbation;
  LogInfo("set ar perturb %d", self->ar_perturbation);
  //self->siteid = siteid;
  //Py_INCREF((PyObject *)self->siteid);

  NumEventPrior_Init_Params(&self->num_event_prior, numevent_fname);
  EventLocationPrior_Init_Params(&self->event_location_prior, evloc_fname);
  EventMagPrior_Init_Params(&self->event_mag_prior, 1, evmag_fname);
  ArrivalTimeJointPrior_Init_Params(&self->arr_time_joint_prior, arrtime_fname);
  ArrivalAzimuthPrior_Init_Params(&self->arr_az_prior, arrazi_fname);
  ArrivalSlownessPrior_Init_Params(&self->arr_slo_prior, arrslo_fname);
  ArrivalAmplitudePrior_Init_Params(&self->arr_amp_prior, arramp_fname);

  SignalPrior_Init_Params(&self->sig_prior, sig_fname, p_earth->numsites);
  fflush(stdout);

  self->log_trace_cb = log_trace_cb;
  Py_INCREF(log_trace_cb);


  return 0;
}

static PyObject * traceClass_obj;

static PyMethodDef sigvisaMethods[] = {
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
  
  srand(seed);

  Py_INCREF(Py_None);
  
  return Py_None;
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
  
  

  for (int i=0; i < self->numsegments; ++i) {
    for(int j=0; j < NUM_CHANS; ++j) {
      Signal_t *channel = self->p_segments[i].p_channels[j];
      if (channel != NULL) {
	if (channel->py_array != NULL) {
	  Py_DECREF(channel->py_array);
	}
	free(self->p_segments[i].p_channels[j]);
      }
    }
  }
  free(self->p_segments);

  for (int i=0; i < self->numsegments; ++i) {
    for(int j=0; j < NUM_CHANS; ++j)  {
      Signal_t *channel = self->p_wave_segments[i].p_channels[j];
      if (channel != NULL) {
	if (channel->py_array != NULL) {
	  Py_DECREF(channel->py_array);
	}
	free(self->p_wave_segments[i].p_channels[j]);
      }
    }
  }
  free(self->p_wave_segments);


  EventLocationPrior_UnInit(&self->event_location_prior);
  ArrivalTimeJointPrior_UnInit(&self->arr_time_joint_prior);
  SignalPrior_UnInit(&self->sig_prior);

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


static PyObject * py_score_world(SigModel_t * p_sigmodel, PyObject * args)
{
  /* input arguments */
  PyArrayObject * p_events_arrobj;
  PyArrayObject * p_arrtimes_arrobj;
 
  int numevents;
  Event_t * p_events;
  double score;
  
  if (!PyArg_ParseTuple(args, "O!O!",
                        &PyArray_Type, &p_events_arrobj, 
                        &PyArray_Type, &p_arrtimes_arrobj)
      || !p_events_arrobj || !p_arrtimes_arrobj)
    return NULL;

  if ((2 != p_events_arrobj->nd) || (NPY_DOUBLE 
                                     != p_events_arrobj->descr->type_num)
      || (EV_NUM_COLS != p_events_arrobj->dimensions[1]))
  {
    PyErr_SetString(PyExc_ValueError,
                    "score_world: wrong shape or type of events array");
    return NULL;
  }

  convert_eventobj_no_det(p_events_arrobj, &numevents, &p_events);
  
  PyArrayObject * p_arrtimes = convert_arrtimes(p_sigmodel, (PyObject *)p_arrtimes_arrobj);

  int verbose = 1;
  
  // TODO: make world scoring work
  //score = score_world_sig(p_sigmodel, numevents, p_events, p_arrtimes, verbose);
  score = -1;

  free_events(numevents, p_events);
  
  return Py_BuildValue("d", score);
}


/* ==== Make a Python Array Obj. from a PyObject, ================
       generates a double vector w/ contiguous memory which may be a new allocation if
       the original was not a double type or contiguous 
    !! Must DECREF the object returned from this routine unless it is returned to the
       caller of this routines caller using return PyArray_Return(obj) or
       PyArray_BuildValue with the "N" construct   !!!
  */
  PyArrayObject *pyvector(PyObject *objin)  {
      return (PyArrayObject *) PyArray_ContiguousFromAny(objin,
          NPY_DOUBLE, 1,2);
  }

char * signal_str(Signal_t * signal) {
  char * str = malloc(200*sizeof(char));
  snprintf(str, 200, "Signal: at station %d, sampling rate %f, samples %ld, start time %f, end time %f\n", 
	   signal->siteid, 
	   signal->hz, 
	   signal->len, 
	   signal->start_time, 
	   signal->start_time + (signal->len)/(signal->hz));
  return str;
}

int print_signal(Signal_t * signal) {
  char * s = signal_str(signal);
  int result = fputs(s, stdout);
  free(s);
  return result;
}


PyObject* canonical_channel_name(int num) {
   switch (num) {
   case CHAN_BHZ:
     return PyString_FromString("BHZ");
   case CHAN_BHE:
     return PyString_FromString("BHE");
   case CHAN_BHN:
     return PyString_FromString("BHN");
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
   } else {
     result = CHAN_OTHER;
   } 
   return result;
 }

static PyObject * py_canonical_channel_num(SigModel_t * p_sigmodel, PyObject * args) {
  PyObject * py_chan_str;
  if (!PyArg_ParseTuple(args, "O!", &PyString_Type, &py_chan_str))
    return NULL;

   int result = canonical_channel_num(PyString_AsString(py_chan_str));
   return Py_BuildValue("n", result);
 }

PyObject * py_det_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

  double env_height, env_decay, env_onset;

  if (!PyArg_ParseTuple(args, "ddd", &env_height, &env_decay, &env_onset))
    return NULL;

  double ll = det_likelihood((void *)p_sigmodel, env_height, env_decay, env_onset);
  return Py_BuildValue("d", ll);
}

int signal_to_trace(Signal_t * p_signal, PyObject ** pp_trace) {

  // a trace object contains two members: data, a numpy array, and stats, a python dict.
   npy_intp dims[1];
   dims[0] = p_signal->len;

   PyObject * py_data;
   if (p_signal->py_array == NULL) {
     py_data = (PyObject *)PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, p_signal->p_data);
     p_signal->py_array = (PyArrayObject *) py_data;
   } else {
     py_data = (PyObject *)p_signal->py_array;
   }

   PyObject * py_stats = PyDict_New(); 

   PyObject * key = PyString_FromString("npts_processed");
  PyObject * value = Py_BuildValue("n", p_signal->len);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);

  key = PyString_FromString("starttime_unix");
  value = Py_BuildValue("d", p_signal->start_time);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);

  key = PyString_FromString("window_size");
  value = Py_BuildValue("d", 1.0 / p_signal->hz);
  assert(key != NULL && value != NULL);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);
  key = PyString_FromString("overlap");
  assert(key != NULL && value != NULL);
  value = Py_BuildValue("d", 0);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);
  key = PyString_FromString("sampling_rate");
  value = Py_BuildValue("d", p_signal->hz);
  assert(key != NULL && value != NULL);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);

  key = PyString_FromString("siteid");
  value = Py_BuildValue("n", p_signal->siteid);
  assert(key != NULL && value != NULL);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);
  
  key = PyString_FromString("chanid");
  value = Py_BuildValue("n", 0);
  assert(key != NULL && value != NULL);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);

  key = PyString_FromString("channel");
  value = canonical_channel_name(p_signal->chan);
  assert(key != NULL && value != NULL);
  PyDict_SetItem(py_stats, key, value);
  Py_DECREF(key);
  Py_DECREF(value);

  PyObject * args = Py_BuildValue("OO", py_data, py_stats);
  PyObject * trace =   PyEval_CallObject(traceClass_obj, args);
 
  *pp_trace = trace;

  return 0;
}


int save_pdf_plot(SigModel_t * p_sigmodel, Signal_t * p_signal, char * filename) {

  PyObject * p_trace;

  signal_to_trace(p_signal, &p_trace);
  PyObject * py_text = py_text = Py_BuildValue("s", filename);

  int retval = PyObject_CallFunction(p_sigmodel->log_trace_cb, "OO", 
				   p_trace,
				   py_text);
  return retval;
}


 int trace_to_signal(PyObject * p_trace, Signal_t * p_signal) {

  // a trace object contains two members: data, a numpy array, and stats, a python dict.
  PyObject * py_data =  PyObject_GetAttrString(p_trace, "data");
  if (py_data == NULL) {
    LogFatal("error: py_data is null!\n");
    exit(1);
  }

  

  /*  fprintf(stdout, "py_data is an array with depth %d and dimension %d x %d\n", 
	  ((PyArrayObject *)py_data)->nd,
	  ((PyArrayObject *)py_data)->dimensions[0],  
	  ((PyArrayObject *)py_data)->dimensions[1]); */

  p_signal->py_array = pyvector(py_data);
  Py_INCREF(p_signal->py_array);
  Py_DECREF(py_data);
  CHECK_ERROR

  p_signal->p_data = (double *) p_signal->py_array->data;  
  p_signal->len = PyArray_SIZE(p_signal->py_array);
  
  PyObject * stats = PyObject_GetAttrString(p_trace, "stats");

  PyObject * key = PyString_FromString("starttime_unix");
  PyObject * py_start_time = PyDict_GetItem(stats, key);
  Py_DECREF(key);
  p_signal->start_time = PyFloat_AsDouble(py_start_time);

  
  key = PyString_FromString("window_size");
  PyObject * py_window_size = PyDict_GetItem(stats, key);
  Py_DECREF(key);
  if (py_window_size != NULL) {
    double window_size = PyFloat_AsDouble(py_window_size);
    key = PyString_FromString("overlap");
    PyObject * py_overlap = PyDict_GetItem(stats, key);
    Py_DECREF(key);
    double overlap = PyFloat_AsDouble(py_overlap);
    p_signal->hz = 1.0 / (window_size * (1 - overlap));
  } else {
    key = PyString_FromString("sampling_rate");
    PyObject * py_sampling_rate = PyDict_GetItem(stats, key);
    Py_DECREF(key);
    double sampling_rate = PyFloat_AsDouble(py_sampling_rate);
    p_signal->hz = sampling_rate;
  }

  key = PyString_FromString("siteid");
  PyObject * py_siteid = PyDict_GetItem(stats, key);
  Py_DECREF(key);
  p_signal->siteid = (int)PyInt_AsLong(py_siteid);

  key = PyString_FromString("chanid");
  PyObject * py_chanid = PyDict_GetItem(stats, key);
  Py_DECREF(key);
  p_signal->chanid = (int)PyInt_AsLong(py_chanid);
  CHECK_ERROR

  key = PyString_FromString("channel");
  PyObject * py_chan = PyDict_GetItem(stats, key);
  Py_DECREF(key);
  char* chan_str = PyString_AsString(py_chan);
  p_signal->chan = canonical_channel_num(chan_str);
  
  /*fprintf(stdout, "Converted ");
    print_signal(p_signal);*/
  return 0;
}

int trace_bundle_to_channel_bundle(PyObject * trace_bundle, ChannelBundle_t * pp_channel_bundle) {
  if(!PyList_Check(trace_bundle)) {
    LogFatal("trace_bundle_to_signal_bundle: expected Python list!\n");
    exit(1);
  }
  int n = PyList_Size(trace_bundle);

  Signal_t * temp;

  pp_channel_bundle->len = -1;
  pp_channel_bundle->start_time = -1;
  pp_channel_bundle->hz = -1;
  pp_channel_bundle->siteid = -1;

  int i;
  for (i=0; i < n; ++i) {
    PyObject * p_trace = PyList_GetItem(trace_bundle, i);
    temp = malloc(sizeof(Signal_t));
    trace_to_signal(p_trace, temp);
    pp_channel_bundle->p_channels[temp->chan] = temp;

    /* ensure that all channels within a bundle have the same attributes */
    UPDATE_AND_VERIFY(&pp_channel_bundle->len, temp->len);
    UPDATE_AND_VERIFY(&pp_channel_bundle->start_time, temp->start_time);
    UPDATE_AND_VERIFY(&pp_channel_bundle->hz, temp->hz);
    UPDATE_AND_VERIFY(&pp_channel_bundle->siteid, temp->siteid);
  }
  
  return i;
}

int trace_bundles_to_channel_bundles(PyObject * trace_bundle_list, ChannelBundle_t ** pp_channel_bundles) {
  
  if(!PyList_Check(trace_bundle_list)) {
    LogFatal("trace_bundles_to_signal_bundles: expected Python list!\n");
    exit(1);
  }

  int n = PyList_Size(trace_bundle_list);
  (*pp_channel_bundles) = calloc(n, sizeof(ChannelBundle_t));

  int i;
  for (i=0; i < n; ++i) {
    PyObject * p_trace_bundle = PyList_GetItem(trace_bundle_list, i);
    trace_bundle_to_channel_bundle(p_trace_bundle, (*pp_channel_bundles) + i);
  }

  return i;
}


static PyObject * py_set_signals(SigModel_t *p_sigmodel, PyObject *args) {
  PyObject * p_tracelist_obj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_tracelist_obj))
    return NULL;

  int n = trace_bundles_to_channel_bundles(p_tracelist_obj, &p_sigmodel->p_segments);
  p_sigmodel->numsegments = n;
  
  return Py_BuildValue("n", n);
}

static PyObject * py_set_waves(SigModel_t *p_sigmodel, PyObject *args) {
  PyObject * p_tracelist_obj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_tracelist_obj))
    return NULL;

  int n = trace_bundles_to_channel_bundles(p_tracelist_obj, &p_sigmodel->p_wave_segments);
  p_sigmodel->numsegments = n;
  
  return Py_BuildValue("n", n);
}

PyObject * channel_bundle_to_trace_bundle(ChannelBundle_t * p_segment) {

  PyObject * chan_traces = PyList_New(0);
  for(int j=0; j < NUM_CHANS; ++j) {
    Signal_t *p_signal = (p_segment)->p_channels[j];
    if (p_signal != NULL) {
      PyObject * py_trace;
      signal_to_trace(p_signal, &py_trace);
      PyList_Append(chan_traces, py_trace);
    }
  }
  return chan_traces;
}

static PyObject * py_get_waves(SigModel_t *p_sigmodel, PyObject *args) {
  //PyObject * p_tracelist_obj;
  //  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_tracelist_obj))
  //  return NULL;

  int numsegments = p_sigmodel->numsegments;

  PyObject * channels = PyList_New(0);
  for (int i=0; i < numsegments; ++i) {
    PyObject * chan_traces = channel_bundle_to_trace_bundle(p_sigmodel->p_wave_segments + i);
    PyList_Append(channels, chan_traces);
  }
  
  return channels;
}

static PyObject * py_get_signals(SigModel_t *p_sigmodel, PyObject *args) {
  //PyObject * p_tracelist_obj;
  //  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_tracelist_obj))
  //  return NULL;

  int numsegments = p_sigmodel->numsegments;

  PyObject * channels = PyList_New(0);
  for (int i=0; i < numsegments; ++i) {
    PyObject * chan_traces = channel_bundle_to_trace_bundle(p_sigmodel->p_segments + i);
    PyList_Append(channels, chan_traces);
  }
  
  return channels;
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
      exit(1);
    }

    (*pp_detections)[i].arid_det = (int)PyInt_AsLong(PyTuple_GetItem(p_fakedet, 0));
    (*pp_detections)[i].site_det = (int)PyInt_AsLong(PyTuple_GetItem(p_fakedet, 1));
    (*pp_detections)[i].time_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 2));
    (*pp_detections)[i].amp_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 3));
    (*pp_detections)[i].azi_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 4));
    (*pp_detections)[i].slo_det = (double)PyFloat_AsDouble(PyTuple_GetItem(p_fakedet, 5));
    (*pp_detections)[i].phase_det = (int)PyInt_AsLong(PyTuple_GetItem(p_fakedet, 6));

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


void synthesize_signals(SigModel_t *p_sigmodel, int numevents, Event_t ** pp_events, int numsiteids, int * p_siteids, double start_time, double end_time, double hz) {

  EarthModel_t * p_earth = p_sigmodel->p_earth;

  int numsites =  EarthModel_NumSites(p_earth);
  int numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  for (int i=0; i < numevents; ++i) {
    Event_t * p_event = pp_events[i];

    p_event->p_arrivals = calloc(numsites*numtimedefphases, sizeof(Arrival_t));

    for (int j=0; j < numsiteids; ++j) {
      int site = p_siteids[j];

      for (int phase=0; phase < numtimedefphases; ++phase) {
	Arrival_t * arr = (p_event->p_arrivals + (site-1)*numtimedefphases+phase);

	// for each event, fill in arrivals at each station we care about    
	
	arr->time = EarthModel_ArrivalTime(p_earth, 
					   p_event->evlon, 
					   p_event->evlat, 
					   p_event->evdepth, 
					   p_event->evtime, 
					   phase, site-1);
	
	arr->amp = ArrivalAmplitudePrior_Point(&p_sigmodel->arr_amp_prior, 
					       p_event->evmag, 
					       p_event->evdepth, 
					       arr->time - p_event->evtime, 
					       site-1, phase);
	
	arr->azi = EarthModel_ArrivalAzimuth(p_earth, 
					     p_event->evlon, 
					     p_event->evlat, 
					     site-1);
	arr->slo = EarthModel_ArrivalSlowness(p_earth, 
					      p_event->evlon, 
					      p_event->evlat, 
					      p_event->evdepth, 
					      phase, site-1); 
      }
      
      char * estr = event_str(p_event);
      LogTrace("arrival info for event %s at site %d\n", estr, site);
      free(estr);
    }
  }

  p_sigmodel->numsegments = numsiteids;
  p_sigmodel->p_segments = calloc(numsiteids, sizeof(ChannelBundle_t));
  p_sigmodel->p_wave_segments = calloc(numsiteids, sizeof(ChannelBundle_t));
  
  for (int i=0; i < numsiteids; ++i) {
    int siteid = p_siteids[i];
    
    ChannelBundle_t * p_segment = p_sigmodel->p_segments + i;
    ChannelBundle_t * p_wave_segment = p_sigmodel->p_wave_segments + i;

    p_segment->start_time = start_time;
    p_segment->hz = hz;
    p_segment->siteid = siteid;
    p_segment->len = (int) (end_time - start_time) * hz;

    p_wave_segment->start_time = start_time;
    p_wave_segment->hz = hz;
    p_wave_segment->siteid = siteid;
    p_wave_segment->len = (int) (end_time - start_time) * hz;

    SignalPrior_ThreeAxisEnvelope(&p_sigmodel->sig_prior,
				  p_sigmodel->p_earth,
				  numevents,
				  (const Event_t **)pp_events,
				  p_segment,
				  p_wave_segment);
    LogTrace("generated segment at siteid %d w/ length %ld = (%lf - %lf) * %lf\n", siteid, p_segment->len, end_time, start_time, hz);

    LogTrace("scoring event...\n");
    score_event_sig(p_sigmodel, pp_events[0], 0, NULL);
  }

  for (int i=0; i < numevents; ++i) {
    Event_t * p_event = pp_events[i];
    free(p_event->p_arrivals);
  }

}

Signal_t * alloc_signal(ChannelBundle_t * p_segment) {
  Signal_t * p_signal = (Signal_t *) malloc(sizeof(Signal_t));
  p_signal->start_time = p_segment->start_time;
  p_signal->len = p_segment->len;
  p_signal->hz = p_segment->hz;
  p_signal->siteid = p_segment->siteid;
  p_signal->py_array = NULL;
  return p_signal;
}

static PyObject * py_synthesize_signals(SigModel_t *p_sigmodel, PyObject *args) {
  PyArrayObject * p_evlist_obj;
  PyObject * p_stalist_obj;
  double stime, etime, hz;
  if (!PyArg_ParseTuple(args, "O!O!ddd", &PyArray_Type, &p_evlist_obj, &PyTuple_Type, &p_stalist_obj, &stime, &etime, &hz))
    return NULL;

  int numevents;
  Event_t * p_events;
  convert_eventobj_no_det(p_evlist_obj, &numevents, &p_events);
  Event_t ** pp_events = indirect_event_list(numevents, p_events);

  int numsiteids;
  int * p_siteids;
  
  convert_tuple_int(p_stalist_obj, &numsiteids, &p_siteids);

  synthesize_signals(p_sigmodel, numevents, pp_events, numsiteids, p_siteids, stime, etime, hz);

  free(p_siteids);
  free(pp_events);
  free(p_events);
  
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

char * arrival_str(const Arrival_t * p_arr) {
  char * str = malloc(100*sizeof(char));
  snprintf(str, 100, "time %.4lf amp %.4lf azi %.4lf slo %.4lf",
         p_arr->time, p_arr->amp, p_arr->azi, p_arr->slo);
  return str;
}

void print_arrival(const Arrival_t * p_arr) {
  char *s = arrival_str(p_arr);
  puts(s);
  free(s);
}

char * event_str(const Event_t * p_event) {
  char * str = malloc(100*sizeof(char));
  snprintf(str, 100, 
	   "%4.1f E %4.1f N %.0f km %.0f s %.1f mb score %.1f orid %d",
	   p_event->evlon, p_event->evlat, p_event->evdepth,
	   p_event->evtime, p_event->evmag, p_event->evscore,
	   p_event->orid);
  return str;
}

void print_event(const Event_t * p_event)
{
  char *s = event_str(p_event);
  puts(s);
  free(s);
}


double ChannelBundle_EndTime(ChannelBundle_t * b) {
  return b->start_time + (b->len / b->hz);
}

int have_signal(SigModel_t * p_sigmodel, int site, double start_time, double end_time) {
  
  for (int i=0; i < p_sigmodel->numsegments; ++i) {
    
    ChannelBundle_t * p_seg = p_sigmodel->p_segments + i;

    double seg_start = p_seg->start_time;
    double seg_end = ChannelBundle_EndTime(p_seg);

    if (p_seg->siteid == site && seg_start <= end_time && seg_end >= start_time) {
      return 1;
    }
    
  }
  return 0;
}


