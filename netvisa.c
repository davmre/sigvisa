#include <math.h>
#include <stdlib.h>

#define NETVISA_MAIN_MODULE
#include "netvisa.h"

static int py_net_model_init(NetModel_t *self, PyObject *args);
static void py_net_model_dealloc(NetModel_t * self);
static PyObject * py_score_world(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_score_event(NetModel_t * p_netmodel, PyObject * args);

static PyMethodDef NetModel_methods[] = {
  {"score_world", (PyCFunction)py_score_world, METH_VARARGS,
   "score_world(events, ev_detlist, verbose) "
   "-> log probability\n"},
  {"score_event", (PyCFunction)py_score_event, METH_VARARGS,
   "score_event(event, detlist) "
   "-> log probability\n"},
  {NULL}  /* Sentinel */
};

static PyTypeObject py_net_model = {
    PyObject_HEAD_INIT(NULL)
    0,                                       /*ob_size*/
    "netvisa.NetModel",                      /*tp_name*/
    sizeof(NetModel_t),                      /*tp_basicsize*/
    0,                                       /*tp_itemsize*/
    (destructor)py_net_model_dealloc,        /*tp_dealloc*/
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
    "Network Processing Model",              /* tp_doc */
    0,                                       /* tp_traverse */
    0,                                       /* tp_clear */
    0,                                       /* tp_richcompare */
    0,                                       /* tp_weaklistoffset */
    0,                                       /* tp_iter */
    0,                                       /* tp_iternext */
    NetModel_methods,                        /* tp_methods */
    0,                                       /* tp_members */
    0,                                       /* tp_getset */
    0,                                       /* tp_base */
    0,                                       /* tp_dict */
    0,                                       /* tp_descr_get */
    0,                                       /* tp_descr_set */
    0,                                       /* tp_dictoffset */
    (initproc)py_net_model_init,             /* tp_init */
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
    {"ArrivalAzimuth", (PyCFunction)py_EarthModel_ArrivalAzimuth,
     METH_VARARGS,
     "Compute the arrival azimuth of an event at a site\n"
     "ArrivalAzimuth(evlon, evlat, siteid) -> (esaz, seaz)",
    },
    {"ArrivalSlowness", (PyCFunction)py_EarthModel_ArrivalSlowness,
     METH_VARARGS,
     "Compute the arrival slowness of an event at a site\n"
     "ArrivalSlowness(evlon, evlat, evdepth, int phaseid, siteid) -> slowness"
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
    {NULL}  /* Sentinel */
};

static PyTypeObject py_EarthModel = {
    PyObject_HEAD_INIT(NULL)
    0,                                       /*ob_size*/
    "netvisa.EarthModel",                    /*tp_name*/
    sizeof(EarthModel_t),                    /*tp_basicsize*/
    0,                                       /*tp_itemsize*/
    (destructor)py_EarthModel_UnInit,        /*tp_dealloc*/
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

static PyMethodDef netvisaMethods[] = {
/*
  {"score_world", py_score_world, METH_VARARGS,
   "score_world(net_model, events, evlist, verbose) "
   "-> log probability\n"},
*/
  {NULL, NULL}
};

void initnetvisa()
{
  PyObject * m;
  
  py_net_model.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_net_model) < 0)
    return;

  py_EarthModel.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_EarthModel) < 0)
    return;

  m = Py_InitModule3("netvisa", netvisaMethods,
                     "Network Vertically Integrated Seismological Processing");
  import_array();/* Must be present for NumPy. Called first after above line.*/

  Py_INCREF(&py_net_model);
  PyModule_AddObject(m, "NetModel", (PyObject *)&py_net_model);

  Py_INCREF(&py_EarthModel);
  PyModule_AddObject(m, "EarthModel", (PyObject *)&py_EarthModel);
}

static void alloc_detections(PyArrayObject * detectionsobj, 
                             int * p_ndetections,
                             Detection_t ** p_p_detections)
{
  int ndetections;
  Detection_t * p_detections;
  int i;
  
  ndetections = detectionsobj->dimensions[0];

  p_detections = (Detection_t *)calloc(ndetections, sizeof(*p_detections));

  for(i=0; i<ndetections; i++)
  {
    p_detections[i].site_det = ARRAY2(detectionsobj, i, DET_SITE_COL);
    p_detections[i].arid_det = ARRAY2(detectionsobj, i, DET_ARID_COL);
    p_detections[i].time_det = ARRAY2(detectionsobj, i, DET_TIME_COL);
    p_detections[i].deltim_det = ARRAY2(detectionsobj, i, DET_DELTIM_COL);
    p_detections[i].azi_det = ARRAY2(detectionsobj, i, DET_AZI_COL);
    p_detections[i].delaz_det = ARRAY2(detectionsobj, i, DET_DELAZ_COL);
    p_detections[i].slo_det = ARRAY2(detectionsobj, i, DET_SLO_COL);
    p_detections[i].delslo_det = ARRAY2(detectionsobj, i, DET_DELSLO_COL);
    p_detections[i].snr_det = ARRAY2(detectionsobj, i, DET_SNR_COL);
    p_detections[i].phase_det = ARRAY2(detectionsobj, i, DET_PHASE_COL);
    p_detections[i].amp_det = ARRAY2(detectionsobj, i, DET_AMP_COL);
    p_detections[i].per_det = ARRAY2(detectionsobj, i, DET_PER_COL);
  }
  
  *p_ndetections = ndetections;
  *p_p_detections = p_detections;
}

static void free_detections(int ndetections, Detection_t * p_detections)
{
  free(p_detections);
}

static void alloc_site_up(PyArrayObject * siteupobj, int * p_nsites,
                          int * p_ntime, int ** p_p_site_up)
{
  int nsites;
  int ntime;
  
  int * p_site_up;
  int i;
  int j;
  
  nsites = siteupobj->dimensions[0];
  ntime = siteupobj->dimensions[1];

  p_site_up = (int *)calloc(nsites * ntime, sizeof(*p_site_up));

  for (i=0; i<nsites; i++)
    for (j=0; j<ntime; j++)
      p_site_up[i * ntime + j] = BOOLARRAY2(siteupobj, i, j);
  
  *p_nsites = nsites;
  *p_ntime = ntime;
  *p_p_site_up = p_site_up;
}

static void free_site_up(int nsites, int ntime, int * p_site_up)
{
  free(p_site_up);
}


static int py_net_model_init(NetModel_t *self, PyObject *args)
{
  EarthModel_t * p_earth;
  double start_time;
  double end_time;
  PyArrayObject * detectionsobj;
  PyArrayObject * siteupobj;
  const char * numevent_fname;
  const char * evloc_fname;
  const char * evmag_fname;
  const char * evdet_fname;
  const char * arrtime_fname;
  const char * numfalse_fname;
  const char * arraz_fname;
  const char * arrslo_fname;
  const char * arrphase_fname;
  
  if (!PyArg_ParseTuple(args, "O!ddO!O!sssssssss", &py_EarthModel, &p_earth,
                        &start_time, &end_time, 
                        &PyArray_Type, &detectionsobj,
                        &PyArray_Type, &siteupobj,
                        &numevent_fname, &evloc_fname, &evmag_fname, 
                        &evdet_fname, &arrtime_fname, &numfalse_fname,
                        &arraz_fname, &arrslo_fname, &arrphase_fname)
      || !detectionsobj || !siteupobj)
    return -1;
  
  if (end_time <= start_time)
  {
    PyErr_SetString(PyExc_ValueError, 
                    "NetModel: error: end_time <= start_time");
    return -1;
  }
  
  if ((2 != detectionsobj->nd) || (NPY_DOUBLE !=detectionsobj->descr->type_num)
      || (DET_NUM_COLS != detectionsobj->dimensions[1]))
  {
    PyErr_SetString(PyExc_ValueError, "net_model_init: incorrect shape or type"
                    " of detections array");
    return -1;
  }

  if ((2 != siteupobj->nd) || (NPY_BOOL != siteupobj->descr->type_num)
      || (((int)floor((end_time - start_time) / UPTIME_QUANT))
          != siteupobj->dimensions[1]))
  {
    PyErr_SetString(PyExc_ValueError, "net_model_init: incorrect shape or type"
                    " of site_up array");
    return -1;
  }

  self->start_time = start_time;
  self->end_time = end_time;

  self->p_earth = p_earth;
  Py_INCREF((PyObject *)self->p_earth);

  alloc_detections(detectionsobj, &self->numdetections, &self->p_detections);

  alloc_site_up(siteupobj, &self->numsites, &self->numtime, &self->p_site_up);
  
  NumEventPrior_Init_Params(&self->num_event_prior, numevent_fname);
  
  EventLocationPrior_Init_Params(&self->event_location_prior, evloc_fname);
  
  EventMagPrior_Init_Params(&self->event_mag_prior, 1, evmag_fname);

  EventDetectionPrior_Init_Params(&self->event_det_prior, evdet_fname);

  ArrivalTimePrior_Init_Params(&self->arr_time_prior, arrtime_fname);
  
  NumFalseDetPrior_Init_Params(&self->num_falsedet_prior, numfalse_fname);
 
  ArrivalAzimuthPrior_Init_Params(&self->arr_az_prior, arraz_fname);

  ArrivalSlownessPrior_Init_Params(&self->arr_slo_prior, arrslo_fname);

  ArrivalPhasePrior_Init_Params(&self->arr_phase_prior, arrphase_fname);
  
  return 0;
}

static void py_net_model_dealloc(NetModel_t * self)
{
  if (self->p_earth)
  {
    Py_DECREF((PyObject *)self->p_earth);
    self->p_earth = NULL;
  }
  
  free_detections(self->numdetections, self->p_detections);
  self->p_detections = NULL;

  free_site_up(self->numsites, self->numtime, self->p_site_up);
  self->p_site_up = NULL;
  
  EventLocationPrior_UnInit(&self->event_location_prior);

  ArrivalTimePrior_UnInit(&self->arr_time_prior);

  NumFalseDetPrior_UnInit(&self->num_falsedet_prior);

  ArrivalAzimuthPrior_UnInit(&self->arr_az_prior);

  ArrivalSlownessPrior_UnInit(&self->arr_slo_prior);

  ArrivalPhasePrior_UnInit(&self->arr_phase_prior);
  
  self->ob_type->tp_free((PyObject*)self);
}

static void convert_event_detections(Event_t * p_event, int numsites,
                                     int numtimedefphases, int numdetections,
                                     Detection_t * p_detections,
                                     PyObject * phasedet_list)
{
  Py_ssize_t j;

  p_event->p_detids = (int *)malloc(numsites * numtimedefphases *
                                    sizeof(*p_event->p_detids));

  /* initialize all detections to -1, i.e. no detection */
  for (j=0; j<numsites * numtimedefphases; j++)
    p_event->p_detids[j] = -1;


  assert(phasedet_list);
  assert(PyList_Check(phasedet_list));
    
  for (j = 0; j < PyList_GET_SIZE(phasedet_list); j ++)
  {
    PyObject * phasedet_tuple;
    PyObject * phaseobj;
    PyObject * detobj;
    long phaseid;
    long detid;
    int siteid;
      
    phasedet_tuple = PyList_GetItem(phasedet_list, j);
    assert(phasedet_tuple);
    assert(PyTuple_Check(phasedet_tuple));

    phaseobj = PyTuple_GetItem(phasedet_tuple, 0);
    assert(phaseobj && PyInt_Check(phaseobj));
      
    detobj = PyTuple_GetItem(phasedet_tuple, 1);
    assert(detobj && PyInt_Check(detobj));
      
    phaseid = PyInt_AS_LONG(phaseobj);
    detid = PyInt_AS_LONG(detobj);

    assert((detid >= 0) && (detid < numdetections));
    siteid = p_detections[detid].site_det;

    p_event->p_detids[siteid * numtimedefphases + phaseid] = detid;
  }
}

static void convert_eventobj(PyArrayObject * p_events_arrobj, 
                             PyObject * p_evlist_obj,
                             int numsites, int numtimedefphases,
                             int numdetections,
                             Detection_t * p_detections,
                             int * p_numevents, Event_t ** p_p_events)
{
  Py_ssize_t numevents;
  Event_t * p_events;
  Py_ssize_t i;
  
  assert(PyList_Check(p_evlist_obj));
  
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

    convert_event_detections(p_event, numsites, numtimedefphases,
                             numdetections, p_detections,
                             PyList_GetItem(p_evlist_obj, i));
  }
  
  *p_numevents = numevents;
  *p_p_events = p_events;
}

static void free_events(int numevents, Event_t * p_events)
{
  int i;
  for (i=0; i<numevents; i++)
    free(p_events[i].p_detids);
  
  free(p_events);
}

static PyObject * py_score_world(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  PyArrayObject * p_events_arrobj;
  PyObject * p_evlist_obj;
  int verbose;
 
  int numevents;
  Event_t * p_events;
  double score;
  
  if (!PyArg_ParseTuple(args, "O!O!i",
                        &PyArray_Type, &p_events_arrobj, 
                        &PyList_Type, &p_evlist_obj, &verbose)
      || !p_events_arrobj || !p_evlist_obj)
    return NULL;

  if ((2 != p_events_arrobj->nd) || (NPY_DOUBLE 
                                     != p_events_arrobj->descr->type_num)
      || (EV_NUM_COLS != p_events_arrobj->dimensions[1]))
  {
    PyErr_SetString(PyExc_ValueError,
                    "score_world: wrong shape or type of events array");
    return NULL;
  }

  convert_eventobj(p_events_arrobj, p_evlist_obj,
                   EarthModel_NumSites(p_netmodel->p_earth),
                   EarthModel_NumTimeDefPhases(p_netmodel->p_earth),
                   p_netmodel->numdetections, p_netmodel->p_detections,
                   &numevents, &p_events);
  
  score = score_world(p_netmodel, numevents, p_events, verbose);
  
  free_events(numevents, p_events);
  
  return Py_BuildValue("d", score);
}

static PyObject * py_score_event(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  PyArrayObject * p_event_arrobj;
  PyObject * p_detlist_obj;
 
  Event_t * p_event;
  double numsc, locsc, magsc, detsc, dettimesc, detazsc, detslosc, detphasesc;
  double score;
  int possdetcnt, detcnt;
  
  if (!PyArg_ParseTuple(args, "O!O!",
                        &PyArray_Type, &p_event_arrobj, 
                        &PyList_Type, &p_detlist_obj)
      || !p_event_arrobj || !p_detlist_obj)
    return NULL;
  
  if ((1 != p_event_arrobj->nd) || (NPY_DOUBLE 
                                     != p_event_arrobj->descr->type_num)
      || (EV_NUM_COLS != p_event_arrobj->dimensions[0]))
  {
    PyErr_SetString(PyExc_ValueError,
                    "score_event: wrong shape or type of event array");
    return NULL;
  }
  
  p_event = (Event_t *)calloc(1, sizeof(*p_event));
  p_event->evlon = ARRAY1(p_event_arrobj, EV_LON_COL);
  p_event->evlat = ARRAY1(p_event_arrobj, EV_LAT_COL);
  p_event->evdepth = ARRAY1(p_event_arrobj, EV_DEPTH_COL);
  p_event->evtime = ARRAY1(p_event_arrobj, EV_TIME_COL);
  p_event->evmag = ARRAY1(p_event_arrobj, EV_MB_COL);
  
  convert_event_detections(p_event, 
                           EarthModel_NumSites(p_netmodel->p_earth),
                           EarthModel_NumTimeDefPhases(p_netmodel->p_earth),
                           p_netmodel->numdetections, p_netmodel->p_detections,
                           p_detlist_obj);
  
  numsc = locsc = magsc = detsc = dettimesc = detazsc = detslosc = 0;
  possdetcnt = detcnt = 0;
  
  score_event(p_netmodel, p_event, &numsc, &locsc, &magsc, &detsc, &dettimesc, 
              &detazsc, &detslosc, &detphasesc,
              &possdetcnt, &detcnt);
  
  score = numsc + locsc + magsc + detsc + dettimesc + detazsc + detslosc
    + detphasesc;
  
  free_events(1, p_event);
  
  return Py_BuildValue("d", score);
}
