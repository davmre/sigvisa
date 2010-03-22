#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdlib.h>

#include "netvisa.h"

static int py_net_model_init(NetModel_t *self, PyObject *args);
static void py_net_model_dealloc(NetModel_t * self);
static PyObject * py_score_world(PyObject * self, PyObject * args);

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
    "Network Model",                         /* tp_doc */
    0,                                       /* tp_traverse */
    0,                                       /* tp_clear */
    0,                                       /* tp_richcompare */
    0,                                       /* tp_weaklistoffset */
    0,                                       /* tp_iter */
    0,                                       /* tp_iternext */
    0,                                       /* tp_methods */
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

static PyMethodDef netvisaMethods[] = {
  {"score_world", py_score_world, METH_VARARGS,
   "score_world(net_model, events, verbose) -> log probability\n"},
  {NULL, NULL}
};

void initnetvisa()
{
  PyObject * m;
  
  py_net_model.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_net_model) < 0)
    return;

  m = Py_InitModule3("netvisa", netvisaMethods,
                     "Network Vertically Integrated Seismological Processing");
  import_array();/* Must be present for NumPy. Called first after above line.*/

  Py_INCREF(&py_net_model);
  PyModule_AddObject(m, "NetModel", (PyObject *)&py_net_model);
}

static int py_net_model_init(NetModel_t *self, PyObject *args)
{
  double start_time;
  double end_time;
  const char * numevent_fname;
  const char * evloc_fname;

  if (!PyArg_ParseTuple(args, "ddss", &start_time, &end_time, &numevent_fname,
                        &evloc_fname))
    return -1;
  
  if (end_time <= start_time)
  {
    PyErr_SetString(PyExc_ValueError, 
                    "NetModel: error: end_time <= start_time");
    return -1;
  }
  
  NumEventPrior_Init_Params(&self->num_event_prior, 2, numevent_fname,
                            end_time - start_time);
  
  EventLocationPrior_Init_Params(&self->event_location_prior, 1, evloc_fname);

  return 0;
}

static void py_net_model_dealloc(NetModel_t * self)
{
  EventLocationPrior_UnInit(&self->event_location_prior);
  self->ob_type->tp_free((PyObject*)self);
}

static void alloc_events(PyArrayObject * p_events_arrobj, int * p_numevents,
                         Event_t ** p_p_events)
{
  int numevents;
  Event_t * p_events;
  int i;
  
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

static void free_events(Event_t * p_events)
{
  free(p_events);
}

static PyObject * py_score_world(PyObject * self, PyObject * args)
{
  /* input arguments */
  NetModel_t * p_netmodel;
  PyArrayObject * p_events_arrobj;
  int verbose;
 
  int numevents;
  Event_t * p_events;
  double score;
  
  if (!PyArg_ParseTuple(args, "O!O!i", &py_net_model, &p_netmodel,
                        &PyArray_Type, &p_events_arrobj, &verbose)
      || (NULL == p_netmodel) || (NULL == p_events_arrobj))
    return NULL;

  if ((2 != p_events_arrobj->nd) || (NPY_DOUBLE 
                                     != p_events_arrobj->descr->type_num)
      || (EV_NUM_COLS != p_events_arrobj->dimensions[1]))
  {
    PyErr_SetString(PyExc_ValueError,
                    "score_world: wrong shape or type of events array");
    return NULL;
  }

  alloc_events(p_events_arrobj, &numevents, &p_events);
  
  score = score_world(p_netmodel, numevents, p_events, verbose);
  
  free_events(p_events);
  
  return Py_BuildValue("d", score);
}
