#include <math.h>
#include <stdlib.h>

#include "sigvisa.h"

static int py_sig_model_init(SigModel_t *self, PyObject *args);
static void py_net_model_dealloc(SigModel_t * self);

static PyObject * py_score_world(SigModel_t * p_netmodel, PyObject * args);

static PyMethodDef NetModel_methods[] = {
  {"score_world", (PyCFunction)py_score_world, METH_VARARGS,
   "score_world(events, ev_detlist, verbose) "
   "-> log probability\n"},
};

static PyTypeObject py_net_model = {
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
  NetModel_t * p_netmodel;
  PyObject * siteid;
  double start_time;
  double end_time;


  const char * stanoise_fname;
  const char * env_fname;

  
  if (!PyArg_ParseTuple(args, "ddO!O!O",  &start_time, &end_time, &py_EarthModel, &p_earth, &py_NetModel, &p_netmodel, &siteid, &stanoise_fname, &env_fname))
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

  self->p_netmodel = p_netmodel;
  Py_INCREF((PyObject *)self->p_netmodel);

  self->siteid = siteid;
  Py_INCREF((PyObject *)self->siteid);

  
  // is this (split into envelope and noise) the right way to structure the model?
  StationNoisePrior_Init_Params(&self->sta_noise_prior, stanoise_fname);
  EnvelopePrior_Init_Params(&env_prior->env_prior, env_fname);

  return 0;
}

static void py_sig_model_dealloc(NetModel_t * self)
{
  if (self->p_earth)
  {
    Py_DECREF((PyObject *)self->p_earth);
    self->p_earth = NULL;
  }
  
  if (self->p_netmodel)
  {
    Py_DECREF((PyObject *)self->p_netmodel);
    self->p_netmodel = NULL;
  }
  
  if (self->p_siteid)
  {
    Py_DECREF((PyObject *)self->p_siteid);
    self->p_siteid = NULL;
  }

  StationNoisePrior_UnInit(&self->station_noise_prior);
  EnvelopePrior_UnInit(&self->envelope_prior);

}


static PyObject * py_score_world(SigModel_t * p_netmodel, PyObject * args)
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
  
  score = score_world(p_sigmodel, numevents, p_events, verbose);
  
  free_events(numevents, p_events);
  
  return Py_BuildValue("d", score);
}
