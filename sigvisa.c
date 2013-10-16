#include <math.h>
#include <stdlib.h>

#define SIGVISA_MAIN_MODULE
#include "sigvisa.h"

PyObject * traceClass_obj;

static int py_sig_model_init(SigModel_t *self, PyObject *args);
static void py_sig_model_dealloc(SigModel_t * self);

static PyObject * py_mean_travel_time(SigModel_t * p_sigmodel,PyObject *args);
static PyObject * py_mean_travel_time_coord(SigModel_t * p_sigmodel,PyObject *args);
static PyObject * py_arrtime_logprob(SigModel_t * p_sigmodel,PyObject *args);
static PyObject * py_event_location_prior_logprob(SigModel_t * p_sigmodel,PyObject *args);
static PyObject * py_event_location_prior_sample(SigModel_t * p_sigmodel,PyObject *args);
static PyObject * py_event_mag_prior_logprob(SigModel_t * p_sigmodel, PyObject *args);
static PyObject * py_event_mag_prior_distribution(SigModel_t * p_sigmodel, PyObject *args);

static PyMethodDef SigModel_methods[] = {
  {"mean_travel_time", (PyCFunction)py_mean_travel_time, METH_VARARGS,
   "mean_travel_time(evlon, evlat, evdepth, siteid-1, phaseid-1)"
   " -> travel time in seconds"},
  {"mean_travel_time_coord", (PyCFunction)py_mean_travel_time_coord, METH_VARARGS,
   "mean_travel_time(evlon, evlat, evdepth, sitelon, sitelat, sitedepth, phaseid-1)"
   " -> travel time in seconds"},
  {"arrtime_logprob", (PyCFunction)py_arrtime_logprob, METH_VARARGS,
   "arrtime_logprob(arrtime, pred_arrtime, det_deltime, siteid-1, phaseid-1)"
   " -> log probability"},
  {"event_location_prior_logprob", (PyCFunction)py_event_location_prior_logprob, METH_VARARGS,
   "event_location_prior_logprob(evlon, evlat, evdepth)"
   " -> log prior probability of event location"},
  {"event_location_prior_sample", (PyCFunction)py_event_location_prior_sample, METH_VARARGS,
   "event_location_prior_sample()"
   " -> sample an event location from the prior"},
  {"event_mag_prior_logprob", (PyCFunction)py_event_mag_prior_logprob, METH_VARARGS,
   "event_mag_prior_logprob(mb)"
   " -> log prior probability of event magnitude"},
  {"event_mag_prior_distribution", (PyCFunction)py_event_mag_prior_distribution, METH_VARARGS,
   "event_mag_prior_distribution()"
   " -> log prior probability of event magnitude"},

  /*
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
    {"site_info", (PyCFunction)py_EarthModel_SiteInfo, METH_VARARGS,
     "Return information about a site\n"
     "site_info(sta, time) -> lon, lat, elev, isarr, ontime, offtime, ref_siteid"
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
  const char * numevent_fname;
  const char * evloc_fname;
  const char * evmag_fname;
  const char * arrtime_fname;
  const char * arrazi_fname;
  const char * arrslo_fname;
  const char * arramp_fname;

  if (!PyArg_ParseTuple(args, "Osssssss", &p_earth, &numevent_fname, &evloc_fname, &evmag_fname, &arrtime_fname, &arrazi_fname, &arrslo_fname, &arramp_fname)) {
    printf("ERROR: can't parse input args\n");
    CHECK_ERROR;
    return -1;
  }

  // TODO: check that siteid array is good

  self->p_earth = p_earth;
  Py_INCREF((PyObject *)self->p_earth);

  NumEventPrior_Init_Params(&self->num_event_prior, numevent_fname);
  EventLocationPrior_Init_Params(&self->event_location_prior, evloc_fname);
  EventMagPrior_Init_Params(&self->event_mag_prior, 1, evmag_fname);
  ArrivalTimeJointPrior_Init_Params(&self->arr_time_joint_prior, arrtime_fname);
  ArrivalAzimuthPrior_Init_Params(&self->arr_az_prior, arrazi_fname);
  ArrivalSlownessPrior_Init_Params(&self->arr_slo_prior, arrslo_fname);
  ArrivalAmplitudePrior_Init_Params(&self->arr_amp_prior, arramp_fname);

  //int numsites = EarthModel_NumSites(p_earth);

  return 0;
}

static PyMethodDef sigvisaMethods[] = {
  {NULL, NULL}
};

void initsigvisa_c(void)
{
  PyObject * m;

  py_sig_model.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_sig_model) < 0)
    return;

  py_EarthModel.tp_new = PyType_GenericNew;
  if (PyType_Ready(&py_EarthModel) < 0)
    return;


  m = Py_InitModule3("sigvisa_c", sigvisaMethods,
                     "Signal-Based Vertically Integrated Seismological Processing");
  import_array();/* Must be present for NumPy. Called first after above line.*/

  Py_INCREF(&py_sig_model);
  PyModule_AddObject(m, "SigModel", (PyObject *)&py_sig_model);

  Py_INCREF(&py_EarthModel);
  PyModule_AddObject(m, "EarthModel", (PyObject *)&py_EarthModel);

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

static PyObject * py_mean_travel_time_coord(SigModel_t * p_sigmodel,
                                      PyObject * args)
{
  double evlon, evlat, evdepth;
  double sitelon, sitelat, siteelev;
  int phaseid;
  double trvtime;

  EarthModel_t * p_earth;

  if (!PyArg_ParseTuple(args, "ddddddi", &evlon, &evlat, &evdepth,
                        &sitelon, &sitelat, &siteelev, &phaseid))
    return NULL;

  p_earth = p_sigmodel->p_earth;

  trvtime = EarthModel_ArrivalTime_Coord(p_earth, evlon, evlat, evdepth, 0, phaseid,
                                   sitelon, sitelat, siteelev);

  return Py_BuildValue("d", trvtime);
}


static PyObject * py_mean_travel_time(SigModel_t * p_sigmodel,
                                      PyObject * args)
{
  double evlon, evlat, evdepth, evtime;
  int phaseid;
  double trvtime;
  const char *sitename;

  EarthModel_t * p_earth;

  if (!PyArg_ParseTuple(args, "ddddsi", &evlon, &evlat, &evdepth, &evtime,
                        &sitename, &phaseid))
    return NULL;

  p_earth = p_sigmodel->p_earth;

  trvtime = EarthModel_ArrivalTime(p_earth, evlon, evlat, evdepth, 0, phaseid,
                                   sitename);
  Site_t * p_site = get_site(p_sigmodel->p_earth, sitename, evtime);

  if (trvtime == -2 || p_site == NULL) {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid site name, or site does not exist at specified time.");
    return NULL;
  }

  int ref_siteid = p_site->ref_siteid;

  trvtime += ArrivalTimePrior_MeanResidual(&p_sigmodel->arr_time_joint_prior.single_prior,
                                           ref_siteid-1, phaseid);

  return Py_BuildValue("d", trvtime);
}

static PyObject * py_event_location_prior_logprob(SigModel_t * p_sigmodel,
						  PyObject * args)
{
  double lon;
  double lat;
  double depth;

  double logprob;

  if (!PyArg_ParseTuple(args, "ddd", &lon, &lat, &depth))
    return NULL;

  logprob = EventLocationPrior_LogProb(&p_sigmodel->event_location_prior,
				       lon, lat, depth);
  return Py_BuildValue("d", logprob);
}

static PyObject * py_event_location_prior_sample(SigModel_t * p_sigmodel,
						 PyObject * args)
{
  double lon;
  double lat;
  double depth;

  EventLocationPrior_Sample(&p_sigmodel->event_location_prior,
			    &lon, &lat, &depth);

  return Py_BuildValue("ddd", lon, lat, depth);
}


static PyObject * py_event_mag_prior_logprob(SigModel_t * p_sigmodel,
					     PyObject * args)
{
  double val;
  double logprob;

  if (!PyArg_ParseTuple(args, "d", &val))
    return NULL;

  logprob = EventMagPrior_LogProb(&p_sigmodel->event_mag_prior,
				       val, 0);
  return Py_BuildValue("d", logprob);
}

static PyObject * py_event_mag_prior_distribution(SigModel_t * p_sigmodel,
						  PyObject * args)
{
  return Py_BuildValue("dd", p_sigmodel->event_mag_prior.min_mag, p_sigmodel->event_mag_prior.mag_rate);
}


static PyObject * py_arrtime_logprob(SigModel_t * p_sigmodel,
                                     PyObject * args)
{
  double arrtime;
  double pred_arrtime;
  double det_deltime;
  int siteid;
  int phaseid;

  double logprob;

  if (!PyArg_ParseTuple(args, "dddii", &arrtime, &pred_arrtime, &det_deltime,
                        &siteid, &phaseid))
    return NULL;

  logprob = ArrivalTimePrior_LogProb(&p_sigmodel->arr_time_joint_prior.single_prior,
                                     arrtime, pred_arrtime,
                                     det_deltime, siteid, phaseid);

  return Py_BuildValue("d", logprob);
}

static void py_sig_model_dealloc(SigModel_t * self)
{
  if (self->p_earth)
    {
      Py_DECREF((PyObject *)self->p_earth);
      self->p_earth = NULL;
    }


  EventLocationPrior_UnInit(&self->event_location_prior);
  ArrivalTimeJointPrior_UnInit(&self->arr_time_joint_prior);


}
