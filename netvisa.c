/*
 * Copyright (c) 2012, Bayesian Logic, Inc.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Bayesian Logic, Inc. nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 * Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
 */

#include <math.h>
#include <stdlib.h>

#define NETVISA_MAIN_MODULE

#include "sigvisa.h"

static int py_net_model_init(NetModel_t *self, PyObject *args);
static void py_net_model_dealloc(NetModel_t * self);
static PyObject * py_score_world(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_logprob_false(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_score_event(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_prob_event(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_score_event_det(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_logprob_event_misdet(NetModel_t * p_netmodel,
                                          PyObject * args);
static PyObject * py_invert_det(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_location_logprob(NetModel_t * p_netmodel,PyObject * args);
static PyObject * py_location_sample(NetModel_t * p_netmodel);
static PyObject * py_detection_logprob(NetModel_t * p_netmodel,PyObject *args);

static PyObject * py_logamp(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_arramp_zval_logprob_cdf(NetModel_t * p_netmodel,
                                             PyObject *args);
static PyObject * py_falseamp_logprob_cdf(NetModel_t * p_netmodel,
                                          PyObject *args);
static PyObject * py_arrtime_logprob(NetModel_t * p_netmodel,PyObject *args);
static PyObject * py_mean_travel_time(NetModel_t * p_netmodel,PyObject *args);
static PyObject * py_mean_amplitude(NetModel_t * p_netmodel,PyObject *args);
static PyObject * py_arraz_logprob(NetModel_t * p_netmodel,PyObject *args);
static PyObject * py_arrslo_logprob(NetModel_t * p_netmodel,PyObject *args);
static PyObject * py_arrphase_logprob(NetModel_t * p_netmodel,PyObject *args);
static PyObject * py_falsephase_logprob(NetModel_t * p_netmodel,PyObject *args);
static PyObject * py_enable_sec_arr(NetModel_t * self);
static PyObject * py_disable_sec_arr(NetModel_t * self);
static PyObject * py_set_temperature(NetModel_t * self, PyObject * args);
static PyObject * py_score_coda_coda(NetModel_t * p_netmodel, PyObject * args);
static PyObject * py_score_phase_coda(NetModel_t * p_netmodel, PyObject * args);


static PyMethodDef NetModel_methods[] = {
  {"score_world", (PyCFunction)py_score_world, METH_VARARGS,
   "score_world(events, ev_detlist, verbose) "
   "-> log probability\n"},
  {"logprob_false", (PyCFunction)py_logprob_false, METH_VARARGS,
   "logprob_false(falsedets, verbose) "
   "-> log probability\n"},
  {"score_event", (PyCFunction)py_score_event, METH_VARARGS,
   "score_event(event, detlist) -> log probability ratio"},
  {"score_event_det", (PyCFunction)py_score_event_det, METH_VARARGS,
   "score_event_det(event, phaseid, detnum) -> log probability ratio\n"},
  {"logprob_event_misdet", (PyCFunction)py_logprob_event_misdet, METH_VARARGS,
   "logprob_event_misdet(event, phaseid, siteid) -> "
   "log probability of mis-detection\n"},
  {"prob_event", (PyCFunction)py_prob_event, METH_VARARGS,
   "prob_event(event, detlist) -> log probability ratio"},
  {"infer", (PyCFunction)py_infer, METH_VARARGS,
   "infer(runid, numsamples, window, step, threads, propose_events, verbose,"
   "write_cb)\n -> events, ev_detlist"},
  {"propose", (PyCFunction)py_propose, METH_VARARGS,
   "propose(time_low, time_high, det_low, det_high, degree_step, num_step)\n"
   " -> events, ev_detlist"},
  {"invert_det", (PyCFunction)py_invert_det, METH_VARARGS,
   "invert_det(detnum, perturb?) -> (lon, lat, depth, time) or None"},
  {"location_logprob", (PyCFunction)py_location_logprob, METH_VARARGS,
   "location_logprob(lon, lat, depth) -> probability"},
  {"location_sample", (PyCFunction)py_location_sample, METH_NOARGS,
   "location_sample() -> (lon, lat, depth)"},
  {"detection_logprob", (PyCFunction)py_detection_logprob, METH_VARARGS,
   "detection_logprob(isdet, evdepth, evmag, dist, siteid, phaseid)"
   " -> log probability"},
  {"arramp_zval_logprob_cdf", (PyCFunction)py_arramp_zval_logprob_cdf, 
   METH_VARARGS,
   "arramp_zval_logprob_cdf(mb, depth, ttime, siteid, phaseid, amp)\n"
   " -> zval, log prob, cdf"},
  {"logamp", (PyCFunction)py_logamp, METH_VARARGS,
   "logamp(mb, depth, ttime, siteid, phaseid)\n -> log(amp)"},
  {"falseamp_logprob_cdf", (PyCFunction)py_falseamp_logprob_cdf, METH_VARARGS,
   "falseamp_logprob_cdf(siteid, amp) -> log prob, cdf"},
  {"arrtime_logprob", (PyCFunction)py_arrtime_logprob, METH_VARARGS,
   "arrtime_logprob(arrtime, pred_arrtime, det_deltime, siteid, phaseid)"
   " -> log probability"},
  {"mean_travel_time", (PyCFunction)py_mean_travel_time, METH_VARARGS,
   "mean_travel_time(evlon, evlat, evdepth, siteid, phaseid)"
   " -> travel time in seconds"},
  {"mean_amplitude", (PyCFunction)py_mean_amplitude, METH_VARARGS,
   "mean_amplitude(mb, depth, ttime, siteid, phaseid)"
   " -> mean amplitude"},
  {"arraz_logprob", (PyCFunction)py_arraz_logprob, METH_VARARGS,
   "arraz_logprob(arraz, pred_arraz, det_delaz, siteid, phaseid)"
   " -> log probability"},
  {"arrslo_logprob", (PyCFunction)py_arrslo_logprob, METH_VARARGS,
   "arrslo_logprob(arrslo, pred_arrslo, det_delslo, siteid, phaseid)"
   " -> log probability"},
  {"arrphase_logprob", (PyCFunction)py_arrphase_logprob, METH_VARARGS,
   "arrphase_logprob(arrphaseid, phaseid) -> log probability"},
  {"falsephase_logprob", (PyCFunction)py_falsephase_logprob, METH_VARARGS,
   "falsephase_logprob(arrphaseid) -> log probability"},
  {"enable_sec_arr", (PyCFunction)py_enable_sec_arr, METH_NOARGS,
   "enable_sec_arr(): enables secondary arrivals"},
  {"disable_sec_arr", (PyCFunction)py_disable_sec_arr, METH_NOARGS,
   "disable_sec_arr(): disables secondary arrivals"},
  {"set_temperature", (PyCFunction)py_set_temperature, METH_VARARGS,
   "set_temperature(temp): set the temperature for constraints"},
  {"score_coda_coda", (PyCFunction)py_score_coda_coda, METH_VARARGS,
   "score_coda_coda(secdetnum, detnum)"
   "-> log probability of secdetnum being a secondary coda of detnum"
   " minus log probability of secdetnum being noise"},
  {"score_phase_coda", (PyCFunction)py_score_phase_coda, METH_VARARGS,
   "score_phase_coda(secdetnum, detnum)"
   "-> log probability of secdetnum being a secondary coda of detnum"
   " minus log probability of secdetnum being noise"},
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
    "netvisa.EarthModel",                    /*tp_name*/
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

static PyMethodDef netvisaMethods[] = {
  {NULL, NULL}
};

void initnetvisa(void)
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

static void alloc_site_up(PyArrayObject * siteupobj,
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
  
  *p_ntime = ntime;
  *p_p_site_up = p_site_up;
}

static void free_site_up(int nsites, int ntime, int * p_site_up)
{
  free(p_site_up);
}

static void init_det_logprob(NetModel_t * self)
{
  int detnum;

  /* initialize prev_detnum and logprob */
  for (detnum = 0; detnum < self->numdetections; detnum++)
  {
    Detection_t * p_det = self->p_detections + detnum;
    
    p_det->prev_det = -1;
    p_det->logprob_det = logprob_noise(self, p_det, NULL);
  }

  /* look at each detection and see if it caused a secondary detection */
  for (detnum = 0; detnum < self->numdetections; detnum++)
  {
    int secdetnum;
    Detection_t * p_det = self->p_detections + detnum;
    
    for (secdetnum = (detnum+1); secdetnum < self->numdetections; 
         secdetnum ++)
    {
      Detection_t * p_secdet = self->p_detections + secdetnum;
      /* sanity check to avoid searching forever */
      if (p_secdet->time_det > (p_det->time_det + 300))
        break;
      /* needs to be the same site to be a coda arrival */
      if (p_secdet->site_det != p_det->site_det)
        continue;
      /* did we find a coda arrival ? */
      if (SecDetPrior_Time_Possible(&self->sec_det_prior,
                                    p_secdet->time_det, p_det->time_det))
      {
        double coda_logprob = logprob_coda_coda(self, p_secdet, p_det);

        p_secdet->logprob_det = logprob_noise(self, p_secdet, p_det);

        /* we will consider this a secondary detection only if the coda->coda
         * explanation is better than the noise explanation */
        if (coda_logprob > p_secdet->logprob_det)
        {
          p_secdet->prev_det = detnum;
          p_secdet->logprob_det = coda_logprob;
        }
      }
      break;
    }
  }
}


static int py_net_model_init(NetModel_t *self, PyObject *args)
{
  EarthModel_t * p_earth;
  double start_time;
  double end_time;
  PyArrayObject * detectionsobj;
  PyArrayObject * siteupobj;
  const char * secdet_fname;
  const char * numevent_fname;
  const char * evloc_fname;
  const char * evmag_fname;
  const char * evdet_fname;
  const char * arrtime_fname;
  const char * numfalse_fname;
  const char * arraz_fname;
  const char * arrslo_fname;
  const char * arrphase_fname;
  const char * arrsnr_fname;
  const char * arramp_fname;
  
  if (!PyArg_ParseTuple(args, "O!ddO!O!ssssssssssss", &py_EarthModel, 
                        &p_earth,
                        &start_time, &end_time, 
                        &PyArray_Type, &detectionsobj,
                        &PyArray_Type, &siteupobj, 
                        &secdet_fname,
                        &numevent_fname, &evloc_fname, &evmag_fname, 
                        &evdet_fname, &arrtime_fname, &numfalse_fname,
                        &arraz_fname, &arrslo_fname, &arrphase_fname,
                        &arrsnr_fname, &arramp_fname)
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

  /* note: we expect detections for an additional period of time beyond the
   * end_time */
  if ((2 != siteupobj->nd) || (NPY_BOOL != siteupobj->descr->type_num)
      || (((int)ceil((MAX_TRAVEL_TIME + end_time - start_time) / UPTIME_QUANT))
          != siteupobj->dimensions[1]))
  {
    PyErr_SetString(PyExc_ValueError, "net_model_init: incorrect shape or type"
                    " of site_up array");
    return -1;
  }

  self->temperature = 100.0;
  self->start_time = start_time;
  self->end_time = end_time;
  self->numsites = siteupobj->dimensions[0];

  self->p_earth = p_earth;
  Py_INCREF((PyObject *)self->p_earth);

  alloc_detections(detectionsobj, &self->numdetections, &self->p_detections);
  
  alloc_site_up(siteupobj, &self->num_uptime, &self->p_site_up);
  
  /* we will enable secondary arrivals, it can be modified by calling one of
   * the enable/disable methods */
  self->enable_sec_arr = 1;
  
  SecDetPrior_Init_Params(&self->sec_det_prior, secdet_fname);

  NumEventPrior_Init_Params(&self->num_event_prior, numevent_fname);
  
  EventLocationPrior_Init_Params(&self->event_location_prior, evloc_fname);
  
  EventMagPrior_Init_Params(&self->event_mag_prior, 1, evmag_fname);

  EventDetectionPrior_Init_Params(&self->event_det_prior, evdet_fname);

  ArrivalTimePrior_Init_Params(&self->arr_time_prior, arrtime_fname);
  
  NumFalseDetPrior_Init_Params(&self->num_falsedet_prior, numfalse_fname);
 
  ArrivalAzimuthPrior_Init_Params(&self->arr_az_prior, arraz_fname);

  ArrivalSlownessPrior_Init_Params(&self->arr_slo_prior, arrslo_fname);

  ArrivalPhasePrior_Init_Params(&self->arr_phase_prior, arrphase_fname);

  ArrivalSNRPrior_Init_Params(&self->arr_snr_prior, arrsnr_fname);

  ArrivalAmplitudePrior_Init_Params(&self->arr_amp_prior, arramp_fname);

  init_det_logprob(self);
  
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

  free_site_up(self->numsites, self->num_uptime, self->p_site_up);
  self->p_site_up = NULL;

  SecDetPrior_UnInit(&self->sec_det_prior);
  
  EventLocationPrior_UnInit(&self->event_location_prior);

  ArrivalTimePrior_UnInit(&self->arr_time_prior);

  NumFalseDetPrior_UnInit(&self->num_falsedet_prior);

  ArrivalAzimuthPrior_UnInit(&self->arr_az_prior);

  ArrivalSlownessPrior_UnInit(&self->arr_slo_prior);

  ArrivalPhasePrior_UnInit(&self->arr_phase_prior);

  ArrivalSNRPrior_UnInit(&self->arr_snr_prior);

  ArrivalAmplitudePrior_UnInit(&self->arr_amp_prior);
  
  self->ob_type->tp_free((PyObject*)self);
}

static void convert_event_detections(Event_t * p_event, int numsites,
                                     int numtimedefphases, int numdetections,
                                     Detection_t * p_detections,
                                     PyObject * phasedet_list)
{
  Py_ssize_t j;

  p_event->p_all_detids = (int *) malloc(numsites * numtimedefphases *
                                         MAX_PHASE_DET *
                                         sizeof(*p_event->p_all_detids));
  
  /* allocate and initialize the number of detections to 0 */
  p_event->p_num_dets = (int *) calloc(numsites * numtimedefphases,
                                       sizeof(*p_event->p_num_dets));
  
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
    int numdet;
    int k;
      
    phasedet_tuple = PyList_GetItem(phasedet_list, j);
    assert(phasedet_tuple);
    assert(PyTuple_Check(phasedet_tuple));

    phaseobj = PyTuple_GetItem(phasedet_tuple, 0);
    assert(phaseobj && PyInt_Check(phaseobj));
    
    detobj = PyTuple_GetItem(phasedet_tuple, 1);
    assert(detobj && PyInt_Check(detobj));
    
    numdet = MIN((int) PyTuple_GET_SIZE(phasedet_tuple) - 1, MAX_PHASE_DET);
    
    phaseid = PyInt_AS_LONG(phaseobj);
    detid = PyInt_AS_LONG(detobj);

    assert((detid >= 0) && (detid < numdetections));
    siteid = p_detections[detid].site_det;

    p_event->p_num_dets[siteid * numtimedefphases + phaseid] = numdet;
    
    for (k=0; k<numdet; k++)
    {
      detobj = PyTuple_GetItem(phasedet_tuple, k+1);
      assert(detobj && PyInt_Check(detobj));
      
      detid = PyInt_AS_LONG(detobj);
      assert((detid >= 0) && (detid < numdetections));
    
      p_event->p_all_detids[siteid * numtimedefphases * MAX_PHASE_DET 
                            + phaseid * MAX_PHASE_DET + k] = detid;
    }
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

void free_events(int numevents, Event_t * p_events)
{
  int i;
  for (i=0; i<numevents; i++)
  {
    free(p_events[i].p_all_detids);
    free(p_events[i].p_num_dets);
  }
  
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

static PyObject * py_logprob_false(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  PyObject * p_falsedets_obj;
  int verbose;
  int numdets;
  int * p_detids;
  int i;
  
  double score;
  
  if (!PyArg_ParseTuple(args, "O!i",
                        &PyList_Type, &p_falsedets_obj, &verbose)
      || !p_falsedets_obj)
    return NULL;

  numdets = PyList_Size(p_falsedets_obj);
  
  p_detids = (int *)malloc(numdets * sizeof(*p_detids));

  if (!p_detids)
    return NULL;

  for (i=0; i<numdets; i++)
  {
    PyObject * p_item = PyList_GetItem(p_falsedets_obj, i);

    if (!p_item)
    {
      free(p_detids);
      return NULL;
    }

    p_detids[i] = (int)PyInt_AsLong(p_item);
  }

  if (verbose)
    printf("%d false detections\n", numdets);

  score = logprob_false(p_netmodel, numdets, p_detids, verbose);
  
  free(p_detids);

  return Py_BuildValue("d", score);
}

static PyObject * py_score_event(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  PyArrayObject * p_event_arrobj;
  PyObject * p_detlist_obj;
 
  Event_t * p_event;
  double score;
  
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
  
  score = score_event(p_netmodel, p_event);
  
  free_events(1, p_event);
  
  return Py_BuildValue("d", score);
}

static PyObject * py_score_event_det(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  PyArrayObject * p_event_arrobj;
  int phaseid;
  int detnum;
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
  Detection_t * p_det;
 
  Event_t * p_event;
  double distance, pred_az;
  int poss;
  double score;

  
  if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &p_event_arrobj, 
                        &phaseid, &detnum)
      || !p_event_arrobj)
    return NULL;
  
  if ((1 != p_event_arrobj->nd) || (NPY_DOUBLE 
                                     != p_event_arrobj->descr->type_num)
      || (EV_NUM_COLS != p_event_arrobj->dimensions[0]))
  {
    PyErr_SetString(PyExc_ValueError,
                    "score_event_det: wrong shape or type of event array");
    return NULL;
  }
  
  p_earth = p_netmodel->p_earth;
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  numsites = EarthModel_NumSites(p_earth);
  
  if ((phaseid < 0) || (phaseid >= numtimedefphases))
  {
    PyErr_SetString(PyExc_ValueError, "score_event_det: invalid phaseid");
    return NULL;
  }

  if ((detnum < 0) || (detnum >= p_netmodel->numdetections))
  {
    PyErr_SetString(PyExc_ValueError, "score_event_det: invalid detnum");
    return NULL;
  }

  p_event = (Event_t *)calloc(1, sizeof(*p_event));
  p_event->evlon = ARRAY1(p_event_arrobj, EV_LON_COL);
  p_event->evlat = ARRAY1(p_event_arrobj, EV_LAT_COL);
  p_event->evdepth = ARRAY1(p_event_arrobj, EV_DEPTH_COL);
  p_event->evtime = ARRAY1(p_event_arrobj, EV_TIME_COL);
  p_event->evmag = ARRAY1(p_event_arrobj, EV_MB_COL);

  p_event->p_all_detids = (int *) malloc(numsites * numtimedefphases *
                                         MAX_PHASE_DET *
                                         sizeof(*p_event->p_all_detids));
  
  /* allocate and initialize the number of detections to 0 */
  p_event->p_num_dets = (int *) calloc(numsites * numtimedefphases,
                                       sizeof(*p_event->p_num_dets));

  p_det = p_netmodel->p_detections + detnum;
  
  p_event->p_num_dets[p_det->site_det * numtimedefphases + phaseid] = 1;
  p_event->p_all_detids[p_det->site_det * numtimedefphases * MAX_PHASE_DET
                        + phaseid * MAX_PHASE_DET + 0] = detnum;
  
  distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                              p_det->site_det);

  pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                      p_event->evlat, p_det->site_det);

  poss = score_event_site_phase(p_netmodel, p_event, p_det->site_det, phaseid,
                                distance, pred_az, &score);
  
  free_events(1, p_event);
  
  if (poss)
    return Py_BuildValue("d", score);
  /* if the detections is not possible then return None */
  else
  {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject * py_logprob_event_misdet(NetModel_t * p_netmodel,
                                          PyObject * args)
{
  /* input arguments */
  PyArrayObject * p_event_arrobj;
  int phaseid;
  int siteid;
  EarthModel_t * p_earth;
  int numsites;
  int numtimedefphases;
 
  Event_t * p_event;
  double distance, pred_az;
  int poss;
  double score;

  
  if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &p_event_arrobj, 
                        &phaseid, &siteid)
      || !p_event_arrobj)
    return NULL;
  
  if ((1 != p_event_arrobj->nd) || (NPY_DOUBLE 
                                     != p_event_arrobj->descr->type_num)
      || (EV_NUM_COLS != p_event_arrobj->dimensions[0]))
  {
    PyErr_SetString(PyExc_ValueError,
                    "prob_event_misdet: wrong shape or type of event array");
    return NULL;
  }
  
  p_earth = p_netmodel->p_earth;
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  numsites = EarthModel_NumSites(p_earth);
  
  if ((phaseid < 0) || (phaseid >= numtimedefphases))
  {
    PyErr_SetString(PyExc_ValueError, "prob_event_misdet: invalid phaseid");
    return NULL;
  }

  if ((siteid < 0) || (siteid >= numsites))
  {
    PyErr_SetString(PyExc_ValueError, "prob_event_misdet: invalid siteid");
    return NULL;
  }

  p_event = (Event_t *)calloc(1, sizeof(*p_event));
  p_event->evlon = ARRAY1(p_event_arrobj, EV_LON_COL);
  p_event->evlat = ARRAY1(p_event_arrobj, EV_LAT_COL);
  p_event->evdepth = ARRAY1(p_event_arrobj, EV_DEPTH_COL);
  p_event->evtime = ARRAY1(p_event_arrobj, EV_TIME_COL);
  p_event->evmag = ARRAY1(p_event_arrobj, EV_MB_COL);

  p_event->p_all_detids = (int *) malloc(numsites * numtimedefphases *
                                         MAX_PHASE_DET *
                                         sizeof(*p_event->p_all_detids));
  
  /* allocate and initialize the number of detections to 0 */
  p_event->p_num_dets = (int *) calloc(numsites * numtimedefphases,
                                       sizeof(*p_event->p_num_dets));

  distance = EarthModel_Delta(p_earth, p_event->evlon, p_event->evlat,
                              siteid);

  pred_az = EarthModel_ArrivalAzimuth(p_earth, p_event->evlon,
                                      p_event->evlat, siteid);

  poss = score_event_site_phase(p_netmodel, p_event, siteid, phaseid,
                                distance, pred_az, &score);
  
  free_events(1, p_event);
  
  if (poss)
    return Py_BuildValue("d", score);
  /* if the detections is not possible then return None */
  else
  {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject * py_set_temperature(NetModel_t * self, PyObject * args)
{
  double temperature;
  if (!PyArg_ParseTuple(args, "d", &temperature))
    return NULL;
  self->temperature = temperature;
  Py_INCREF(Py_None);
  return Py_None;  
}

static PyObject * py_score_coda_coda(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  int secdetnum;
  int detnum;
  Detection_t * p_det;
  Detection_t * p_secdet;
  
  if (!PyArg_ParseTuple(args, "ii", &secdetnum, &detnum))
    return NULL;

  if ((secdetnum < 0) || (secdetnum >= p_netmodel->numdetections))
  {
    PyErr_SetString(PyExc_ValueError, 
                    "score_coda_coda: error: illegal secdetnum");
    return NULL;
  }
  
  if ((detnum < 0) || (detnum >= p_netmodel->numdetections))
  {
    PyErr_SetString(PyExc_ValueError, 
                    "score_coda_coda: error: illegal detnum");
    return NULL;
  }

  p_det = p_netmodel->p_detections + detnum;
  p_secdet = p_netmodel->p_detections + secdetnum;

  if (p_det->site_det != p_secdet->site_det)
  {
    PyErr_SetString(PyExc_ValueError, 
                    "score_coda_coda: error: detections from different sites");
    return NULL;
  }

  if (!SecDetPrior_Time_Possible(&p_netmodel->sec_det_prior,
                                 p_secdet->time_det,
                                 p_det->time_det))
  {
    return Py_BuildValue("d", -1.0);
  }
  
  return Py_BuildValue("d", score_coda_coda(p_netmodel, p_secdet, p_det));
}

static PyObject * py_score_phase_coda(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  int secdetnum;
  int detnum;
  Detection_t * p_det;
  Detection_t * p_secdet;
  
  if (!PyArg_ParseTuple(args, "ii", &secdetnum, &detnum))
    return NULL;

  if ((secdetnum < 0) || (secdetnum >= p_netmodel->numdetections))
  {
    PyErr_SetString(PyExc_ValueError, 
                    "score_phase_coda: error: illegal secdetnum");
    return NULL;
  }
  
  if ((detnum < 0) || (detnum >= p_netmodel->numdetections))
  {
    PyErr_SetString(PyExc_ValueError, 
                    "score_phase_coda: error: illegal detnum");
    return NULL;
  }

  p_det = p_netmodel->p_detections + detnum;
  p_secdet = p_netmodel->p_detections + secdetnum;

  if (p_det->site_det != p_secdet->site_det)
  {
    PyErr_SetString(PyExc_ValueError, 
                    "score_phase_coda: error: detections from different sites");
    return NULL;
  }

  if (!SecDetPrior_Time_Possible(&p_netmodel->sec_det_prior,
                                 p_secdet->time_det,
                                 p_det->time_det))
  {
    return Py_BuildValue("d", -1.0);
  }
  
  return Py_BuildValue("d", score_phase_coda(p_netmodel, p_secdet, p_det));
}

static PyObject * py_prob_event(NetModel_t * p_netmodel, PyObject * args)
{
  /* input arguments */
  PyArrayObject * p_event_arrobj;
  PyObject * p_detlist_obj;
 
  Event_t * p_event;
  double prob;
  
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
                    "prob_event: wrong shape or type of event array");
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
  
  prob = prob_event(p_netmodel, p_event);
  
  free_events(1, p_event);
  
  return Py_BuildValue("d", prob);
}

static PyObject * py_invert_det(NetModel_t * p_netmodel, PyObject * args)
{
  int detnum;
  int perturb;
  int status;
  Event_t event;

  if (!PyArg_ParseTuple(args, "ii", &detnum, &perturb))
    return NULL;

  if ((detnum < 0) || (detnum >= p_netmodel->numdetections))
  {
    PyErr_SetString(PyExc_ValueError, "invert_det: illegal detnum");
    return NULL;
  }

  status = invert_detection(p_netmodel->p_earth, p_netmodel->p_detections +
                            detnum, &event, perturb);
  
  if (status == 0)
    return Py_BuildValue("dddd", event.evlon, event.evlat,
                         event.evdepth, event.evtime);
  else
  {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

static PyObject * py_location_logprob(NetModel_t * p_netmodel, PyObject * args)
{
  double lon;
  double lat;
  double depth;
  
  if (!PyArg_ParseTuple(args, "ddd", &lon, &lat, &depth))
    return NULL;


  return Py_BuildValue("d", EventLocationPrior_LogProb(&p_netmodel
                                                       ->event_location_prior,
                                                       lon, lat, depth));
}

static PyObject * py_location_sample(NetModel_t * p_netmodel)
{
  double lon;
  double lat;
  double depth;
  
  EventLocationPrior_Sample(&p_netmodel->event_location_prior,
                            &lon, &lat, &depth);
  
  return Py_BuildValue("ddd", lon, lat, depth);
}

static PyObject * py_detection_logprob(NetModel_t * p_netmodel,
                                       PyObject * args)
{
  int isdet;
  double evdepth;
  double evmag;
  double dist;
  int siteid;
  int phaseid;
  double dderror;
  
  if (!PyArg_ParseTuple(args, "idddii", &isdet, &evdepth, &evmag, &dist,
                        &siteid, &phaseid))
    return NULL;

  /* we are assuming a noise value of 0 */
  dderror = dist_depth_range_error(p_netmodel->p_earth, phaseid, evdepth,
                                   dist);

  return Py_BuildValue("d",
                       EventDetectionPrior_LogProb(&p_netmodel
                                                   ->event_det_prior,
                                                   isdet, evdepth, evmag,
                                                   dist, siteid, phaseid,
                                           dderror / p_netmodel->temperature));
}

static PyObject * py_arramp_zval_logprob_cdf(NetModel_t * p_netmodel,
                                             PyObject * args)
{
  double mb, depth, ttime, amp;
  int siteid, phaseid;

  double zval, logprob, cdf;
  
  if (!PyArg_ParseTuple(args, "dddiid", &mb, &depth, &ttime,
                        &siteid, &phaseid, &amp))
    return NULL;
  
  /* a -1 amplitude indicates that the amplitude was not observed */
  if (-1 != amp)
  {
    zval = ArrivalAmplitudePrior_zval(&p_netmodel->arr_amp_prior,
                                      mb, depth, ttime, siteid,
                                      phaseid, amp);
    
    logprob = ArrivalAmplitudePrior_LogProb(&p_netmodel->arr_amp_prior,
                                            mb, depth, ttime, siteid,
                                            phaseid, amp);
  
    cdf = ArrivalAmplitudePrior_cdf(&p_netmodel->arr_amp_prior,
                                    mb, depth, ttime, siteid,
                                    phaseid, amp);
    
    return Py_BuildValue("ddd", zval, logprob, cdf);
  }
  else
  {
    Py_INCREF(Py_None);
    return Py_None;
  }  
}

static PyObject * py_logamp(NetModel_t * p_netmodel, PyObject * args)
{
  double mb, depth, ttime;
  int siteid, phaseid;

  double out_logamp;
  
  if (!PyArg_ParseTuple(args, "dddii", &mb, &depth, &ttime,
                        &siteid, &phaseid))
    return NULL;
  
  out_logamp = ArrivalAmplitudePrior_logamp(&p_netmodel->arr_amp_prior,
                                            mb, depth, ttime, siteid,
                                            phaseid);
  return Py_BuildValue("d", out_logamp);
}

static PyObject * py_falseamp_logprob_cdf(NetModel_t * p_netmodel,
                                          PyObject * args)
{
  double amp;
  int siteid;

  double logprob, cdf;
  
  if (!PyArg_ParseTuple(args, "id", &siteid, &amp))
    return NULL;
  
  /* a -1 amplitude indicates that the amplitude was not observed */
  if (-1 != amp)
  {
    logprob = FalseArrivalAmplitudePrior_LogProb(&p_netmodel->arr_amp_prior,
                                                 siteid, amp);
  
    cdf = FalseArrivalAmplitudePrior_cdf(&p_netmodel->arr_amp_prior,
                                         siteid, amp);
  
    return Py_BuildValue("dd", logprob, cdf);
  }
  else
  {
    Py_INCREF(Py_None);
    return Py_None;
  }  
}

static PyObject * py_arrtime_logprob(NetModel_t * p_netmodel,
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

  logprob = ArrivalTimePrior_LogProb(&p_netmodel->arr_time_prior,
                                     arrtime, pred_arrtime,
                                     det_deltime, siteid, phaseid);
  
  return Py_BuildValue("d", logprob);
}

static PyObject * py_mean_amplitude(NetModel_t * p_netmodel,
				    PyObject * args)
{
  double mb, depth, ttime;
  int siteid;
  int phaseid;
  double amp;
  
  if (!PyArg_ParseTuple(args, "dddii", &mb, &depth, &ttime,
                        &siteid, &phaseid))
    return NULL;

  
  amp = ArrivalAmplitudePrior_Point(&p_netmodel->arr_amp_prior,
					 mb, depth, ttime,
					 siteid, phaseid);
  
  return Py_BuildValue("d", amp);
}

static PyObject * py_mean_travel_time(NetModel_t * p_netmodel,
                                      PyObject * args)
{
  double evlon, evlat, evdepth;
  int siteid;
  int phaseid;
  double trvtime;

  EarthModel_t * p_earth;
  
  if (!PyArg_ParseTuple(args, "dddii", &evlon, &evlat, &evdepth,
                        &siteid, &phaseid))
    return NULL;

  p_earth = p_netmodel->p_earth;

  trvtime = EarthModel_ArrivalTime(p_earth, evlon, evlat, evdepth, 0, phaseid,
                                   siteid);
  
  trvtime += ArrivalTimePrior_MeanResidual(&p_netmodel->arr_time_prior,
                                           siteid, phaseid);
  
  return Py_BuildValue("d", trvtime);
}

static PyObject * py_arraz_logprob(NetModel_t * p_netmodel,
                                   PyObject * args)
{
  double arraz;
  double pred_arraz;
  double det_delaz;
  int siteid;
  int phaseid;

  double logprob;
  
  if (!PyArg_ParseTuple(args, "dddii", &arraz, &pred_arraz, &det_delaz,
                        &siteid, &phaseid))
    return NULL;

  logprob = ArrivalAzimuthPrior_LogProb(&p_netmodel->arr_az_prior,
                                        arraz, pred_arraz,
                                        det_delaz, siteid, phaseid);
  
  return Py_BuildValue("d", logprob);
}

static PyObject * py_arrslo_logprob(NetModel_t * p_netmodel,
                                    PyObject * args)
{
  double arrslo;
  double pred_arrslo;
  double det_delslo;
  int siteid;
  int phaseid;

  double logprob;
  
  if (!PyArg_ParseTuple(args, "dddii", &arrslo, &pred_arrslo, &det_delslo,
                        &siteid, &phaseid))
    return NULL;

  logprob = ArrivalSlownessPrior_LogProb(&p_netmodel->arr_slo_prior,
                                         arrslo, pred_arrslo,
                                         det_delslo, siteid, phaseid);
  
  return Py_BuildValue("d", logprob);
}

static PyObject * py_arrphase_logprob(NetModel_t * p_netmodel, PyObject * args)
{
  int arrphaseid;
  int phaseid;
  
  double logprob;
  
  if (!PyArg_ParseTuple(args, "ii", &arrphaseid, &phaseid))
    return NULL;
  
  logprob = ArrivalPhasePrior_LogProb(&p_netmodel->arr_phase_prior,
                                      arrphaseid, phaseid);
  
  return Py_BuildValue("d", logprob);
}

static PyObject * py_falsephase_logprob(NetModel_t * p_netmodel, PyObject *args)
{
  int arrphaseid;
  
  double logprob;
  
  if (!PyArg_ParseTuple(args, "i", &arrphaseid))
    return NULL;
  
  logprob = FalseArrivalPhasePrior_LogProb(&p_netmodel->arr_phase_prior,
                                           arrphaseid);
  
  return Py_BuildValue("d", logprob);
}

static PyObject * py_enable_sec_arr(NetModel_t * p_netmodel)
{
  p_netmodel->enable_sec_arr = 1;
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject * py_disable_sec_arr(NetModel_t * p_netmodel)
{
  p_netmodel->enable_sec_arr = 0;
  Py_INCREF(Py_None);
  return Py_None;
}


void convert_events_to_pyobj(const EarthModel_t * p_earth,
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
    ARRAY2(p_eventsobj, i, EV_LON_COL) = p_event->evlon;
    ARRAY2(p_eventsobj, i, EV_LAT_COL) = p_event->evlat;
    ARRAY2(p_eventsobj, i, EV_DEPTH_COL) = p_event->evdepth;
    ARRAY2(p_eventsobj, i, EV_TIME_COL) = p_event->evtime;
    ARRAY2(p_eventsobj, i, EV_MB_COL) = p_event->evmag;
    ARRAY2(p_eventsobj, i, EV_ORID_COL) = (double) p_event->orid;

    p_detlistobj = PyList_New(0);
    
    /* copy over the (phaseid, detnum) of the event */
    numsites = EarthModel_NumSites(p_earth);
    numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
    
    for (siteid = 0; siteid < numsites; siteid ++)
    {
      for (phaseid = 0; phaseid < numtimedefphases; phaseid ++)
      {
        int numdet;
        
        numdet = p_event->p_num_dets[siteid * numtimedefphases + phaseid];

        if (numdet > 0)
        {
          int pos;
          PyObject * p_phase_det_obj;
          
          /* first the phase and then the detnums */
          p_phase_det_obj = PyTuple_New(numdet + 1);
          
          /* tuple set_item steals a reference so we don't need to decr it */
          PyTuple_SetItem(p_phase_det_obj, 0, Py_BuildValue("i", phaseid));
          
          for (pos=0; pos<numdet; pos++)
          {
            int detnum;
            detnum = p_event->p_all_detids[siteid * numtimedefphases 
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

Event_t * alloc_event_net(NetModel_t * p_netmodel)
{
  Event_t * p_event;
  int numsites;
  int numtimedefphases;
  
  p_event = (Event_t *) calloc(1, sizeof(*p_event));
  
  numsites = EarthModel_NumSites(p_netmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_netmodel->p_earth);
 
  p_event->p_all_detids = (int *) malloc(numsites * numtimedefphases *
                                         MAX_PHASE_DET *
                                         sizeof(*p_event->p_all_detids));
  
  /* allocate and initialize the number of detections to 0 */
  p_event->p_num_dets = (int *) calloc(numsites * numtimedefphases,
                                       sizeof(*p_event->p_num_dets));

  return p_event;
}

void copy_event_net(NetModel_t * p_netmodel, Event_t * p_tgt_event,
		    const Event_t * p_src_event)
{
  int * p_tgt_all_detids;
  int * p_tgt_num_dets;
  int numsites;
  int numtimedefphases;
  
  numsites = EarthModel_NumSites(p_netmodel->p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_netmodel->p_earth);
 
  /* save the detids pointer */
  p_tgt_all_detids = p_tgt_event->p_all_detids;
  p_tgt_num_dets = p_tgt_event->p_num_dets;
  
  /* copy the event */
  *p_tgt_event = *p_src_event;

  /* restore the detids pointer */
  p_tgt_event->p_all_detids = p_tgt_all_detids;
  p_tgt_event->p_num_dets = p_tgt_num_dets;
  
  /* copy the detids */
  memcpy(p_tgt_event->p_num_dets, p_src_event->p_num_dets,
         numsites * numtimedefphases * sizeof(*p_src_event->p_num_dets));
  memcpy(p_tgt_event->p_all_detids, p_src_event->p_all_detids,
         numsites * numtimedefphases * MAX_PHASE_DET 
         * sizeof(*p_src_event->p_all_detids));
}

void print_event_detections(EarthModel_t * p_earth, const Event_t * p_event)
{
  int detcnt;
  int numsites;
  int numtimedefphases;
  int siteid;
  int phaseid;
  
  numsites = EarthModel_NumSites(p_earth);
  numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);
  
  detcnt = 0;
  
  for (siteid = 0; siteid < numsites; siteid ++)
  {
    for (phaseid = 0; phaseid < numtimedefphases; phaseid ++)
    {
      int numdet;
      
      numdet = p_event->p_num_dets[siteid * numtimedefphases + phaseid];

      if (numdet > 0)
      {
        int pos;
        
        for (pos = 0; pos < numdet; pos ++)
        {
          int detnum = p_event->p_all_detids[siteid * numtimedefphases 
                                             * MAX_PHASE_DET 
                                             + phaseid * MAX_PHASE_DET + pos];
          if (!detcnt)
          printf("[");
          printf("%d ", detnum);
          detcnt ++;
        }
      }
    }
  }

  if (detcnt)
    printf("]\n");
}

