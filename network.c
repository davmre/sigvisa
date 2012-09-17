#include <math.h>
#include <stdlib.h>

#define SIGVISA_NETWORK
#include "sigvisa.h"


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


Event_t ** indirect_event_list(int numevents, Event_t * evlist) {

  Event_t ** ilist = calloc(numevents, sizeof(Event_t *));

  for (int i=0; i < numevents; ++i) {
    ilist[i] = &evlist[i];
  }
  return ilist;
}




void convert_eventobj_no_det(PyArrayObject * p_events_arrobj,
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

PyObject * py_set_fake_detections(SigModel_t *p_sigmodel, PyObject *args) {
  PyObject * p_detlist_obj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &p_detlist_obj))
    return NULL;


  int n = convert_fake_detections(p_detlist_obj, &p_sigmodel->p_detections);
  p_sigmodel->numdetections = n;

  return Py_BuildValue("n", n);
}




void convert_event_to_pyobj(const Event_t * p_event, PyObject * p_eventsobj, int i) {
  ARRAY2(p_eventsobj, i, EV_LON_COL) = p_event->evlon;
  ARRAY2(p_eventsobj, i, EV_LAT_COL) = p_event->evlat;
  ARRAY2(p_eventsobj, i, EV_DEPTH_COL) = p_event->evdepth;
  ARRAY2(p_eventsobj, i, EV_TIME_COL) = p_event->evtime;
  ARRAY2(p_eventsobj, i, EV_MB_COL) = p_event->evmag;
  ARRAY2(p_eventsobj, i, EV_ORID_COL) = (double) p_event->orid;
  //  ARRAY2(p_eventsobj, i, EV_SCORE_COL) = p_event->evscore;
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
      (p_event->p_arrivals + i*numtimedefphases + j)->phaseid = j+1;
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
