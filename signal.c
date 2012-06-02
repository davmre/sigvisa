#include <math.h>
#include <stdlib.h>

#include "sigvisa.h"


/* ****************************************************************
   Lookup functions for band and channel names/numbers
   ****************************************************************
*/

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


PyObject * py_canonical_channel_num(PyObject * self, PyObject * args) {
  PyObject * py_chan_str;
  if (!PyArg_ParseTuple(args, "O!", &PyString_Type, &py_chan_str))
    return NULL;

   int result = canonical_channel_num(PyString_AsString(py_chan_str));
   return Py_BuildValue("n", result);
 }

PyObject * py_canonical_band_num(PyObject * self, PyObject * args) {
  PyObject * py_band_str;
  if (!PyArg_ParseTuple(args, "O!", &PyString_Type, &py_band_str))
    return NULL;

   int result = canonical_band_num(PyString_AsString(py_band_str));
   return Py_BuildValue("n", result);
 }





/* ****************************************************************
   Constructors and Destructors 
   ****************************************************************
*/
Trace_t * alloc_trace() {
  Trace_t * p_trace = calloc(1, sizeof(Trace_t));
  p_trace->hz = DEFAULT_HZ;
  return p_trace;
}

void free_trace(Trace_t * p_trace) {
  /* if there is a Python array that still refers to this trace data,
     we make it responsible for freeing the data later. */
  if (p_trace->py_array != NULL) {
    p_trace->py_array->flags |= NPY_OWNDATA;
  } else if (p_trace->p_data != NULL) {
    free(p_trace->p_data);
  }

  free(p_trace);
}

Channel_t * alloc_channel(Segment_t * p_segment) {
  Channel_t * p_channel = (Channel_t *) calloc(1, sizeof(Channel_t));
  p_channel->start_time = p_segment->start_time;
  p_channel->len = p_segment->len;
  p_channel->hz = p_segment->hz;
  p_channel->siteid = p_segment->siteid;
  return p_channel;
}

void alloc_segment_inner(Segment_t * p_segment) {
  for (int c=0; c < NUM_CHANS; ++c) {
    p_segment->p_channels[c] = calloc(1, sizeof(Channel_t));
    Channel_t * p_channel = p_segment->p_channels[c];
    p_channel->start_time = p_segment->start_time;
    p_channel->len = p_segment->len;
    p_channel->hz = p_segment->hz;
    p_channel->siteid = p_segment->siteid;
    p_channel->chan = c;
    for (int b=0; b < NUM_BANDS; ++b) {
      p_channel->p_bands[b] = alloc_trace();
      Trace_t * p_band = p_channel->p_bands[b];
      p_band->start_time = p_segment->start_time;
      p_band->len = p_segment->len;
      p_band->hz = p_segment->hz;
      p_band->siteid = p_segment->siteid;
      p_band->chan = c;
      p_band->band = b;
      p_band->p_data = calloc(p_band->len, sizeof(double));
      if(p_band->p_data == NULL) {
	LogError("memory allocation failed!");
	exit(EXIT_FAILURE);
      }
    }
  }
}

double Segment_EndTime(Segment_t * b) {
  return b->start_time + (b->len / b->hz);
}

void free_segment_inner(Segment_t * p_segment) {
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


  *pp_trace = calloc(1, sizeof(Trace_t));

  PyArrayObject * py_array = (PyArrayObject *) PyArray_ContiguousFromAny(py_data, NPY_DOUBLE, 1,2);
  Py_INCREF(py_array);
  Py_DECREF(py_data);
  CHECK_ERROR;
  (*pp_trace)->p_data = (double *) (py_array)->data;
  (*pp_trace)->py_array = py_array;
  (*pp_trace)->len = PyArray_SIZE(py_array);

  PyObject * stats = PyObject_GetAttrString(py_trace, "stats");

  pydict_get_double(stats, "starttime_unix", & (*pp_trace)->start_time);
  pydict_get_double(stats, "sampling_rate", & (*pp_trace)->hz);
  pydict_get_int(stats, "siteid", (long*)&(*pp_trace)->siteid);

  char * chan_str;
  pydict_get_string(stats, "channel", &chan_str);
  (*pp_trace)->chan = canonical_channel_num(chan_str);

  char * band_str;
  pydict_get_string(stats, "band", &band_str);
  (*pp_trace)->band = canonical_band_num(band_str);

  pydict_get_double(stats, "noise_floor", & (*pp_trace)->noise_floor);
  pydict_get_double(stats, "p_time", & (*pp_trace)->p_time);
  pydict_get_double(stats, "s_time", & (*pp_trace)->s_time);
  pydict_get_int(stats, "p_phaseid", (long*)& (*pp_trace)->p_phaseid);
  pydict_get_int(stats, "s_phaseid", (long*)& (*pp_trace)->s_phaseid);

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
      free_segment_inner(new_segment);
      free(new_segment);
    }
  }

  *pp_segments = realloc(*pp_segments, idx*sizeof(Segment_t));
  return idx;
}

PyObject * py_set_signals(SigModel_t *p_sigmodel, PyObject *args) {
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

PyObject * build_trace(Trace_t * p_trace) {

  // a trace object contains two members: data, a numpy array, and stats, a python dict.
   npy_intp dims[1];
   dims[0] = p_trace->len;

   PyObject * py_data;
   if (p_trace->py_array == NULL) {
     py_data = (PyObject *)PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, p_trace->p_data);
     p_trace->py_array = py_data;
   } else {
     py_data = (PyObject *)p_trace->py_array;
   }

   PyObject * py_stats = PyDict_New();

   PyObject * key = PyString_FromString("npts");
   PyObject * value = Py_BuildValue("n", p_trace->len);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("starttime_unix");
   value = Py_BuildValue("d", p_trace->start_time);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("sampling_rate");
   value = Py_BuildValue("d", p_trace->hz);
   assert(key != NULL && value != NULL);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("siteid");
   value = Py_BuildValue("n", p_trace->siteid);
   assert(key != NULL && value != NULL);
   PyDict_SetItem(py_stats, key, value);
   Py_DECREF(key);
   Py_DECREF(value);

   key = PyString_FromString("channel");
   value = canonical_channel_name(p_trace->chan);
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
      PyObject * py_band_trace = build_trace(p_channel->p_bands[j]);
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


PyObject * py_get_signals(SigModel_t *p_sigmodel, PyObject *args) {
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


/* 
   ****************************************************************
   Signal Synthesis
   ****************************************************************
*/

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


PyObject * py_synthesize_signals(SigModel_t *p_sigmodel, PyObject *args) {
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


PyObject * py_synthesize_signals_det(SigModel_t *p_sigmodel, PyObject *args) {
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

