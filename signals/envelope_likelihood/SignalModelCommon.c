#include "../../sigvisa.h"

void init_signal_model(SignalModel_t * p_model, char * model_name, int numsites) {


  if (strcmp(model_name, "spectral_envelope") == 0) {
    p_model->set_params = (SMSetParams_f)&Spectral_Envelope_Model_Set_Params;
    p_model->has_model = (SMHasModel_f)&Spectral_Envelope_Model_Has_Model;
    p_model->likelihood = (SMLikelihood_f)&Spectral_Envelope_Model_Likelihood;
    p_model->sample =  (SMSample_f)&Spectral_Envelope_Model_Sample;
    p_model->uninit = (SMUnInit_f)&Spectral_Envelope_Model_UnInit;

    p_model->pv_params = calloc(1, sizeof(Spectral_Envelope_Model_t));

    Spectral_Envelope_Model_Init_Params(p_model->pv_params,  numsites);
  } else {
    LogError("unrecognized signal model %s", model_name);
    exit(EXIT_FAILURE);
  }

}

void uninit_signal_model(SignalModel_t * p_model) {

  p_model->uninit(p_model->pv_params);
  free(p_model->pv_params);
}

void evt_arrival_times(const Event_t * p_event, int siteid, int numtimedefphases, double * first_arrival, double *last_arrival) {
  *first_arrival = DBL_MAX;
  *last_arrival = DBL_MIN;
  //  printf("called with siteid %d\n", siteid);
  fflush(stdout);
  for (int i=0; i < MAX_PHASE(numtimedefphases); ++i) {
    if (!USE_PHASE(i)) continue;
    double phase_arr_time = (p_event->p_arrivals + (siteid-1)*numtimedefphases + i)->time;
    if (phase_arr_time <= 0) continue;
    if (phase_arr_time < *first_arrival) {
      *first_arrival = phase_arr_time;
    }
    if (phase_arr_time > *last_arrival) {
      *last_arrival = phase_arr_time;
    }
  }
}

const Event_t ** augment_events(int numevents, const Event_t ** events, const Event_t *event) {
  const Event_t ** augmented =  (const Event_t **)malloc((numevents+1)* sizeof(Event_t *));
  CHECK_PTR(augmented);
  memcpy(augmented, events, numevents*sizeof(Event_t *));
  augmented[numevents] = event;
  return augmented;
}

/* Return the score for this event: the ratio of signal likelihoods
   between a world where this event exists, and one where it
   doesn't. */
double Signal_Score_Event_Site(SigModel_t * p_sigmodel, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events) {

  SignalModel_t * p_model = &p_sigmodel->signal_model;

  double score = 0;
  Segment_t * p_segment;

  int numtimedefphases = EarthModel_NumTimeDefPhases(p_sigmodel->p_earth);

  for (int i=0; i < p_sigmodel->numsegments; ++i) {

    p_segment = p_sigmodel->p_segments + i;

    if (p_segment->siteid != siteid)  {
      continue;
    }
    //printf("scoring event at siteid %d...\n", siteid);
    /* compute the time period during which the event will affect the station */
    double first_envelope_time, last_envelope_time;
    evt_arrival_times(p_event, siteid, numtimedefphases, &first_envelope_time, &last_envelope_time);
    last_envelope_time += MAX_ENVELOPE_LENGTH;

    /* if this trace doesn't fall within that period, skip it */
    if (p_segment->start_time > last_envelope_time || Segment_EndTime(p_segment) < first_envelope_time) {
      //printf("     skipping signal segment %d: first env %lf last env %lf seg start %lf seg end %lf\n", i, first_envelope_time, last_envelope_time, p_segment->start_time, Segment_EndTime(p_segment));
      continue;
    }

    /* we compute scores for the background event set, and for an
       augmented event set which includes the specified event. */

    int num_basic_arrivals, num_augmented_arrivals;
    Arrival_t ** pp_basic_arrivals;
    Arrival_t ** pp_augmented_arrivals;
    arrival_list(p_sigmodel->p_earth, siteid, p_segment->start_time, Segment_EndTime(p_segment), num_other_events, pp_other_events, &num_basic_arrivals, &pp_basic_arrivals);

    const Event_t ** augmented_events = augment_events(num_other_events, pp_other_events, p_event);
    arrival_list(p_sigmodel->p_earth, siteid, p_segment->start_time, Segment_EndTime(p_segment), num_other_events+1, augmented_events, &num_augmented_arrivals, &pp_augmented_arrivals);

    double event_lp , background_lp;
    event_lp = p_model->likelihood(p_sigmodel, p_segment, num_augmented_arrivals, (const Arrival_t **)pp_augmented_arrivals);
    background_lp = p_model->likelihood(p_sigmodel, p_segment, num_basic_arrivals, (const Arrival_t **)pp_basic_arrivals);
    free(augmented_events);
    free(pp_basic_arrivals);
    free(pp_augmented_arrivals);

    score += (event_lp - background_lp);

    //LogInfo("   segment %d siteid %d contributed score %lf = event_lp %lf - background_lp %lf (perturb %d)", i, p_segment->siteid, event_lp - background_lp, event_lp, background_lp, p_sigmodel->ar_perturbation);

  }

  return score;

}

/* Return the score for this event: the ratio of signal likelihoods
   between a world where this event exists, and one where it
   doesn't. */
double Signal_Score_Event(SigModel_t * p_sigmodel, const Event_t * p_event, int num_other_events, const Event_t ** pp_other_events) {


  int numsites = EarthModel_NumSites(p_sigmodel->p_earth);

  double score = 0;
  for (int siteid = 1; siteid <= numsites; ++siteid) {
    score += Signal_Score_Event_Site(p_sigmodel, p_event,
					  siteid, num_other_events,
					  pp_other_events);
  }

  return score;
}



/* given a sigmodel with a set of detections, and a signal segment
   from some station, creates a list of those arrivals detected at
   that station within the time period of the segment*/
void det_arrivals(SigModel_t * p_sigmodel, Segment_t * p_segment, int * num_arrivals, Arrival_t *** ppp_arrivals) {


  // for each segment, compute a list of arrivals

  *num_arrivals = 0;
  int num_alloced = 20;
  *ppp_arrivals = calloc(num_alloced, sizeof(Arrival_t *));
  CHECK_PTR(*ppp_arrivals);

  for (int d = 0; d < p_sigmodel->numdetections; ++d) {
    Detection_t * p_det = p_sigmodel->p_detections + d;

    if (p_segment != NULL) {
      if (p_det->site_det != p_segment->siteid-1) continue;
      if (p_det->time_det + MAX_ENVELOPE_LENGTH < p_segment->start_time) continue;
      if (p_det->time_det > Segment_EndTime(p_segment)) continue;
    }

    if (++(*num_arrivals) > num_alloced) {
      num_alloced *= 2;
      *ppp_arrivals = realloc(*ppp_arrivals, num_alloced * sizeof(Arrival_t *));
      CHECK_PTR(*ppp_arrivals);
    }

    Arrival_t * p_arr = calloc(1, sizeof(Arrival_t));
    *(*ppp_arrivals+(*num_arrivals)-1) = p_arr;
    p_arr->time = p_det->time_det;
    p_arr->amp = p_det->amp_det;
    p_arr->azi = p_det->azi_det;
    p_arr->slo = p_det->slo_det;
    p_arr->phaseid = p_det->phase_det+1;
    p_arr->siteid = p_det->site_det+1;
  }
}


double det_likelihood(SigModel_t * p_sigmodel, int write_log) {



  SignalModel_t * p_model = &p_sigmodel->signal_model;


  double ll = 0;


  LogTrace("called det_likelihood on %d segments", p_sigmodel->numsegments);

  for (int i=0; i < p_sigmodel->numsegments; ++i) {

    Segment_t * p_segment = p_sigmodel->p_segments + i;


    // for each segment, compute a list of arrivals
    int num_arrivals=0;
    Arrival_t ** pp_arrivals;
    det_arrivals(p_sigmodel, p_segment, &num_arrivals, &pp_arrivals);


    /* ------------ begin logging ------ */
    if(write_log) {

     char desc[50];
    snprintf(desc, 50, "real_signal_%d", i);
    //save_pdf_plot(p_sigmodel, p_segment->p_channels[CHAN_BHZ], desc, "g-");
    Segment_t * pred_segment = calloc(1, sizeof(Segment_t));
    pred_segment->start_time = p_segment->start_time;
    pred_segment->hz = p_segment->hz;
    pred_segment->siteid = p_segment->siteid;
    pred_segment->len = p_segment->len;
    p_model->sample(p_model->pv_params,
		    p_sigmodel->p_earth,
		    pred_segment,
		    num_arrivals, (const Arrival_t **)pp_arrivals,
		    0, 0);
    snprintf(desc, 50, "pred_signal_%d", i);
    //save_pdf_plot(p_sigmodel, pred_segment->p_channels[CHAN_BHZ], desc, "r-");
    }

    /* ------------ end logging ------ */

    double seg_ll = p_model->likelihood(p_sigmodel, p_segment, num_arrivals, (const Arrival_t **)pp_arrivals);
    // double seg_ll_iid = segment_likelihood_iid(p_sigmodel, p_segment, num_arrivals, (const Arrival_t **)pp_arrivals);
    assert(!isnan(seg_ll) && seg_ll > -1*DBL_MAX);

    LogTrace("segment site %d start %lf contributed ll %lf from %d arrivals", p_segment->siteid, p_segment->start_time, seg_ll, num_arrivals);

    ll += seg_ll;

    for(int i=0; i < num_arrivals; ++i) {
      free(*(pp_arrivals+i));
     }
    free(pp_arrivals);
  }

  return ll;
}

PyObject * py_det_likelihood(SigModel_t * p_sigmodel, PyObject * args) {

  int write_log;

  if (!PyArg_ParseTuple(args, "i", &write_log))
    return NULL;

  double result = det_likelihood(p_sigmodel, write_log);
  CHECK_ERROR;


  return Py_BuildValue("d", result);
}



/* given a list of events, extract the arrivals relevant to a particular siteid within a particular time range */
void arrival_list(EarthModel_t * p_earth, int siteid, double min_time, double max_time, int num_events, const Event_t ** pp_events, int * num_arrivals, Arrival_t *** ppp_arrivals) {

  int numtimedefphases = EarthModel_NumTimeDefPhases(p_earth);

  *num_arrivals = 0;
  *ppp_arrivals = calloc(num_events*MAX_PHASE(numtimedefphases), sizeof(Arrival_t *));

  for (int i=0; i < num_events; ++i) {
    const Event_t * p_event = *(pp_events+i);
    for (int j=0; j < MAX_PHASE(numtimedefphases); ++j) {
      if (!USE_PHASE(j)) continue;
      Arrival_t * p_arr = p_event->p_arrivals + (siteid-1)*numtimedefphases + j;

      //      LogInfo("testing arrival %s", arrival_str(p_arr));
      if (p_arr->amp == 0 || p_arr->time <= 0) {
	continue;
      }
      if (p_arr->time < min_time - MAX_ENVELOPE_LENGTH && p_arr->time > max_time) {
	continue;
      }
      *(*ppp_arrivals + (*num_arrivals)++) = p_arr;
    }
  }
}


PyObject * py_set_params(SigModel_t * p_sigmodel, PyObject * args) {

  SignalModel_t * p_model = &p_sigmodel->signal_model;

  int siteid;
  PyObject * py_dict;

  if (!PyArg_ParseTuple(args, "iO", &siteid, &py_dict))
    return NULL;

  p_model->set_params(p_model->pv_params, siteid, py_dict);
  Py_RETURN_TRUE;
}

PyObject * py_set_all_params(SigModel_t * p_sigmodel, PyObject * args) {

  SignalModel_t * p_model = &p_sigmodel->signal_model;

  PyObject * py_dict;

  if (!PyArg_ParseTuple(args, "O", &py_dict))
    return NULL;

  PyObject *py_key, *py_value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(py_dict, &pos, &py_key, &py_value)) {
    int siteid = PyInt_AsLong(py_key);
    p_model->set_params(p_model->pv_params, siteid, py_value);
  }

  Py_RETURN_TRUE;
}