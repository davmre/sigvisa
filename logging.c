
#include <stdio.h>
#include "sigvisa.h"


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


char * arrival_str(const Arrival_t * p_arr) {
  char * str = malloc(100*sizeof(char));
  snprintf(str, 100, "time %.4lf amp %.4lf azi %.4lf slo %.4lf %d %d",
	   p_arr->time, p_arr->amp, p_arr->azi, p_arr->slo, 
	   p_arr->phase, p_arr->siteid);
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

void print_vector(int n, double * vector) {
  for(int i=0; i < n; ++i) {
    if (i == 0 || vector[i] != vector[i-1])
      fprintf(stdout, "%d %lf \n", i, vector[i]);
  }
}

int save_pdf_plot(SigModel_t * p_sigmodel, Signal_t * p_signal, char * filename, char * format) {

  PyObject * p_trace;

  signal_to_trace(p_signal, &p_trace);
  PyObject * py_text = Py_BuildValue("s", filename);
  PyObject * py_format = Py_BuildValue("s", format);

  PyObject * retval = PyObject_CallFunction(p_sigmodel->log_trace_cb, 
					    "OOO", 
					    p_trace,
					    py_text, py_format);
  return (retval != NULL);
}


void log_segments_events(SigModel_t * p_sigmodel, PyObject * log_segment_cb, int numevents, const Event_t ** pp_events, double max_start_time, PyObject *py_text)
{
  PyObject * retval;
  PyObject * eventsobj;
  PyObject * evarrlistobj;
  int i;
  
  if(py_text == NULL) {
    py_text = Py_BuildValue("s", "");
  }

  convert_events_arrs_to_pyobj(p_sigmodel, p_sigmodel->p_earth, 
			       pp_events,numevents, 
			       &eventsobj, &evarrlistobj);
  
  for (i = 0; i < p_sigmodel->numsegments; ++i) {
    LogTrace("logging segment %d", i);

    ChannelBundle_t * p_real_segment = p_sigmodel->p_segments + i;
    if (p_real_segment->start_time > max_start_time) {
      continue;
    }

    ChannelBundle_t * p_pred_segment = calloc(1, sizeof(ChannelBundle_t));
    memcpy(p_pred_segment, p_real_segment, sizeof(ChannelBundle_t));


    int num_arrivals;
    Arrival_t ** pp_arrivals;
    arrival_list(p_sigmodel->p_earth, p_pred_segment->siteid, p_pred_segment->start_time, ChannelBundle_EndTime(p_pred_segment), numevents, pp_events, &num_arrivals, &pp_arrivals);


    /*SignalPrior_SampleThreeAxisAR(&p_sigmodel->sig_prior,
				  p_sigmodel->p_earth,
				  0, 0,
				  num_arrivals, pp_arrivals,
				  p_pred_segment);*/

    LogError("todo: fix logging");
    exit(EXIT_FAILURE);
    free(pp_arrivals);

    PyObject * real_trace, * pred_trace;
    real_trace = channel_bundle_to_trace_bundle(p_real_segment);
    pred_trace = channel_bundle_to_trace_bundle(p_pred_segment);
    retval = PyObject_CallFunction(log_segment_cb, "OOOOO", 
				   eventsobj, evarrlistobj,
				   real_trace,
				   pred_trace,
				   py_text);

    if (!retval) {
      LogError("log_segment_cb call failed!");
      CHECK_ERROR;
    } else {
      Py_DECREF(retval);
    }

    Py_DECREF(real_trace);
    Py_DECREF(pred_trace);
  }
 
  Py_DECREF(eventsobj);
  Py_DECREF(evarrlistobj);
  
}

static void log_segments(SigModel_t * p_sigmodel, World_t * p_world)
{
  int numevents, i;
  double maxtime;
  double end_time = p_sigmodel->end_time;
  double start_time = p_sigmodel->start_time;

  if (p_world->high_evtime < end_time)
    maxtime = MAX(p_world->low_evtime - MAX_TRAVEL_TIME,
                  start_time);
  else
    maxtime = p_world->high_evtime;

  numevents = 0;
  for (i=p_world->write_evnum;
       (i<p_world->high_evnum) && p_world->pp_events[i]->evtime < maxtime;
       i++)
    numevents ++;

  log_segments_events(p_sigmodel, p_world->log_segment_cb, numevents, (const Event_t **) p_world->pp_events + p_world->write_evnum, maxtime + MAX_TRAVEL_TIME, NULL);

}
