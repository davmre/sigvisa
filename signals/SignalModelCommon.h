#ifndef SIGNAL_MODEL_COMMON
#define SIGNAL_MODEL_COMMON

#include <Python.h>
#include "../sigvisa.h"


typedef void(*SMSetParams_f)(void * pv_params, int station, PyObject * py_dict);

typedef int(*SMHasModel_f)(void * pv_model, int siteid, int chan);

typedef double(*SMLikelihood_f)(void * pv_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals);

typedef void(*SMSample_f)(void * pv_params, EarthModel_t * p_earth, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int samplePerturb, int sampleNoise);

typedef void(*SMUnInit_f)(void * pv_params);

typedef struct SignalModel_t
{
  SMSetParams_f set_params;
  SMHasModel_f has_model;
  SMLikelihood_f likelihood;
  SMSample_f sample;
  SMUnInit_f uninit;

  void * pv_params;

} SignalModel_t;

void init_signal_model(SignalModel_t * p_model, char * model_name, int numsites);
void uninit_signal_model(SignalModel_t * p_model);


void arrival_list(EarthModel_t * p_earth, int siteid, double min_time, double max_time, int num_events, const Event_t ** pp_events, int * num_arrivals, Arrival_t *** ppp_arrivals);
void det_arrivals(void * p_sigmodel_v, Segment_t * p_segment, int * num_arrivals, Arrival_t *** ppp_arrivals);

double Signal_Score_Event_Site(void * pv_sigmodel, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events);

double Signal_Score_Event(void * pv_sigmodel, const Event_t * event, int num_other_events, const Event_t ** pp_other_events);

PyObject * py_set_params(void * pv_sigmodel, PyObject * args);
PyObject * py_set_all_params(void * pv_sigmodel, PyObject * args);
PyObject * py_arr_likelihood(void * pv_sigmodel, PyObject * args);

double det_likelihood(void * pv_sigmodel, int write_log);
PyObject * py_det_likelihood(void * pv_sigmodel, PyObject * args);

double Signal_Score_Event(void * pv_sigmodel, const Event_t * p_event, int num_other_events, const Event_t ** pp_other_events);
double Signal_Score_Event_Site(void * pv_sigmodel, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events);

#endif
