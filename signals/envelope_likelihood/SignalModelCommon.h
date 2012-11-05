#ifndef SIGNAL_MODEL_COMMON
#define SIGNAL_MODEL_COMMON

#include <Python.h>
#include "../../sigvisa.h"

void init_signal_model(SignalModel_t * p_model, char * model_name, int numsites);
void uninit_signal_model(SignalModel_t * p_model);

void arrival_list(EarthModel_t * p_earth, int siteid, double min_time, double max_time, int num_events, const Event_t ** pp_events, int * num_arrivals, Arrival_t *** ppp_arrivals);

double Signal_Score_Event_Site(SigModel_t * p_sigmodel, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events);

double Signal_Score_Event(SigModel_t * p_sigmodel, const Event_t * event, int num_other_events, const Event_t ** pp_other_events);

PyObject * py_set_params(SigModel_t * p_sigmodel, PyObject * args);
PyObject * py_set_all_params(SigModel_t * p_sigmodel, PyObject * args);


double Signal_Score_Event(SigModel_t * p_sigmodel, const Event_t * p_event, int num_other_events, const Event_t ** pp_other_events);
double Signal_Score_Event_Site(SigModel_t * p_sigmodel, const Event_t * p_event, int siteid,int num_other_events, const Event_t ** pp_other_events);

#endif
