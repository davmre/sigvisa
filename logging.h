#ifndef INCLUDE_LOGGING_H
#define INCLUDE_LOGGING_H

#include <liblogger/liblogger_levels.h>
// possible log levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL 
#define LOG_LEVEL LOG_LEVEL_INFO
#define LOG_MODULE_NAME "sigvisa"
#include <liblogger/liblogger.h>

#include "sigvisa.h"

char * signal_str(Channel_t * signal);
int print_signal(Channel_t * signal);

char * arrival_str(const Arrival_t * p_arr);
void print_arrival(const Arrival_t * p_arr);

char * event_str(const Event_t * p_event);
void print_event(const Event_t * p_event);

void print_vector(int n, double * vector);

int save_pdf_plot(SigModel_t * p_sigmodel, Channel_t * p_signal, int band, char * filename, char * format);

void log_segments_events(SigModel_t * p_sigmodel, PyObject * log_segment_cb,int numevents, const Event_t ** pp_events, double max_start_time, PyObject * text);

#endif
