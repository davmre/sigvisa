#ifndef SIGNAL_MODEL_UTIL
#define SIGNAL_MODEL_UTIL

#include "signal_structures.h"

ArrivalWaveform_t * insert_st(ArrivalWaveform_t * p_head, 
			      ArrivalWaveform_t * p_arr);
ArrivalWaveform_t * insert_et(ArrivalWaveform_t * p_head, 
			      ArrivalWaveform_t * p_arr);
ArrivalWaveform_t * remove_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr);
ArrivalWaveform_t * append_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr);

void copy_AR_process(ARProcess_t *dest, ARProcess_t *src);
void free_AR_process(ARProcess_t *ar);
ArrivalWaveform_t * free_ArrivalWaveform(ArrivalWaveform_t * a);

#endif
