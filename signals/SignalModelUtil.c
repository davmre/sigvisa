#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../sigvisa.h"
#include "SignalModelUtil.h"


ArrivalWaveform_t * append_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_active=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->next_active == NULL) {
    // if at the end of the list, add to the end
    p_head->next_active = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_active = append_active(p_head->next_active, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * remove_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  if (p_head == NULL) {
    p_new_head = NULL;
  } else if (p_head == p_arr) {
    p_new_head = p_arr->next_active;
  } else if (p_head->next_active == NULL) {
    p_new_head = p_head;
  } else {
    // otherwise, delete recursively
    p_head->next_active = remove_active(p_head->next_active, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * insert_st(ArrivalWaveform_t * p_head, 
			      ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_start=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->start_time > p_arr->start_time ) {
    // if the new addition comes before the head, make it the new head
    p_arr->next_start = p_head;
    p_new_head = p_arr;
  } else if (p_head->next_start == NULL) {
    // if at the end of the list, add to the end
    p_head->next_start = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_start = insert_st(p_head->next_start, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

ArrivalWaveform_t * insert_et(ArrivalWaveform_t * p_head, 
				 ArrivalWaveform_t * p_arr) {
  ArrivalWaveform_t * p_new_head;
  p_arr->next_end=NULL;
  if (p_head == NULL) {
    p_new_head = p_arr;
  } else if (p_head->end_time > p_arr->end_time ) {
    // if the new addition comes before the head, make it the new head
    p_arr->next_end = p_head;
    p_new_head = p_arr;
  } else if (p_head->next_end == NULL) {
    // if at the end of the list, add to the end
    p_head->next_end = p_arr;
    p_new_head = p_head;
  } else {
    // otherwise, insert recursively
    p_head->next_end = insert_et(p_head->next_end, p_arr);
    p_new_head = p_head;
  }
  return p_new_head;
}

void copy_AR_process(ARProcess *dest, ARProcess *src) {
  memcpy(dest, src, sizeof(ARProcess));
  dest->params = calloc(dest->order, sizeof(double));
  memcpy(dest->params, src->params);
}

void free_AR_process(ARProcess *ar) {
  if (ar->params != NULL) free(ar->params);
  free(ar);
}

// frees an ArrivalWaveform, and returns the next in the start_time linked list
ArrivalWaveform * aw free_ArrivalWaveform(ArrivalWaveform * a) {
  if(a->p_abstract_trace != NULL) {
    free_trace(a->p_abstract_trace);
  }
  if (ar->params != NULL) free(ar->params);

  ArrivalWaveform_t * next_a = a->next_start;
  free(a);
  return next_a;
}


/* populate two linked lists, storing waveform info sorted by
   start_time and end_time respectively */
void init_ArrivalWaveforms(BandModel_t * p_band, int hz, int num_arrivals, Arrival_t ** pp_arrivals, ARWLists_t * arw) {
  for (int i=0; i < num_arrivals; ++i) {

    const Arrival_t * p_arr = *(pp_arrivals + i);

    if (p_arr->amp == 0 || p_arr->time <= 0) continue;

    ArrivalWaveform_t * w = calloc(1, sizeof(ArrivalWaveform_t));
    w->start_time = p_arr->time;
    w->idx = -1; // initialize to -1 since we will increment before the first use

    // for each arrival, get the predicted log-envelope
    w->p_abstract_trace = calloc(1, sizeof(Trace));
    w->p_abstract_trace->hz = hz;
    abstract_spectral_logenv(p_arr, w->p_abstract_trace);

    w->end_time = w->start_time + (double) w->len / hz;

    copy_AR_process(w->ar_process, p_band->wiggle_process);

        double iangle;
    if(!slowness_to_iangle(p_arr->slo, p_arr->phase, &iangle)) {
      //LogTrace("iangle conversion failed from slowness %lf phaseid %d, setting default iangle 45.", p_arr->slo, phase);
      iangle = 45;
    }

    // TODO: FIX COEFFS
    w->bhe_coeff = fabs(SPHERE2X(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->bhn_coeff = fabs(SPHERE2Y(p_arr->azi, iangle)) / fabs(SPHERE2Z(p_arr->azi, iangle));
    w->bhz_coeff = 1;

    arw->st_head = insert_st(arw->st_head, w);
    arw->et_head = insert_et(arw->et_head, w);

  }

  arw->st_ptr = arw->st_head;
  arw->et_ptr = arw->et_head;
  arw->active_arrivals = NULL;
  arw->n_active = 0;

}
