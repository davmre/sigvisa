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

#ifndef NETVISA_INCLUDE
#define NETVISA_INCLUDE

#include <Python.h>
/* we don't want _import_array defined in all files except the main
 * module of netvisa */
#if !defined NETVISA_MAIN_MODULE
  #define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"

#include "priors/NumEventPrior.h"
#include "priors/EventLocationPrior.h"
#include "priors/EventMagPrior.h"
#include "priors/EventDetectionPrior.h"
#include "priors/ArrivalTimePrior.h"
#include "priors/NumFalseDetPrior.h"
#include "priors/ArrivalAzimuthPrior.h"
#include "priors/ArrivalSlownessPrior.h"
#include "priors/ArrivalPhasePrior.h"
#include "priors/ArrivalSNRPrior.h"
#include "priors/ArrivalAmplitudePrior.h"
#include "priors/EarthModel.h"
#include "priors/Laplace.h"
#include "priors/Poisson.h"
#include "priors/Gaussian.h"
#include "priors/LogNormal.h"
#include "priors/Gamma.h"
#include "priors/SecDetPrior.h"

typedef struct NetModel_t
{
  PyObject_HEAD


  /* the temperature determines how strongly the deterministic
  * constraints are enforced. As temperature goes to 0 all constraints
  * are fully enforced */
  double temperature;
  
  double start_time;
  double end_time;
  
  int numdetections;
  struct Detection_t * p_detections;

  int numsites;
  int num_uptime;    /* number of quantized time slots in p_site_up */

  int enable_sec_arr;                  /* enable secondary arrivals */

  int * p_site_up;                         /* numsites x num_uptime */

  EarthModel_t * p_earth;
  
  SecDetPrior_t sec_det_prior;

  NumEventPrior_t num_event_prior;
  EventLocationPrior_t event_location_prior;
  EventMagPrior_t event_mag_prior;
  EventDetectionPrior_t event_det_prior;
  ArrivalTimePrior_t arr_time_prior;
  NumFalseDetPrior_t num_falsedet_prior;
  ArrivalAzimuthPrior_t arr_az_prior;
  ArrivalSlownessPrior_t arr_slo_prior;
  ArrivalPhasePrior_t arr_phase_prior;
  ArrivalSNRPrior_t arr_snr_prior;
  ArrivalAmplitudePrior_t arr_amp_prior;
  
} NetModel_t;

#define UPTIME_QUANT     3600                /* 1 hour */


#define NetModel_IsSiteUp(p_netmodel, siteid, arrtime)\
  ((arrtime >= (p_netmodel)->start_time) &&                     \
   (arrtime < ((p_netmodel)->end_time + MAX_TRAVEL_TIME))       \
   && (p_netmodel)->p_site_up[(siteid) * (p_netmodel)->num_uptime\
   + ((int) floor(((arrtime) - (p_netmodel)->start_time) / UPTIME_QUANT))])

Event_t * alloc_event_net(NetModel_t * p_netmodel);
void copy_event_net(NetModel_t * p_netmodel, Event_t * p_tgt_event,
                const Event_t * p_src_event);
void print_event_detections(EarthModel_t * p_earth, const Event_t * p_event);

#endif // NETVISA_INCLUDE
