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

#ifndef EVENT_LOC_PRIOR_H
#define EVENT_LOC_PRIOR_H


typedef struct EventLocationPrior_t
{
  double lonstep;
  double zstep;
  int numlon;
  int numlat;
  
  double * p_bucketprob;  /* numlon x numlat array of probabilities */

  double * p_lonprob; /* the probability of each longitude size = numlon */
  
  /* numlon x numlat array of latitude probabilities normalized for
   * each longitude */
  double * p_latprob;

  double north_pole_prob;                    /* probability of a
                                              * hypothetical bucket at
                                              * the north pole */
} EventLocationPrior_t;

void EventLocationPrior_Init_Params(EventLocationPrior_t * dist,
                                    const char * filename);

double EventLocationPrior_LogProb(const EventLocationPrior_t * dist,
                                  double lon, double lat, double depth);

double depth_lp(double depth);

void EventLocationPrior_Sample(const EventLocationPrior_t * dist,
                               double * p_lon, double * p_lat, 
                               double * p_depth);

void EventLocationPrior_UnInit(EventLocationPrior_t * dist);

#endif
