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

#define NUM_EDA_COEFFS 12

#define EDA_COEFF_INTERCEPT 0
#define EDA_COEFF_MAG       1
#define EDA_COEFF_DEPTH     2
#define EDA_COEFF_DIST      3
#define EDA_COEFF_DIST0     4
#define EDA_COEFF_DIST35    5
#define EDA_COEFF_DIST40    6
#define EDA_COEFF_DIST12520 7
#define EDA_COEFF_DIST12540 8
#define EDA_COEFF_MAG6      9
#define EDA_COEFF_MAG68    10
#define EDA_COEFF_MD       11

typedef struct EventDetectionPrior_t
{
  int numsites;
  int numtimedefphases;
  double * p_coeffs; /* numsites x numtimedefphases x NUM_EDA_COEFFS */
} EventDetectionPrior_t;

void EventDetectionPrior_Init_Params(EventDetectionPrior_t * dist,
                                     const char * filename);

double EventDetectionPrior_LogProb(const EventDetectionPrior_t * prior,
                                   int is_detected,
                                   double evdepth, double evmag, double dist,
                                   int siteid, int phaseid, double dderror);

void EventDetectionPrior_UnInit(EventDetectionPrior_t * dist);
