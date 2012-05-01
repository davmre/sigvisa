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

typedef struct SiteFalseAmp_t
{
  double wt0;
  double wt1;
  double mean0;
  double mean1;
  double std0;
  double std1;
  
} SiteFalseAmp_t;

typedef struct PhaseAmp_t
{
  double intercept;
  double mb_coeff;
  double depth_coeff;
  double ttime_coeff;
  double ttime0_coeff;
  
  double std;                                /* standard deviation */
  
} PhaseAmp_t;

typedef struct ArrivalAmplitudePrior_t
{
  int numsites;
  int numphases;
 
  SiteFalseAmp_t * p_site_false;             /* numsites x */
  PhaseAmp_t * p_site_phase_amp;             /* numsites x numphases */

} ArrivalAmplitudePrior_t;

void ArrivalAmplitudePrior_Init_Params(ArrivalAmplitudePrior_t * prior,
                                       const char * filename);

double ArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * prior,
                                     double mb, double depth, double ttime,
                                     int siteid, int phaseid, double amp);

double ArrivalAmplitudePrior_logamp(const ArrivalAmplitudePrior_t * prior,
                                    double mb, double depth, double ttime,
                                    int siteid, int phaseid);

double FalseArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * 
                                          prior, int siteid, double amplitude);

double ArrivalAmplitudePrior_cdf(const ArrivalAmplitudePrior_t * prior,
                                 double mb, double depth, double ttime,
                                 int siteid, int phaseid, double amp);

double FalseArrivalAmplitudePrior_cdf(const ArrivalAmplitudePrior_t * 
                                      prior, int siteid, double amplitude);

double ArrivalAmplitudePrior_zval(const ArrivalAmplitudePrior_t * prior,
                                  double mb, double depth, double ttime,
                                  int siteid, int phaseid, double amp);

void ArrivalAmplitudePrior_UnInit(ArrivalAmplitudePrior_t * prior);
