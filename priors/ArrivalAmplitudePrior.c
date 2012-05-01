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
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

#define UNIF_FALSE 0.13         /* uniform prior on false amplitude */

void ArrivalAmplitudePrior_Init_Params(ArrivalAmplitudePrior_t * prior,
                                       const char * filename)
{
  FILE * fp;
  int siteid;
  int phaseid;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (2 != fscanf(fp, "%d %d\n", &prior->numsites, &prior->numphases))
  {
    fprintf(stderr, "error reading num sites, num phases from %s\n", filename);
    exit(1);
  }

  prior->p_site_false = (SiteFalseAmp_t *) calloc(prior->numsites,
                                              sizeof(*prior->p_site_false));
  prior->p_site_phase_amp = (PhaseAmp_t *) calloc(prior->numsites 
                                                  * prior->numphases,
                                             sizeof(*prior->p_site_phase_amp));
  
  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    SiteFalseAmp_t * p_false;
    
    p_false = prior->p_site_false + siteid;
    
    if (6 != fscanf(fp, "%lg %lg %lg %lg %lg %lg\n", &p_false->wt0,
                    &p_false->wt1, &p_false->mean0, &p_false->mean1,
                    &p_false->std0, &p_false->std1))
    {
      fprintf(stderr, "Error reading false amp model for site %d\n", siteid);
      exit(1);
    }
    
    for (phaseid = 0; phaseid < prior->numphases; phaseid ++)
    {
      PhaseAmp_t * p_phase;
      
      p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;
    
      if (6 != fscanf(fp, "%lg %lg %lg %lg %lg %lg\n", &p_phase->intercept,
                      &p_phase->mb_coeff, &p_phase->depth_coeff,
                      &p_phase->ttime_coeff, &p_phase->ttime0_coeff, 
                      &p_phase->std))
      {
        fprintf(stderr, "Error reading amp model for site %d phase %d\n", 
                siteid, phaseid);
        exit(1);
      }
    }
  }
  
  fclose(fp);
}

double ArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * prior,
                                     double mb, double depth, double ttime,
                                     int siteid, int phaseid, double amp)
{
  double logamp;
  PhaseAmp_t * p_phase;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert((phaseid >= 0) && (phaseid < prior->numphases));
  assert(amp > 0);
  
  if (amp < MIN_AMP)
    amp = MIN_AMP;
  else if (amp > MAX_AMP)
    amp = MAX_AMP;
  
  logamp = log(amp);

  p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;

  return Gaussian_logprob(logamp, p_phase->intercept + p_phase->mb_coeff * mb
                          + p_phase->depth_coeff * depth
                          + p_phase->ttime_coeff * ttime
                          + p_phase->ttime0_coeff * exp(-ttime/50.0), 
                          p_phase->std);
}

double ArrivalAmplitudePrior_logamp(const ArrivalAmplitudePrior_t * prior,
                                    double mb, double depth, double ttime,
                                    int siteid, int phaseid)
{
  PhaseAmp_t * p_phase;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert((phaseid >= 0) && (phaseid < prior->numphases));

  p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;

  return p_phase->intercept + p_phase->mb_coeff * mb
    + p_phase->depth_coeff * depth
    + p_phase->ttime_coeff * ttime
    + p_phase->ttime0_coeff * exp(-ttime/50.0);
}

double FalseArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * 
                                          prior, int siteid, double amplitude)
{
  double logamp;
  SiteFalseAmp_t * p_false;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert(amplitude > 0);

  if (amplitude < MIN_AMP)
    amplitude = MIN_AMP;
  else if (amplitude > MAX_AMP)
    amplitude = MAX_AMP;
  
  logamp = log(amplitude);
  
  p_false = prior->p_site_false + siteid;
  
  /* std could be 0 so we add 1e-6
   * we mix with a uniform distribution to avoid extreme values when
   * the amplitude is far from the mean */
  return log(p_false->wt0 * Gaussian_prob(logamp, p_false->mean0, 
                                          p_false->std0 + 1e-6) *(1-UNIF_FALSE)
             + p_false->wt1 * Gaussian_prob(logamp, p_false->mean1, 
                                            p_false->std1 +1e-6)*(1-UNIF_FALSE)
             + UNIF_FALSE / (LOG_MAX_AMP - LOG_MIN_AMP));
}

double ArrivalAmplitudePrior_cdf(const ArrivalAmplitudePrior_t * prior,
                                 double mb, double depth, double ttime,
                                 int siteid, int phaseid, double amp)
{
  double logamp;
  PhaseAmp_t * p_phase;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert((phaseid >= 0) && (phaseid < prior->numphases));
  assert(amp > 0);

  if (amp < MIN_AMP)
    amp = MIN_AMP;
  else if (amp > MAX_AMP)
    amp = MAX_AMP;
  
  logamp = log(amp);

  p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;

  return Gaussian_cdf(logamp, p_phase->intercept + p_phase->mb_coeff * mb
                      + p_phase->depth_coeff * depth
                      + p_phase->ttime_coeff * ttime
                      + p_phase->ttime0_coeff * exp(-ttime/50.0), 
                      p_phase->std);
}

double ArrivalAmplitudePrior_zval(const ArrivalAmplitudePrior_t * prior,
                                  double mb, double depth, double ttime,
                                  int siteid, int phaseid, double amp)
{
  double logamp;
  PhaseAmp_t * p_phase;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert((phaseid >= 0) && (phaseid < prior->numphases));
  assert(amp > 0);

  if (amp < MIN_AMP)
    amp = MIN_AMP;
  else if (amp > MAX_AMP)
    amp = MAX_AMP;
  
  logamp = log(amp);

  p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;

  return (logamp - (p_phase->intercept + p_phase->mb_coeff * mb
                    + p_phase->depth_coeff * depth
                    + p_phase->ttime_coeff * ttime
                    + p_phase->ttime0_coeff * exp(-ttime/50.0))) / p_phase->std;
}

double FalseArrivalAmplitudePrior_cdf(const ArrivalAmplitudePrior_t * 
                                      prior, int siteid, double amplitude)
{
  double logamp;
  SiteFalseAmp_t * p_false;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert(amplitude > 0);
  
  if (amplitude < MIN_AMP)
    amplitude = MIN_AMP;
  else if (amplitude > MAX_AMP)
    amplitude = MAX_AMP;
  
  logamp = log(amplitude);

  if ((logamp < LOG_MIN_AMP) || (logamp > LOG_MAX_AMP))
  {
    fprintf(stderr, "logamp out of bound\n");
    exit(1);
  }

  p_false = prior->p_site_false + siteid;
  
  /* std could be 0 so we add 1e-6
   * we mix with a uniform distribution to avoid extreme values when
   * the amplitude is far from the mean */
  return (p_false->wt0 * Gaussian_cdf(logamp, p_false->mean0, 
                                      p_false->std0 + 1e-6) *(1-UNIF_FALSE)
          + p_false->wt1 * Gaussian_cdf(logamp, p_false->mean1, 
                                        p_false->std1 +1e-6)*(1-UNIF_FALSE)
          + UNIF_FALSE * ((logamp - LOG_MIN_AMP)/(LOG_MAX_AMP - LOG_MIN_AMP)));
}

void ArrivalAmplitudePrior_UnInit(ArrivalAmplitudePrior_t * prior)
{
  free(prior->p_site_false);
  free(prior->p_site_phase_amp);
}
