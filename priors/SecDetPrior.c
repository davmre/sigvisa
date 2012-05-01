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
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "SecDetPrior.h"
#include "Laplace.h"
#include "Gamma.h"

#define READPARAM(cnt, fncall, name)\
  do {                              \
  if ((cnt) != (fncall))            \
  {                     \
  fprintf(stderr, "error reading " #name " from %s\n", filename);  \
  exit(1);                                                              \
  }                                                                     \
  } while(0)

static double degdiff(double angle1, double angle2)
{
  /*
  The difference of two angles given in degrees. The answer is an angle from
  -180 to 180. Positive angles imply angle2 is clockwise from angle1 and -ve
  angles imply counter-clockwise.
  
  >>> int(degdiff(40, 30))
  -10
  >>> int(degdiff(30, 40))
  10
  >>> int(degdiff(361, 40))
  39
  >>> int(degdiff(40, 361))
  -39
  >>> degdiff(40,250)
  -150
  >>> degdiff(40,200)
  160
  >>> degdiff(40, 219)
  179
  >>> degdiff(40, 220)
  180
  >>> degdiff(40, 221)
  -179
  */
  /* bring the angle into the 0 to 360 range */
  double delta = fmod((angle2 - angle1) + 360, 360.0);
  
  /* angles above 180 need to be shifted down by 360 degrees so that 181 is -179
   * 200 is -160 etc. */
  return delta - (delta > 180 ? 360 : 0);
}

void SecDetPrior_Init_Params(SecDetPrior_t * prior, const char * filename)
{
  FILE * fp;
  int i;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  READPARAM(3, fscanf(fp, "%d %lg %lg\n", &prior->num_logamp_bins,
                      &prior->low_logamp, &prior->step_logamp), 
            "number of logamp bins, low logamp, step logamp");

  prior->pc_detprob = (double *)malloc(sizeof(*prior->pc_detprob) 
                                       * prior->num_logamp_bins);
  for (i=0; i<prior->num_logamp_bins; i++)
    READPARAM(1, fscanf(fp, "%lg ", prior->pc_detprob + i),
              "phase -> coda probability");

  prior->cc_detprob = (double *)malloc(sizeof(*prior->cc_detprob) 
                                       * prior->num_logamp_bins);
  for (i=0; i<prior->num_logamp_bins; i++)
    READPARAM(1, fscanf(fp, "%lg ", prior->cc_detprob + i),
              "coda -> coda probability");
  
  READPARAM(4, fscanf(fp, "%lg %lg %lg %lg\n", &prior->pc_time_shape,
                      &prior->pc_min_delay, &prior->pc_time_scale,
                      &prior->pc_max_delay),
            "phase -> coda time shape, location, scale, maximum");
  READPARAM(4, fscanf(fp, "%lg %lg %lg %lg\n", &prior->cc_time_shape,
                      &prior->cc_min_delay, &prior->cc_time_scale,
                      &prior->cc_max_delay),
            "coda -> coda time shape, location, scale, maximum");

  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->pc_az_loc, &prior->pc_az_scale),
            "phase -> coda azimuth location and scale");
  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->cc_az_loc, &prior->cc_az_scale),
            "coda -> coda azimuth location and scale");

  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->pc_slo_loc, &prior->pc_slo_scale),
            "phase -> coda slowness location and scale");
  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->cc_slo_loc, &prior->cc_slo_scale),
            "coda -> coda slowness location and scale");

  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->pc_logamp_loc,
                      &prior->pc_logamp_scale),
            "phase -> coda logamp location and scale");
  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->cc_logamp_loc,
                      &prior->cc_logamp_scale),
            "coda -> coda logamp location and scale");

  READPARAM(3, fscanf(fp, "%lg %lg %lg\n", &prior->pc_snr_pos_prob,
                      &prior->pc_snr_pos_lambda, &prior->pc_snr_neg_lambda),
            "phase -> coda SNR location and scale");
  READPARAM(3, fscanf(fp, "%lg %lg %lg\n", &prior->cc_snr_pos_prob,
                      &prior->cc_snr_pos_lambda, &prior->cc_snr_neg_lambda),
            "coda -> coda SNR location and scale");

  READPARAM(1, fscanf(fp, "%d\n", &prior->numphases), "number of phases");

  prior->pc_phaseprob = (double *)malloc(prior->numphases 
                                         * sizeof(*prior->pc_phaseprob));
  
  for (i=0; i<prior->numphases; i++)
    READPARAM(1, fscanf(fp, "%lg ", prior->pc_phaseprob + i),
              "phase -> coda phase probability");

  prior->cc_phaseprob = (double *)malloc(prior->numphases 
                                         * sizeof(*prior->cc_phaseprob));
  for (i=0; i<prior->numphases; i++)
    READPARAM(1, fscanf(fp, "%lg ", prior->cc_phaseprob + i), 
              "coda -> coda phase probability");
  
  fclose(fp);
}

void SecDetPrior_UnInit(SecDetPrior_t * prior)
{
  free(prior->pc_detprob);
  free(prior->cc_detprob);
  free(prior->pc_phaseprob);
  free(prior->cc_phaseprob);
}

double SecDetPrior_PhaseCodaDet_LogProb(const SecDetPrior_t * prior, int is_det,
                                        double prim_amp)
{
  int index;
  
  /* if prim_amp is -1 we will simply use the min bucket */
  if (prim_amp <= 0)
    index = 0;
  
  else
  {
    index = (int)floor((log(prim_amp) - prior->low_logamp)/prior->step_logamp);
    
    /* keep the index within range */
    if (index < 0)
      index = 0;

    if (index >= prior->num_logamp_bins)
      index = prior->num_logamp_bins - 1;
  }
  
  if (is_det)
    return log(prior->pc_detprob[index]);
  
  else
    return log(1-prior->pc_detprob[index]);
}

double SecDetPrior_CodaCodaDet_LogProb(const SecDetPrior_t * prior, int is_det,
                                       double prim_amp)
{
  int index;
  
  /* if prim_amp is -1 we will simply use the min bucket */
  if (prim_amp <= 0)
    index = 0;
  
  else
  {
    index = (int)floor((log(prim_amp) - prior->low_logamp)/prior->step_logamp);
    
    /* keep the index within range */
    if (index < 0)
      index = 0;

    if (index >= prior->num_logamp_bins)
      index = prior->num_logamp_bins - 1;
  }
  
  if (is_det)
    return log(prior->cc_detprob[index]);
  
  else
    return log(1-prior->cc_detprob[index]);
}

int SecDetPrior_Time_Possible(const SecDetPrior_t * prior, double sec_time,
                              double prim_time)
{
  double delta = sec_time - prim_time;

  return (delta <= prior->pc_max_delay) && (delta > prior->pc_min_delay) ? 1 : 0;
}

double SecDetPrior_PhaseCodaTime_LogProb(const SecDetPrior_t * prior,
                                         double sec_time, double prim_time)
{
  double delta = sec_time - prim_time;
  assert((delta <= prior->pc_max_delay) && (delta > prior->pc_min_delay));
  
  return Gamma_logprob(delta - prior->pc_min_delay, prior->pc_time_shape,
                       prior->pc_time_scale);
}

double SecDetPrior_CodaCodaTime_LogProb(const SecDetPrior_t * prior,
                                        double sec_time, double prim_time)
{
  double delta = sec_time - prim_time;
  assert((delta <= prior->cc_max_delay) && (delta > prior->cc_min_delay));
  
  return Gamma_logprob(delta - prior->cc_min_delay, prior->cc_time_shape,
                       prior->cc_time_scale);
}

double SecDetPrior_PhaseCodaSlow_LogProb(const SecDetPrior_t * prior,
                                         double sec_slow, double prim_slow)
{
  return Laplace_ldensity(prior->pc_slo_loc, prior->pc_slo_scale, 
                          sec_slow - prim_slow);
}

double SecDetPrior_CodaCodaSlow_LogProb(const SecDetPrior_t * prior,
                                        double sec_slow, double prim_slow)
{
  return Laplace_ldensity(prior->cc_slo_loc, prior->cc_slo_scale, 
                          sec_slow - prim_slow);
}

double SecDetPrior_PhaseCodaAzimuth_LogProb(const SecDetPrior_t * prior,
                                            double sec_az, double prim_az)
{
  return Laplace_ldensity(prior->pc_az_loc, prior->pc_az_scale,
                          degdiff(prim_az, sec_az));
}

double SecDetPrior_CodaCodaAzimuth_LogProb(const SecDetPrior_t * prior,
                                           double sec_az, double prim_az)
{
  return Laplace_ldensity(prior->cc_az_loc, prior->cc_az_scale,
                          degdiff(prim_az, sec_az));
}

double SecDetPrior_PhaseCodaPhase_LogProb(const SecDetPrior_t * prior,
                                          int sec_phaseid)
{
  assert(sec_phaseid < prior->numphases);
  
  return log(prior->pc_phaseprob[sec_phaseid]);
}

double SecDetPrior_CodaCodaPhase_LogProb(const SecDetPrior_t * prior,
                                         int sec_phaseid)
{
  assert(sec_phaseid < prior->numphases);
  
  return log(prior->cc_phaseprob[sec_phaseid]);
}

double SecDetPrior_PhaseCodaAmp_LogProb(const SecDetPrior_t * prior,
                                        double sec_amp, double prim_amp)
{
  return Laplace_ldensity(prior->pc_logamp_loc, prior->pc_logamp_scale, 
                          log(sec_amp) - log(prim_amp));
}

double SecDetPrior_CodaCodaAmp_LogProb(const SecDetPrior_t * prior,
                                       double sec_amp, double prim_amp)
{
  return Laplace_ldensity(prior->cc_logamp_loc, prior->cc_logamp_scale, 
                          log(sec_amp) - log(prim_amp));
}

double SecDetPrior_PhaseCodaSNR_LogProb(const SecDetPrior_t * prior,
                                        double sec_snr, double prim_snr)
{
  double delta = sec_snr - prim_snr;
  
  if (delta >= 0)
    return log(prior->pc_snr_pos_prob) + log(prior->pc_snr_pos_lambda) 
      - prior->pc_snr_pos_lambda * delta;
  else
    return log(1-prior->pc_snr_pos_prob) + log(prior->pc_snr_neg_lambda)
      + prior->pc_snr_neg_lambda * delta;
}

double SecDetPrior_CodaCodaSNR_LogProb(const SecDetPrior_t * prior,
                                       double sec_snr, double prim_snr)
{
  double delta = sec_snr - prim_snr;
  
  if (delta >= 0)
    return log(prior->cc_snr_pos_prob) + log(prior->cc_snr_pos_lambda)
      - prior->cc_snr_pos_lambda * delta;
  else
    return log(1-prior->cc_snr_pos_prob) + log(prior->cc_snr_neg_lambda)
      + prior->cc_snr_neg_lambda * delta;
}

