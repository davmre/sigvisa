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
typedef struct SecDetPrior_t
{
  int num_logamp_bins;
  double low_logamp;
  double step_logamp;
  double * pc_detprob;                       /* X num_logamp_bins */
  double * cc_detprob;                       /* X num_logamp_bins */
  double pc_time_shape;
  double pc_min_delay;
  double pc_time_scale;
  double pc_max_delay;
  double cc_time_shape;
  double cc_min_delay;
  double cc_time_scale;
  double cc_max_delay;
  double pc_az_loc;
  double pc_az_scale;
  double cc_az_loc;
  double cc_az_scale;
  double pc_slo_loc;
  double pc_slo_scale;
  double cc_slo_loc;
  double cc_slo_scale;
  double pc_logamp_loc;
  double pc_logamp_scale;
  double cc_logamp_loc;
  double cc_logamp_scale;
  double pc_snr_pos_prob;
  double pc_snr_pos_lambda;
  double pc_snr_neg_lambda;
  double cc_snr_pos_prob;
  double cc_snr_pos_lambda;
  double cc_snr_neg_lambda;
  int numphases;
  double * pc_phaseprob;                        /* numphases */
  double * cc_phaseprob;                        /* numphases */
  
} SecDetPrior_t;

void SecDetPrior_Init_Params(SecDetPrior_t * prior, const char * fname);

void SecDetPrior_UnInit(SecDetPrior_t * prior);

int SecDetPrior_Time_Possible(const SecDetPrior_t * prior, double sec_time,
                              double prim_time);

double SecDetPrior_PhaseCodaDet_LogProb(const SecDetPrior_t * prior, int is_det,
                                        double prim_amp);

double SecDetPrior_CodaCodaDet_LogProb(const SecDetPrior_t * prior, int is_det,
                                       double prim_amp);

double SecDetPrior_PhaseCodaTime_LogProb(const SecDetPrior_t * prior,
                                         double sec_time, double prim_time);

double SecDetPrior_CodaCodaTime_LogProb(const SecDetPrior_t * prior,
                                        double sec_time, double prim_time);

double SecDetPrior_PhaseCodaSlow_LogProb(const SecDetPrior_t * prior,
                                         double sec_slow, double prim_slow);

double SecDetPrior_CodaCodaSlow_LogProb(const SecDetPrior_t * prior,
                                        double sec_slow, double prim_slow);

double SecDetPrior_PhaseCodaAzimuth_LogProb(const SecDetPrior_t * prior,
                                            double sec_az, double prim_az);

double SecDetPrior_CodaCodaAzimuth_LogProb(const SecDetPrior_t * prior,
                                           double sec_az, double prim_az);

double SecDetPrior_PhaseCodaPhase_LogProb(const SecDetPrior_t * prior,
                                          int sec_phaseid);

double SecDetPrior_CodaCodaPhase_LogProb(const SecDetPrior_t * prior,
                                         int sec_phaseid);

double SecDetPrior_PhaseCodaAmp_LogProb(const SecDetPrior_t * prior,
                                        double sec_amp, double prim_amp);

double SecDetPrior_CodaCodaAmp_LogProb(const SecDetPrior_t * prior,
                                       double sec_amp, double prim_amp);

double SecDetPrior_PhaseCodaSNR_LogProb(const SecDetPrior_t * prior,
                                        double sec_snr, double prim_snr);

double SecDetPrior_CodaCodaSNR_LogProb(const SecDetPrior_t * prior,
                                       double sec_snr, double prim_snr);
