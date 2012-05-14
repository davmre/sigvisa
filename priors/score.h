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

double score_world(NetModel_t * p_netmodel,
                   int numevents, Event_t * p_events,
                   int verbose);

double score_event(NetModel_t * p_netmodel, Event_t * p_event);

int score_event_site_phase(NetModel_t * p_netmodel,
                           const Event_t * p_event,
                           int siteid, int phaseid,
                           double distance, double pred_az,
                           double * p_score);

int score_event_site_phase_simple(NetModel_t * p_netmodel,
                                  const Event_t * p_event, int siteid,
                                  int phaseid, double * p_score);

double prob_event(NetModel_t * p_netmodel, Event_t * p_event);

double logprob_false(NetModel_t * p_netmodel, int numdets, int * p_detids,
                     int verbose);

double logprob_noise(NetModel_t * p_netmodel, Detection_t * p_det,
                     Detection_t * p_prev_det);
double logprob_coda_coda(NetModel_t * p_netmodel, Detection_t * p_secdet,
                         Detection_t * p_det);

/* log probability that p_det (which is a phase arrival) generates the
 * coda arrival p_secdet 
 * minus
 * log probability that p_det doesn't generate a secondary arrival and
 * p_secdet is a noise arrival
 */
double score_phase_coda(NetModel_t * p_netmodel, Detection_t * p_secdet,
                        Detection_t * p_det);

/* log probability that p_det generates the coda arrival p_secdet 
 * minus
 * log probability that p_det doesn't generate a secondary arrival and
 * p_secdet is a noise arrival
 */
double score_coda_coda(NetModel_t * p_netmodel, Detection_t * p_secdet,
                       Detection_t * p_det);
