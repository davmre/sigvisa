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

void ArrivalPhasePrior_Init_Params(ArrivalPhasePrior_t * prior,
                                   const char * filename)
{
  FILE * fp;
  int true_phaseid;
  int arr_phaseid;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (2 != fscanf(fp, "%d %d\n", &prior->numtimedefphases, &prior->numphases))
  {
    fprintf(stderr, "error reading num phases from %s\n", filename);
    exit(1);
  }

  prior->phase2phase = (double *)calloc(prior->numtimedefphases 
                                        * prior->numphases,
                                        sizeof(*prior->phase2phase));
  
  prior->logphase2phase = (double *)calloc(prior->numtimedefphases 
                                           * prior->numphases,
                                           sizeof(*prior->logphase2phase));
  
  for (true_phaseid=0; true_phaseid < prior->numtimedefphases; true_phaseid++)
  {
    for (arr_phaseid=0; arr_phaseid < prior->numphases; arr_phaseid++)
    {
      char * fmt;
      if (arr_phaseid < (prior->numphases-1))
        fmt = "%lg ";
      else
        fmt = "%lg\n";
      
      if (1 != fscanf(fp, fmt, &prior->phase2phase[true_phaseid 
                                                   * prior->numphases
                                                   + arr_phaseid]))
      {
        fprintf(stderr, "Error reading phase emission true phaseid %d "
                "arr phaseid %d\n", true_phaseid, arr_phaseid);
        exit(1);
      }

      /* store the log of the phase emission */
      prior->logphase2phase[true_phaseid * prior->numphases + arr_phaseid] =
        log(prior->phase2phase[true_phaseid * prior->numphases + arr_phaseid]);
    }
  }

  prior->falsephase = (double *)calloc(prior->numphases,
                                       sizeof(*prior->falsephase));
  prior->logfalsephase = (double *)calloc(prior->numphases,
                                          sizeof(*prior->logfalsephase));

  for (arr_phaseid=0; arr_phaseid < prior->numphases; arr_phaseid++)
  {
    char * fmt;
    if (arr_phaseid < (prior->numphases-1))
      fmt = "%lg ";
    else
      fmt = "%lg\n";
      
    if (1 != fscanf(fp, fmt, &prior->falsephase[arr_phaseid]))
    {
      fprintf(stderr, "Error reading false phase emission for "
              "arr phaseid %d\n", arr_phaseid);
      exit(1);
    }
    
    /* store the log of the false phase emission */
    prior->logfalsephase[arr_phaseid] = log(prior->falsephase[arr_phaseid]);
  }
  
  fclose(fp);
}

double ArrivalPhasePrior_LogProb(const ArrivalPhasePrior_t * prior,
                                 int arr_phaseid, int true_phaseid)
{
  assert((arr_phaseid < prior->numphases) 
         && (true_phaseid < prior->numtimedefphases));
  
  return prior->logphase2phase[true_phaseid * prior->numphases + arr_phaseid];
}

double FalseArrivalPhasePrior_LogProb(const ArrivalPhasePrior_t * prior,
                                      int arr_phaseid)
{
  assert(arr_phaseid < prior->numphases);
  
  return prior->logfalsephase[arr_phaseid];
}

void ArrivalPhasePrior_UnInit(ArrivalPhasePrior_t * prior)
{
  free(prior->phase2phase);
  free(prior->logphase2phase);
  free(prior->falsephase);
  free(prior->logfalsephase);
}
