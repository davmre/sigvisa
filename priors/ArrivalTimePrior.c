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

#include "../sigvisa.h"

void ArrivalTimePrior_Init_Params(ArrivalTimePrior_t * prior,
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
    fprintf(stderr, "error reading num sites and phases from %s\n", filename);
    exit(1);
  }

  prior->locs = (double *)calloc(prior->numsites * prior->numphases,
                                 sizeof(*prior->locs));
  
  prior->scales = (double *)calloc(prior->numsites * prior->numphases,
                                   sizeof(*prior->scales));

  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    for (phaseid = 0; phaseid < prior->numphases; phaseid ++)
    {
      if (2 != fscanf(fp, "%lg %lg\n", 
                      &prior->locs[siteid * prior->numphases + phaseid],
                      &prior->scales[siteid * prior->numphases + phaseid]))
      {
        fprintf(stderr, "ArrivalTimePrior: can't read loc, scale for siteid %d"
                " phaseid %d in file %s", siteid, phaseid, filename);
        exit(1);
      }
    }
  }
  
  fclose(fp);
}


double ArrivalTimePrior_LogProb(const ArrivalTimePrior_t * prior,
                                double arrtime, double pred_arrtime,
                                double det_deltime, int siteid, int phaseid)
{
  double loc;
  double scale;
  double res;
  
  assert((siteid < prior->numsites) && (phaseid < prior->numphases));
  
  loc = prior->locs[siteid * prior->numphases + phaseid];
  scale = prior->scales[siteid * prior->numphases + phaseid] + det_deltime;

  res = arrtime - pred_arrtime;
  
  return Laplace_ldensity(loc, scale, res);
}

double ArrivalTimePrior_MeanResidual(const ArrivalTimePrior_t * prior,
                                     int siteid, int phaseid)
{
  assert((siteid < prior->numsites) && (phaseid < prior->numphases));
  
  return prior->locs[siteid * prior->numphases + phaseid];
}


void ArrivalTimePrior_UnInit(ArrivalTimePrior_t * prior)
{
  free(prior->locs);
  free(prior->scales);
}

