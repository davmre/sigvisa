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


void NumFalseDetPrior_Init_Params(NumFalseDetPrior_t * prior,
                                  const char * filename)
{
  FILE * fp;
  int siteid;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (1 != fscanf(fp, "%d\n", &prior->numsites))
  {
    fprintf(stderr, "error reading num sites from %s\n", filename);
    exit(1);
  }

  prior->falserates = (double *)calloc(prior->numsites,
                                       sizeof(*prior->falserates));

  prior->logfalserates = (double *)calloc(prior->numsites,
                                          sizeof(*prior->logfalserates));

  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    if (1 != fscanf(fp, "%lg\n", &prior->falserates[siteid])) 
    {
      fprintf(stderr, "NumFalseDetPrior: can't read false rate for siteid %d"
              " in file %s", siteid, filename);
      exit(1);
    }

    prior->logfalserates[siteid] = log(prior->falserates[siteid]);
  }
  
  fclose(fp);
}

double NumFalseDetPrior_LogProb(const NumFalseDetPrior_t * prior,
                                int numfalse, int siteid)
{
  assert(siteid < prior->numsites);
  
  return Poisson_logprob(prior->falserates[siteid], numfalse);
}

void NumFalseDetPrior_UnInit(NumFalseDetPrior_t * prior)
{
  free(prior->falserates);
  free(prior->logfalserates);
}

