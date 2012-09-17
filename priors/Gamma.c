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
#include <math.h>

#include "../sigvisa.h"


#ifndef M_PI
 #define M_PI           ((double) 3.14159265358979323846)
#endif

static double loggamma(double num)
{
  return (num - .5) * log(num) - num + .5 * log(2 * M_PI);  
}

double Gamma_logprob(double val, double shape, double scale)
{
  double logprob= (shape-1) * log(val) - val / scale - loggamma(shape)
    - shape * log(scale);

  if (isinf(logprob))
  {
    printf("logprob isinf: val %lg, shape %lg, scale %lg\n",
           val, shape, scale);
    exit(1);
  }
  
  return logprob;
  
}

double Gamma_cdf(double val, double shape, double scale)
{
  double sum=0;
  double delta = val / 100.0;
  double i;
  
  for (i=delta; i<val; i+=delta)
    sum += exp(Gamma_logprob(i, shape, scale));
  
  return sum * delta;
}
