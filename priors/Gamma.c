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
