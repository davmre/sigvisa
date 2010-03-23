#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "NumEventPrior.h"

void NumEventPrior_Init_Params(NumEventPrior_t * dist, int nparams, 
                               const char * filename, double time_interval)
{
  FILE * fp;
  double rate;
  
  assert(nparams == 2);
  
  fp = fopen(filename, "r");

  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (1 != fscanf(fp, "%lg\n", &rate))
  {
    fprintf(stderr, "error reading rate from %s\n", filename);
    exit(1);
  }
  
  fclose(fp);

  dist->num_event_rate = rate * time_interval;
}

/* log(n!) for n > 0 */
static double log_fact(int n)
{
  int i;
  double sum;
  
  sum = 0;
  for (i=2; i<=n; i++)
    sum += log((double) i);
  return sum;
}

double NumEventPrior_LogProb(const NumEventPrior_t * dist, int val, int nargs)
{
  assert(nargs==0);
  
  if (val < 0)
    return -HUGE_VAL;

#ifdef DEBUG  
  printf("\nNumEventPrior_LogProb: %d events, mean %lf\n", val,
         dist->num_event_rate);
#endif

  return -dist->num_event_rate + val * log(dist->num_event_rate)
    - log_fact(val);
}
