#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "NumEventPrior.h"
#include "Poisson.h"

void NumEventPrior_Init_Params(NumEventPrior_t * dist,
                               const char * filename)
{
  FILE * fp;
  double rate;
  
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

  dist->num_event_rate = rate;
  dist->log_num_event_rate = log(dist->num_event_rate);
}

double NumEventPrior_LogProb(const NumEventPrior_t * dist, int val,
                             double time_interval)
{
  return Poisson_logprob(dist->num_event_rate * time_interval, val);
}
