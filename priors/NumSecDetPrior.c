#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "NumSecDetPrior.h"
#include "Poisson.h"

void NumSecDetPrior_Init_Params(NumSecDetPrior_t * dist,
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
    fprintf(stderr, "error reading secdet rate from %s\n", filename);
    exit(1);
  }
  
  fclose(fp);

  dist->num_secdet_rate = rate;
  dist->log_num_secdet_rate = log(dist->num_secdet_rate);
}

double NumSecDetPrior_LogProb(const NumSecDetPrior_t * dist, int val)
{
  return Poisson_logprob(dist->num_secdet_rate, val);
}
