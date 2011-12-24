#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

void ArrivalSNRPrior_Init_Params(ArrivalSNRPrior_t * prior,
                                 const char * filename)
{
  FILE * fp;
  int bin, phase;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (3 != fscanf(fp, "%d %d %d\n", &prior->step_size, &prior->num_bins,
        &prior->num_phases))
  {
    fprintf(stderr, "error reading step size, num bins and phases from %s\n", 
            filename);
    exit(1);
  }

  prior->true_prob = (double *)calloc(prior->num_bins * prior->num_phases,
                                      sizeof(*prior->true_prob));
  prior->false_prob = (double *)calloc(prior->num_bins,
                                       sizeof(*prior->false_prob));
  
  for (bin=0; bin<prior->num_bins; bin++)
    if (1 != fscanf(fp, "%lg ", prior->false_prob + bin))
    {
      fprintf(stderr, "Error reading false prob bin %d\n", bin);
      exit(1);
    }
  
  for (phase=0; phase < prior->num_phases; phase++)
    for (bin=0; bin < prior->num_bins; bin++)
      if (1 != fscanf(fp, "%lg ", prior->true_prob 
                      + phase * prior->num_bins + bin))
      {
        fprintf(stderr, "Error reading true prob phase %d bin %d\n", phase,
                bin);
        exit(1);
      }
  
  fclose(fp);
}

double ArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                               int siteid, int phaseid, double snr)
{
  int bin;
  
  assert((snr > 0) && (phaseid >= 0) && (phaseid < prior->num_phases));
  
  bin = (int) floor(snr / prior->step_size);

  if (bin >= prior->num_bins)
    bin = prior->num_bins - 1;
  
  return log(prior->true_prob[phaseid * prior->num_bins + bin]);
}

double FalseArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                                    int siteid, double snr)
{
  int bin;
  
  assert(snr > 0);
  
  bin = (int) floor(snr / prior->step_size);

  if (bin >= prior->num_bins)
    bin = prior->num_bins - 1;
  
  return log(prior->false_prob[bin]);
}

void ArrivalSNRPrior_UnInit(ArrivalSNRPrior_t * prior)
{
  free(prior->true_prob);
  free(prior->false_prob);
}
