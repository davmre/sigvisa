#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../sigvisa.h"

void ArrivalSNRPrior_Init_Params(ArrivalSNRPrior_t * prior,
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

  prior->true_mean = (double *)calloc(prior->numsites,
                                      sizeof(*prior->true_mean));
  prior->true_sigma = (double *)calloc(prior->numsites,
                                       sizeof(*prior->true_sigma));
  prior->false_mean = (double *)calloc(prior->numsites,
                                       sizeof(*prior->false_mean));
  prior->false_sigma = (double *)calloc(prior->numsites,
                                        sizeof(*prior->false_sigma));
  
  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    if (4 != fscanf(fp, "%lg %lg %lg %lg\n", prior->true_mean + siteid,
                    prior->true_sigma + siteid, prior->false_mean + siteid,
                    prior->false_sigma + siteid))
    {
      fprintf(stderr, "Error reading SNR mean/sigma for site %d\n", siteid);
      exit(1);
    }
  }

  fclose(fp);
}

double ArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                               int siteid, double snr)
{
  assert((siteid >= 0) && (siteid < prior->numsites));
  
  
  return LogNormal_ldensity(prior->true_mean[siteid],
                            prior->true_sigma[siteid], snr);
}

double FalseArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                                    int siteid, double snr)
{
  assert((siteid >= 0) && (siteid < prior->numsites));
  
  return LogNormal_ldensity(prior->false_mean[siteid],
                            prior->false_sigma[siteid], snr);
}

void ArrivalSNRPrior_UnInit(ArrivalSNRPrior_t * prior)
{
  free(prior->true_mean);
  free(prior->true_sigma);
  free(prior->false_mean);
  free(prior->false_sigma);
}
