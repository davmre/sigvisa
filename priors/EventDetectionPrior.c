#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

void EventDetectionPrior_Init_Params(EventDetectionPrior_t * prior, 
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

  prior->p_coeff_mag = (double *)malloc(sizeof(*prior->p_coeff_mag)
                                        * prior->numsites);

  prior->p_coeff_dist = (double *)malloc(sizeof(*prior->p_coeff_dist)
                                         * prior->numsites);

  prior->p_coeff_phases = (double *)malloc(sizeof(*prior->p_coeff_phases)
                                          * prior->numsites 
                                          * prior->numphases);

  prior->p_coeff_bias = (double *)malloc(sizeof(*prior->p_coeff_bias)
                                        * prior->numsites);
  
  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    if (1 != fscanf(fp, "%lg ", &prior->p_coeff_mag[siteid]))
    {
      fprintf(stderr, "error reading mag-coeff for siteid %d in file %s\n",
              siteid, filename);
      exit(1);
    }
    if (1 != fscanf(fp, "%lg ", &prior->p_coeff_dist[siteid]))
    {
      fprintf(stderr, "error reading dist-coeff for siteid %d in file %s\n",
              siteid, filename);
      exit(1);
    }
    for (phaseid = 0; phaseid < prior->numphases; phaseid ++)
    {
      if (1 != fscanf(fp, "%lg ",&prior->p_coeff_phases[siteid 
                                                       * prior->numphases
                                                       + phaseid]))
      {
        fprintf(stderr, "error reading phaseid %d for siteid %d in file %s\n",
                phaseid, siteid, filename);
        exit(1);
      }
    }
    if (1 != fscanf(fp, "%lg\n", &prior->p_coeff_bias[siteid]))
    {
      fprintf(stderr, "error reading bias for siteid %d in file %s\n",
              siteid, filename);
      exit(1);
    }
  }
  
  fclose(fp);
}

double EventDetectionPrior_LogProb(const EventDetectionPrior_t * prior,
                                   int is_detected,
                                   double evdepth, double evmag, double dist,
                                   int siteid, int phaseid)
{
  double logodds;
  double logprob;
  
  logodds = prior->p_coeff_mag[siteid] * evmag
    + prior->p_coeff_dist[siteid] * dist
    + prior->p_coeff_phases[siteid * prior->numphases + phaseid]
    + prior->p_coeff_bias[siteid];

  if (is_detected)
    logprob = - log(1 + exp(-logodds));
  else
    logprob = - log(1 + exp(logodds));

  if (!isfinite(logprob))
  {
    printf("depth %.1lf mag dist %.1lf %.1lf siteid %d phaseid %d "
           "isdet %d logodds %lg\n", 
           evdepth, evmag, dist, siteid, phaseid, is_detected,logodds);
    exit(1);
  }
  
  return logprob;
}

void EventDetectionPrior_UnInit(EventDetectionPrior_t * prior)
{
  free(prior->p_coeff_mag);
  free(prior->p_coeff_phases);
  free(prior->p_coeff_bias);
}

