#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

void EventDetectionPrior_Init_Params(EventDetectionPrior_t * prior, 
                                     int nparams, const char * filename)
{
  FILE * fp;
  int siteid;
  int phaseid;
  
  assert(nparams == 1);
  
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

  prior->p_site_mag = (double *)malloc(sizeof(*prior->p_site_mag)
                                        * prior->numsites);

  prior->p_site_phases = (double *)malloc(sizeof(*prior->p_site_phases)
                                          * prior->numsites 
                                          * prior->numphases);

  prior->p_site_bias = (double *)malloc(sizeof(*prior->p_site_bias)
                                        * prior->numsites);
  
  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    if (1 != fscanf(fp, "%lg ", &prior->p_site_mag[siteid]))
    {
      fprintf(stderr, "error reading mag-coeff for siteid %d in file %s\n",
              siteid, filename);
    }
    for (phaseid = 0; phaseid < prior->numphases; phaseid ++)
    {
      if (1 != fscanf(fp, "%lg ",&prior->p_site_phases[siteid 
                                                       * prior->numphases
                                                       + phaseid]))
      {
        fprintf(stderr, "error reading phaseid %d for siteid %d in file %s\n",
                phaseid, siteid, filename);
      }
    }
    if (1 != fscanf(fp, "%lg\n", &prior->p_site_bias[siteid]))
    {
      fprintf(stderr, "error reading bias for siteid %d in file %s\n",
              siteid, filename);
    }
  }
  
  fclose(fp);
}

double EventDetectionPrior_LogProb(const EventDetectionPrior_t * prior,
                                   int is_detected, int nargs,
                                   int siteid, int phaseid, double evmag,
                                   double * evloc)
{
  double logodds;
  
  assert(nargs == 4);

  logodds = prior->p_site_mag[siteid] * evmag
    + prior->p_site_phases[siteid * prior->numphases + phaseid]
    + prior->p_site_bias[siteid];
  
  if (BLOG_TRUE == is_detected)
    return - log(1 + exp(-logodds));
  else if (BLOG_FALSE == is_detected)
    return - log(1 + exp(logodds));
  else
    assert(0);             /* is_detected is neither true nor false */  
}

void EventDetectionPrior_UnInit(EventDetectionPrior_t * prior)
{
  free(prior->p_site_mag);
  free(prior->p_site_phases);
  free(prior->p_site_bias);
}

