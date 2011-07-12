#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

#define UNIF_FALSE 0.1          /* uniform prior on false amplitude */

void ArrivalAmplitudePrior_Init_Params(ArrivalAmplitudePrior_t * prior,
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

  if (5 != fscanf(fp, "%d %d %lg %lg %lg\n",
                  &prior->numsites, &prior->numphases, &prior->min_logamp,
                  &prior->max_logamp, &prior->step_logamp))
  {
    fprintf(stderr, "error reading num sites, num phases from %s\n", filename);
    exit(1);
  }

  prior->numstep = (int) ((prior->max_logamp - prior->min_logamp)
                          /prior->step_logamp);
  

  prior->p_site_false = (double *) calloc(prior->numsites * prior->numstep,
                                          sizeof(*prior->p_site_false));
  prior->p_site_phase_amp = (PhaseAmp_t *) calloc(prior->numsites 
                                                  * prior->numphases,
                                             sizeof(*prior->p_site_phase_amp));
  
  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    double * p_false;
    int i;
    
    p_false = prior->p_site_false + siteid * prior->numstep;
    
    for (i=0; i<prior->numstep; i++)
    {
      if (1 != fscanf(fp, "%lg ", p_false + i))
      {
        fprintf(stderr, "error reading false amp density for site %d\n",
                siteid);
        exit(1);
      }
    }
    
    for (phaseid = 0; phaseid < prior->numphases; phaseid ++)
    {
      PhaseAmp_t * p_phase;
      
      p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;
    
      if (5 != fscanf(fp, "%lg %lg %lg %lg %lg\n", &p_phase->intercept,
                      &p_phase->mb_coeff, &p_phase->depth_coeff,
                      &p_phase->ttime_coeff, &p_phase->std))
      {
        fprintf(stderr, "Error reading amp model for site %d phase %d\n", 
                siteid, phaseid);
        exit(1);
      }
    }
  }
  
  fclose(fp);
}

double ArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * prior,
                                     double mb, double depth, double ttime,
                                     int siteid, int phaseid, double amp)
{
  double logamp;
  PhaseAmp_t * p_phase;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert((phaseid >= 0) && (phaseid < prior->numphases));
  assert(amp > 0);
  
  logamp = log(amp);

  p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;

  return Gaussian_logprob(logamp, p_phase->intercept + p_phase->mb_coeff * mb
                          + p_phase->depth_coeff * depth
                          + p_phase->ttime_coeff * ttime, p_phase->std);
}

double FalseArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * 
                                          prior, int siteid, double amplitude)
{
  double logamp;
  int idx;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  
  assert(amplitude > 0);
  
  logamp = log(amplitude);
  
  assert((logamp > prior->min_logamp) && (logamp < prior->max_logamp));
  
  idx = (int) ((logamp - prior->min_logamp) / prior->step_logamp);
  
  return log(prior->p_site_false[siteid * prior->numstep + idx]
             * (1-UNIF_FALSE) + (1.0/10.0)*UNIF_FALSE);
}

double FalseArrivalAmplitudePrior_cdf(const ArrivalAmplitudePrior_t * 
                                      prior, int siteid, double amplitude)
{
  double logamp;
  int idx;
  int i;
  double sum;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  
  assert(amplitude > 0);
  
  logamp = log(amplitude);
  
  assert((logamp > prior->min_logamp) && (logamp < prior->max_logamp));
  
  idx = (int) ((logamp - prior->min_logamp) / prior->step_logamp);

  sum = 0;
  
  for (i=0; i<=idx; i++)
    sum += prior->p_site_false[siteid * prior->numstep + i];

  sum *= prior->step_logamp;
  
  return sum;
}


void ArrivalAmplitudePrior_UnInit(ArrivalAmplitudePrior_t * prior)
{
  free(prior->p_site_false);
  free(prior->p_site_phase_amp);
}
