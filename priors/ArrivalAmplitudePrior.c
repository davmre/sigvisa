#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../sigvisa.h"

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

  if (2 != fscanf(fp, "%d %d\n", &prior->numsites, &prior->numphases))
  {
    fprintf(stderr, "error reading num sites, num phases from %s\n", filename);
    exit(1);
  }

  prior->p_site_false = (SiteFalseAmp_t *) calloc(prior->numsites,
                                              sizeof(*prior->p_site_false));
  prior->p_site_phase_amp = (PhaseAmp_t *) calloc(prior->numsites 
                                                  * prior->numphases,
                                             sizeof(*prior->p_site_phase_amp));
  
  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    SiteFalseAmp_t * p_false;
    
    p_false = prior->p_site_false + siteid;
    
    if (6 != fscanf(fp, "%lg %lg %lg %lg %lg %lg\n", &p_false->wt0,
                    &p_false->wt1, &p_false->mean0, &p_false->mean1,
                    &p_false->std0, &p_false->std1))
    {
      fprintf(stderr, "Error reading false amp model for site %d\n", siteid);
      exit(1);
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

  if (amp < 0)
    amp = 1;
  else if (amp > MAX_AMP)
    amp = MAX_AMP;
  
  logamp = log(amp);

  p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;

  return Gaussian_logprob(logamp, p_phase->intercept + p_phase->mb_coeff * mb
                          + p_phase->depth_coeff * depth
                          + p_phase->ttime_coeff * ttime, p_phase->std);
}

double ArrivalAmplitudePrior_Point(const ArrivalAmplitudePrior_t * prior,
                                     double mb, double depth, double ttime,
                                     int siteid, int phaseid)
{
  PhaseAmp_t * p_phase;
  
  assert((siteid >= 0) && (siteid < prior->numsites));
  assert((phaseid >= 0) && (phaseid < prior->numphases));

  p_phase = prior->p_site_phase_amp + siteid * prior->numphases + phaseid;

  return exp(p_phase->intercept + p_phase->mb_coeff * mb
	     + p_phase->depth_coeff * depth
	     + p_phase->ttime_coeff * ttime);
}

double FalseArrivalAmplitudePrior_LogProb(const ArrivalAmplitudePrior_t * 
                                          prior, int siteid, double amplitude)
{
  double logamp;
  SiteFalseAmp_t * p_false;
  
  assert((siteid >= 0) && (siteid < prior->numsites));

  if (amplitude < 0)
    amplitude = 1;
  else if (amplitude > MAX_AMP)
    amplitude = MAX_AMP;
  
  logamp = log(amplitude);
  
  p_false = prior->p_site_false + siteid;
  
  /* std could be 0 so we add 1e-6
   * we mix with a uniform distribution to avoid extreme values when
   * the amplitude is far from the mean */
  return log(p_false->wt0 * Gaussian_prob(logamp, p_false->mean0, 
                                          p_false->std0 + 1e-6) *(1-UNIF_FALSE)
             + p_false->wt1 * Gaussian_prob(logamp, p_false->mean1, 
                                            p_false->std1 +1e-6)*(1-UNIF_FALSE)
             + UNIF_FALSE / LOG_MAX_AMP);
}

void ArrivalAmplitudePrior_UnInit(ArrivalAmplitudePrior_t * prior)
{
  free(prior->p_site_false);
  free(prior->p_site_phase_amp);
}
