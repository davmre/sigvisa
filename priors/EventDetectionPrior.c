#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

#define SKIP_TO_CHAR(fp, ch) while(fgetc(fp) != (ch))

void EventDetectionPrior_Init_Params(EventDetectionPrior_t * prior, 
                                     const char * filename)
{
  FILE * fp;
  int siteid;
  int phaseid;
  int i;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (2 != fscanf(fp, "%d %d\n", &prior->numtimedefphases, &prior->numsites))
  {
    fprintf(stderr, "error reading num sites and phases from %s\n", filename);
    exit(1);
  }

  /* skip a line */
  SKIP_TO_CHAR(fp, '\n');
  
  prior->p_coeffs = (double *)malloc(sizeof(*prior->p_coeffs)
                                     * prior->numsites
                                     * prior->numtimedefphases
                                     * NUM_EDA_COEFFS);

  for (phaseid = 0; phaseid < prior->numtimedefphases; phaseid ++)
  {
    for (siteid = 0; siteid < prior->numsites; siteid ++)
    {
      char * fmt;
      double coeff;
      
      /* skip the phase name */
      SKIP_TO_CHAR(fp, ',');
      /* skip the site id */
      SKIP_TO_CHAR(fp, ',');

      for (i=0; i<NUM_EDA_COEFFS; i++)
      {
        if ((NUM_EDA_COEFFS-1) == i)
          fmt = "%lg\n";
        else
          fmt = "%lg,";
        
        if (1 != fscanf(fp, fmt, &coeff))
        {
          fprintf(stderr, "error reading coeff %d for phaseid %d siteid %d in "
                  "file %s\n", i, phaseid, siteid, filename);
          exit(1);
        }

        prior->p_coeffs[(siteid * prior->numtimedefphases + phaseid)
                        * NUM_EDA_COEFFS + i] = coeff;
      }
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
  double * p_coeffs;

  assert((phaseid < prior->numtimedefphases) && (siteid < prior->numsites));
  
  p_coeffs = prior->p_coeffs 
    + (siteid * prior->numtimedefphases + phaseid) * NUM_EDA_COEFFS;

  logodds = p_coeffs[EDA_COEFF_INTERCEPT]
    + p_coeffs[EDA_COEFF_MAG] * evmag
    + p_coeffs[EDA_COEFF_DEPTH] * evdepth
    + p_coeffs[EDA_COEFF_DIST] * dist
    + p_coeffs[EDA_COEFF_DIST0] * Gaussian_prob(dist, 0, 5)
    + p_coeffs[EDA_COEFF_DIST35] * Gaussian_prob(dist, 35, 20)
    + p_coeffs[EDA_COEFF_DIST40] * Gaussian_prob(dist, 40, 20)
    + p_coeffs[EDA_COEFF_DIST12520] * Gaussian_prob(dist, 125, 20)
    + p_coeffs[EDA_COEFF_DIST12540] * Gaussian_prob(dist, 125, 40)
    + p_coeffs[EDA_COEFF_DIST145] * Gaussian_prob(dist, 145, 10)
    + p_coeffs[EDA_COEFF_DIST170] * Gaussian_prob(dist, 170, 20)
    + p_coeffs[EDA_COEFF_DIST175] * Gaussian_prob(dist, 175, 30)
    + p_coeffs[EDA_COEFF_MAG6] * Gaussian_prob(evmag, 6, 5.5)
    + p_coeffs[EDA_COEFF_MAG68] * Gaussian_prob(evmag, 6, 8)
    + p_coeffs[EDA_COEFF_MD] * (7 - evmag) * dist;
  
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
  free(prior->p_coeffs);
}

