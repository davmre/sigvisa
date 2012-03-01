#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "SecDetPrior.h"
#include "Laplace.h"
#include "Gamma.h"

#define READPARAM(cnt, fncall, name)\
  do {                              \
  if ((cnt) != (fncall))            \
  {                     \
  fprintf(stderr, "error reading " #name " from %s\n", filename);  \
  exit(1);                                                              \
  }                                                                     \
  } while(0)

static double degdiff(double angle1, double angle2)
{
  /*
  The difference of two angles given in degrees. The answer is an angle from
  -180 to 180. Positive angles imply angle2 is clockwise from angle1 and -ve
  angles imply counter-clockwise.
  
  >>> int(degdiff(40, 30))
  -10
  >>> int(degdiff(30, 40))
  10
  >>> int(degdiff(361, 40))
  39
  >>> int(degdiff(40, 361))
  -39
  >>> degdiff(40,250)
  -150
  >>> degdiff(40,200)
  160
  >>> degdiff(40, 219)
  179
  >>> degdiff(40, 220)
  180
  >>> degdiff(40, 221)
  -179
  */
  /* bring the angle into the 0 to 360 range */
  double delta = fmod((angle2 - angle1) + 360, 360.0);
  
  /* angles above 180 need to be shifted down by 360 degrees so that 181 is -179
   * 200 is -160 etc. */
  return delta - (delta > 180 ? 360 : 0);
}

void SecDetPrior_Init_Params(SecDetPrior_t * prior, const char * filename)
{
  FILE * fp;
  int i;
  
  fp = fopen(filename, "r");
  
  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  READPARAM(3, fscanf(fp, "%d %lg %lg\n", &prior->num_logamp_bins,
                      &prior->low_logamp, &prior->step_logamp), 
            "number of logamp bins, low logamp, step logamp");

  prior->detprob = (double *)malloc(sizeof(*prior->detprob) 
                                       * prior->num_logamp_bins);
  for (i=0; i<prior->num_logamp_bins; i++)
    READPARAM(1, fscanf(fp, "%lg ", prior->detprob + i),
              "coda probability");

  READPARAM(4, fscanf(fp, "%lg %lg %lg %lg\n", &prior->time_shape,
                      &prior->min_delay, &prior->time_scale,
                      &prior->max_delay),
            "coda time shape, location, scale, maximum");

  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->az_loc, &prior->az_scale),
            "coda azimuth location and scale");

  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->slo_loc, &prior->slo_scale),
            "coda slowness location and scale");

  READPARAM(2, fscanf(fp, "%lg %lg\n", &prior->logamp_loc,
                      &prior->logamp_scale),
            "coda logamp location and scale");
  
  READPARAM(3, fscanf(fp, "%lg %lg %lg\n", &prior->snr_pos_prob,
                      &prior->snr_pos_lambda, &prior->snr_neg_lambda),
            "coda SNR location and scale");
  
  READPARAM(1, fscanf(fp, "%d\n", &prior->numphases), "number of phases");

  prior->phaseprob = (double *)malloc(prior->numphases 
                                         * sizeof(*prior->phaseprob));
  
  for (i=0; i<prior->numphases; i++)
    READPARAM(1, fscanf(fp, "%lg ", prior->phaseprob + i),
              "coda phase probability");
  
  fclose(fp);
}

void SecDetPrior_UnInit(SecDetPrior_t * prior)
{
  free(prior->detprob);
  free(prior->phaseprob);
}

double SecDetPrior_Det_LogProb(const SecDetPrior_t * prior, int is_det,
                                        double prim_amp)
{
  int index;
  
  /* if prim_amp is -1 we will simply use the min bucket */
  if (prim_amp <= 0)
    index = 0;
  
  else
  {
    index = (int)floor((log(prim_amp) - prior->low_logamp)/prior->step_logamp);
    
    /* keep the index within range */
    if (index < 0)
      index = 0;

    if (index >= prior->num_logamp_bins)
      index = prior->num_logamp_bins - 1;
  }
  
  if (is_det)
    return log(prior->detprob[index]);
  
  else
    return log(1-prior->detprob[index]);
}

int SecDetPrior_Time_Possible(const SecDetPrior_t * prior, double sec_time,
                              double prim_time)
{
  double delta = sec_time - prim_time;

  return (delta <= prior->max_delay) && (delta > prior->min_delay) ? 1 : 0;
}

double SecDetPrior_Time_LogProb(const SecDetPrior_t * prior,
                                         double sec_time, double prim_time)
{
  double delta = sec_time - prim_time;
  assert((delta <= prior->max_delay) && (delta > prior->min_delay));
  
  return Gamma_logprob(delta - prior->min_delay, prior->time_shape,
                       prior->time_scale);
}

double SecDetPrior_Slow_LogProb(const SecDetPrior_t * prior,
                                         double sec_slow, double prim_slow)
{
  return Laplace_ldensity(prior->slo_loc, prior->slo_scale, 
                          sec_slow - prim_slow);
}

double SecDetPrior_Azimuth_LogProb(const SecDetPrior_t * prior,
                                            double sec_az, double prim_az)
{
  return Laplace_ldensity(prior->az_loc, prior->az_scale,
                          degdiff(prim_az, sec_az));
}

double SecDetPrior_Phase_LogProb(const SecDetPrior_t * prior,
                                          int sec_phaseid)
{
  assert(sec_phaseid < prior->numphases);
  
  return log(prior->phaseprob[sec_phaseid]);
}

double SecDetPrior_Amp_LogProb(const SecDetPrior_t * prior,
                                        double sec_amp, double prim_amp)
{
  return Laplace_ldensity(prior->logamp_loc, prior->logamp_scale, 
                          log(sec_amp) - log(prim_amp));
}

double SecDetPrior_SNR_LogProb(const SecDetPrior_t * prior,
                                        double sec_snr, double prim_snr)
{
  double delta = sec_snr - prim_snr;
  
  if (delta >= 0)
    return log(prior->snr_pos_prob) + log(prior->snr_pos_lambda) 
      - prior->snr_pos_lambda * delta;
  else
    return log(1-prior->snr_pos_prob) + log(prior->snr_neg_lambda)
      + prior->snr_neg_lambda * delta;
}
