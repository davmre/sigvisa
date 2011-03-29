#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

void ArrivalTimePrior_Init_Params(ArrivalTimePrior_t * prior,
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

  prior->locs = (double *)calloc(prior->numsites * prior->numphases,
                                 sizeof(*prior->locs));
  
  prior->scales = (double *)calloc(prior->numsites * prior->numphases,
                                   sizeof(*prior->scales));

  prior->minvals = (double *)calloc(prior->numsites * prior->numphases,
                                    sizeof(*prior->minvals));
  
  prior->maxvals = (double *)calloc(prior->numsites * prior->numphases,
                                    sizeof(*prior->maxvals));
  
  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    for (phaseid = 0; phaseid < prior->numphases; phaseid ++)
    {
      if (4 != fscanf(fp, "%lg %lg %lg %lg\n", 
                      &prior->locs[siteid * prior->numphases + phaseid],
                      &prior->scales[siteid * prior->numphases + phaseid],
                      &prior->minvals[siteid * prior->numphases + phaseid],
                      &prior->maxvals[siteid * prior->numphases + phaseid]))
      {
        fprintf(stderr, "ArrivalTimePrior: can't read params for siteid %d"
                " phaseid %d in file %s", siteid, phaseid, filename);
        exit(1);
      }
    }
  }
  
  fclose(fp);
}


double ArrivalTimePrior_LogProb(const ArrivalTimePrior_t * prior,
                                double arrtime, double pred_arrtime,
                                double det_deltime, int siteid, int phaseid)
{
  double loc;
  double scale;
  double res;
  double minval;
  double maxval;
  
  assert((siteid < prior->numsites) && (phaseid < prior->numphases));
  
  loc = prior->locs[siteid * prior->numphases + phaseid];
  scale = prior->scales[siteid * prior->numphases + phaseid] + det_deltime;
  
  minval = prior->minvals[siteid * prior->numphases + phaseid];
  
  maxval = prior->minvals[siteid * prior->numphases + phaseid];
  
  res = arrtime - pred_arrtime;

  if ((res > maxval) || (res < minval))
    return -INFINITY;
  
  return Laplace_trunc_ldensity(loc, scale, minval, maxval, res);
}

double ArrivalTimePrior_MeanResidual(const ArrivalTimePrior_t * prior,
                                     int siteid, int phaseid)
{
  assert((siteid < prior->numsites) && (phaseid < prior->numphases));
  
  return prior->locs[siteid * prior->numphases + phaseid];
}


void ArrivalTimePrior_UnInit(ArrivalTimePrior_t * prior)
{
  free(prior->locs);
  free(prior->scales);
  free(prior->minvals);
  free(prior->maxvals);
}

