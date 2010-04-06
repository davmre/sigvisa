#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

void ArrivalSlownessPrior_Init_Params(ArrivalSlownessPrior_t * prior,
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

  for (siteid = 0; siteid < prior->numsites; siteid ++)
  {
    for (phaseid = 0; phaseid < prior->numphases; phaseid ++)
    {
      if (2 != fscanf(fp, "%lg %lg\n", 
                      &prior->locs[siteid * prior->numphases + phaseid],
                      &prior->scales[siteid * prior->numphases + phaseid]))
      {
        fprintf(stderr, "ArrivalSlownessPrior: can't read loc, scale for "
                "siteid %d phaseid %d in file %s", siteid, phaseid, filename);
        exit(1);
      }
    }
  }
  
  fclose(fp);
}


double ArrivalSlownessPrior_LogProb(const ArrivalSlownessPrior_t * prior,
                                    double arrslo, double pred_arrslo,
                                    double det_delslo, int siteid, int phaseid)
{
  double loc;
  double scale;
  double res;
  
  assert((siteid < prior->numsites) && (phaseid < prior->numphases));
  
  loc = prior->locs[siteid * prior->numphases + phaseid];
  scale = prior->scales[siteid * prior->numphases + phaseid] + det_delslo;

  res = arrslo - pred_arrslo;
  
  return Laplace_ldensity(loc, scale, res);
}


void ArrivalSlownessPrior_UnInit(ArrivalSlownessPrior_t * prior)
{
  free(prior->locs);
  free(prior->scales);
}

