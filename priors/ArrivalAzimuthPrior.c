#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../sigvisa.h"

void ArrivalAzimuthPrior_Init_Params(ArrivalAzimuthPrior_t * prior,
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
        fprintf(stderr, "ArrivalAzimuthPrior: can't read loc, scale for "
                "siteid %d phaseid %d in file %s", siteid, phaseid, filename);
        exit(1);
      }
    }
  }
  
  fclose(fp);
}


double ArrivalAzimuthPrior_LogProb(const ArrivalAzimuthPrior_t * prior,
                                   double arraz, double pred_arraz,
                                   double det_delaz, int siteid, int phaseid)
{
  double loc;
  double scale;
  double res;
  
  assert((siteid < prior->numsites) && (phaseid < prior->numphases));
  
  loc = prior->locs[siteid * prior->numphases + phaseid];
  scale = prior->scales[siteid * prior->numphases + phaseid] + det_delaz;

  res = EarthModel_DiffAzimuth(pred_arraz, arraz);
  
  return Laplace_ldensity(loc, scale, res);
}


void ArrivalAzimuthPrior_UnInit(ArrivalAzimuthPrior_t * prior)
{
  free(prior->locs);
  free(prior->scales);
}

