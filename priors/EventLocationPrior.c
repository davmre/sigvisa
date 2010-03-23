#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

void EventLocationPrior_Init_Params(EventLocationPrior_t * dist, int nparams,
                                    const char * filename)
{
  FILE * fp;
  double discount;
  int lonidx;
  int latidx;
  
  assert(nparams == 1);
  
  fp = fopen(filename, "r");

  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (1 != fscanf(fp, "%lg\n", &discount))
  {
    fprintf(stderr, "error reading discount from %s\n", filename);
    exit(1);
  }
  
  if (2 != fscanf(fp, "%lg %lg\n", &dist->lonstep, &dist->zstep))
  {
    fprintf(stderr, "error reading lonstep and zstep from %s\n", filename);
    exit(1);
  }

  dist->numlon = (int) (360.0 / dist->lonstep);
  dist->numlat = (int) (2.0 / dist->zstep);

  dist->p_bucketprob = (double *)calloc(dist->numlon * dist->numlat, 
                                        sizeof(*dist->p_bucketprob));
  
  for (lonidx = 0; lonidx < dist->numlon; lonidx ++)
  {
    for (latidx = 0; latidx < dist->numlat; latidx ++)
    {
      char * fmt;
      
      if (latidx == (dist->numlat-1))
        fmt = "%lg\n";
      else
        fmt = "%lg ";
      
      if (1 != fscanf(fp, "%lg ",
                      &dist->p_bucketprob[lonidx * dist->numlat + latidx]))
      {
        fprintf(stderr, "error reading %s at row %d col %d\n",
                filename, lonidx, latidx);
        exit(1);
      }
    }
  }

  fclose(fp);
}

void EventLocationPrior_UnInit(EventLocationPrior_t * dist)
{
  free(dist->p_bucketprob);
}


double EventLocationPrior_LogProb(const EventLocationPrior_t * dist,
                                  const double * location, int nargs)
{
  double lon;
  double lat;
  double depth;
  
  int lonidx;
  int latidx;
  
  assert(nargs==0);

  if (location[0] == 180)
    lon = -180;
  else
    lon = location[0];
  
  if (location[1] == 90)
    lat = 89;
  else
    lat = location[1];
  
  depth = location[2];

  /* the event location better be valid */
  if ((lon < -180) || (lon > 180) || (lat < -90) || (lat > 90) 
      || (depth < MIN_DEPTH) || (depth > MAX_DEPTH))
  {
    printf("lon %lf lat %lf depth %lf\n", lon, lat, depth);
    exit(1);
    
    return -HUGE_VAL;
  }
  

  lonidx = (int) ((lon - (-180)) / dist->lonstep);
  latidx = (int) ((LAT2Z(lat) - (-1)) / dist->zstep);

  return log(dist->p_bucketprob[lonidx * dist->numlat + latidx])\
    - log(MAX_DEPTH-MIN_DEPTH);
}

