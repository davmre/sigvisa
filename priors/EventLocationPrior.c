#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

void EventLocationPrior_Init_Params(EventLocationPrior_t * dist,
                                    const char * filename)
{
  FILE * fp;
  double discount;
  int lonidx;
  int latidx;
  
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

  dist->p_lonprob = (double *)calloc(dist->numlon, sizeof(*dist->p_lonprob));
  dist->p_latprob = (double *)calloc(dist->numlon * dist->numlat,
                                     sizeof(*dist->p_latprob));

  for (lonidx = 0; lonidx < dist->numlon; lonidx ++)
  {
    dist->p_lonprob[lonidx] = 0;
    
    for (latidx = 0; latidx < dist->numlat; latidx ++)
    {
      dist->p_lonprob[lonidx] += dist->p_bucketprob[lonidx * dist->numlat 
                                                    + latidx];
      
    }

    for (latidx = 0; latidx < dist->numlat; latidx ++)
    {
      dist->p_latprob[lonidx * dist->numlat + latidx] 
        = dist->p_bucketprob[lonidx * dist->numlat + latidx]
        / dist->p_lonprob[lonidx];
    }
  }
}

void EventLocationPrior_UnInit(EventLocationPrior_t * dist)
{
  free(dist->p_bucketprob);
  free(dist->p_lonprob);
  free(dist->p_latprob);
}


double EventLocationPrior_LogProb(const EventLocationPrior_t * dist,
                                  double lon, double lat, double depth)
{
  int lonidx;
  int latidx;
  
  if (lon == 180)
    lon = -180;
  
  if (lat == 90)
    lat = 89;

  /* the event location better be valid */
  if ((lon < -180) || (lon > 180) || (lat < -90) || (lat > 90) 
      || (depth < MIN_DEPTH) || (depth > MAX_DEPTH))
  {
    printf("Error: invalid location -- lon %lf lat %lf depth %lf\n",
           lon, lat, depth);
    exit(1);
    
    return -HUGE_VAL;
  }
  

  lonidx = (int) floor((lon - (-180)) / dist->lonstep);
  latidx = (int) floor((LAT2Z(lat) - (-1)) / dist->zstep);

  /* the probability density at any point in the bucket is uniform within the
   * bucket so we divide the bucket probability by the area of the bucket */
  return log(dist->p_bucketprob[lonidx * dist->numlat + latidx])\
    - log(dist->lonstep) - log(dist->zstep) - log(MAX_DEPTH-MIN_DEPTH);
}

static int sample_vec(int veclen, double * p_probvec)
{
  int i;
  double tgt;
  double cum;
  
  tgt = RAND_DOUBLE;
  
  cum = 0.0;
  for (i=0; (i<veclen); i++)
  {
    cum += p_probvec[i];

    if (cum >= tgt)
      break;
  }
  
  assert(i < veclen);
  
  return i;
}

void EventLocationPrior_Sample(const EventLocationPrior_t * dist,
                               double * p_lon, double * p_lat, 
                               double * p_depth)
{
  int lonidx;
  int latidx;
  
  lonidx = sample_vec(dist->numlon, dist->p_lonprob);
  *p_lon = -180 + (lonidx + RAND_DOUBLE) * dist->lonstep;

  latidx = sample_vec(dist->numlat, dist->p_latprob + lonidx * dist->numlat);
  *p_lat = RAD2DEG * asin(-1 + (latidx + RAND_DOUBLE) * dist->zstep);

  *p_depth = RAND_UNIFORM(MIN_DEPTH, MAX_DEPTH);
}

