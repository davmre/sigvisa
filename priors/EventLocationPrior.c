/*
 * Copyright (c) 2012, Bayesian Logic, Inc.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Bayesian Logic, Inc. nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 * Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"

#define BUCKET(loni, lati) dist->p_bucketprob[(loni) * dist->numlat + (lati)]

void EventLocationPrior_Init_Params(EventLocationPrior_t * dist,
                                    const char * filename)
{
  FILE * fp;
  double uniform;
  int lonidx;
  int latidx;
  double cum_lon_prob;
  
  fp = fopen(filename, "r");

  if (!fp)
  {
    fprintf(stderr, "Unable to open file: %s\n", filename);
    exit(1);
  }

  if (1 != fscanf(fp, "%lg\n", &uniform))
  {
    fprintf(stderr, "error reading uniform prior from %s\n", filename);
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

  cum_lon_prob = 0;
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

    cum_lon_prob += dist->p_lonprob[lonidx];
  }

  for (lonidx = 0; lonidx < dist->numlon; lonidx ++)
  {
    dist->p_lonprob[lonidx] /= cum_lon_prob;
  }

  /* initialize the probability of a hypothetical bucket at the north-pole,
   * this is useful for interpolating */
  dist->north_pole_prob = 0;
  for (lonidx = 0; lonidx < dist->numlon; lonidx ++)
    dist->north_pole_prob += dist->p_bucketprob[lonidx * dist->numlat 
                                                + dist->numlat-1];
  dist->north_pole_prob /= dist->numlon;
}

void EventLocationPrior_UnInit(EventLocationPrior_t * dist)
{
  free(dist->p_bucketprob);
  free(dist->p_lonprob);
  free(dist->p_latprob);
}

/* interpolate the value inside a rectangle given the values at the
 * four corners lower-left, lower-right, upper-left, upper-right and
 * the x, y values plus the x_size and y_size of the rectangle
 * (assuming 0,0 is lower-left)
 */
double interpolate_rect(double ll_val, double lr_val, double ul_val,
                        double ur_val, double x, double y,
                        double x_max, double y_max)
{
  double ll_wt, lr_wt, ul_wt, ur_wt;

  ll_wt = ((x_max - x) / x_max) * ((y_max - y) / y_max);
  lr_wt = (    x       / x_max) * ((y_max - y) / y_max);
  ul_wt = ((x_max - x) / x_max) * (    y       / y_max);
  ur_wt = (    x       / x_max) * (    y       / y_max);


  return (ll_wt * ll_val + lr_wt * lr_val + ul_wt * ul_val + ur_wt * ur_val)
    / (ll_wt + lr_wt + ul_wt + ur_wt);
}


double EventLocationPrior_LogProb(const EventLocationPrior_t * dist,
                                  double lon, double lat, double depth)
{
  double val;
  
  /* the event location better be valid */
  if ((lon < -180) || (lon > 180) || (lat < -90) || (lat > 90) 
      || (depth < MIN_DEPTH) || (depth > MAX_DEPTH))
  {
    printf("Error: invalid location -- lon %lf lat %lf depth %lf\n",
           lon, lat, depth);
    exit(1);
    
    return -HUGE_VAL;
  }

  if (lat == 90)
    val = dist->north_pole_prob;

  else
  {
    int lonidx, lonidx2;
    int latidx, latidx2;

    double lon_rem, z_rem;

    if (lon == 180)
      lon = -180;

    lonidx = (int) floor((lon - (-180)) / dist->lonstep);
    latidx = (int) floor((LAT2Z(lat) - (-1)) / dist->zstep);
    
    lon_rem = lon - (-180) - (lonidx * dist->lonstep);
    z_rem = LAT2Z(lat) - (-1) - (latidx * dist->zstep);

    lonidx2 = (lonidx + 1) % dist->numlon;
    
    if ((latidx+1) < dist->numlat)
    {
      latidx2 = latidx + 1;
    
      val = interpolate_rect(BUCKET(lonidx, latidx), BUCKET(lonidx2, latidx),
                             BUCKET(lonidx, latidx2), BUCKET(lonidx2, latidx2),
                             lon_rem, z_rem, dist->lonstep, dist->zstep);

    }
    else
    {
      val = interpolate_rect(BUCKET(lonidx, latidx), BUCKET(lonidx2, latidx),
                             dist->north_pole_prob, dist->north_pole_prob,
                             lon_rem, z_rem, dist->lonstep, dist->zstep);
    }
  }
  
  /* the probability density at any point in the bucket is uniform within the
   * bucket so we divide the bucket probability by the area of the bucket */
  return log(val)\
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
  
  if (i >= veclen)
  {
    printf("sample_vec: tgt=%f cum=%f\n", tgt, cum);
    exit(1);
  }
  
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

