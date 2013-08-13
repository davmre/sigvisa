#include "cover_tree.hpp"
#include "vector_mult.hpp"

#include <cmath>
#include <unistd.h>

double dist_euclidean(const point p1, const point p2, double BOUND_IGNORED, const double *scales, void *dims) {
  return sqrt(sqdist_euclidean(p1.p, p2.p, BOUND_IGNORED, scales, dims));
}

double dist_euclidean(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales, const void *dims) {
  return sqrt(sqdist_euclidean(p1, p2, BOUND_IGNORED, scales, dims));
}

double sqdist_euclidean(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales, const void *dims) {
  int d = *(int *)dims;
  double sqdist = 0;
  double diff = 0;
  //printf("sqdist dim %d: ", d);
  for (int i=0; i < d; ++i) {
    diff = (p1[i] - p2[i]) / scales[i];
    sqdist += (diff * diff);
    //printf("+ %f*%f ", diff, diff);
  }
  //printf(" = %f\n", sqdist);
  return sqdist;
}


double euclidean_se_deriv_wrt_i(int i, const double * p1, const double * p2, const double *variance, const double *scales, const double *dims) {
  int d = *(int *)dims;
  double sqd = sqdist_euclidean(p1, p2, -1, scales, (const void*)dims);
  if (i==0) {
    return exp(-1 * sqd);
  } else if (i - 1 < d) {
    double diff = (p1[i-1] - p2[i-1]);
    return variance[0] * exp(-1 * sqd) * 2 * (diff*diff / pow(scales[i-1],3) );
  } else {
    printf("taking derivative wrt unrecognized parameter %d!\n", i);
    exit(-1);
    return 0;
  }
}


double pair_dist_euclidean(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales, void *dims) {
  double d1 = sqdist_euclidean(p1.pt1, p2.pt1, BOUND_IGNORED, scales, dims);
  double d2 = sqdist_euclidean(p1.pt2, p2.pt2, BOUND_IGNORED, scales, dims);
  return sqrt(d1 + d2);
}



static const double AVG_EARTH_RADIUS_KM = 6371.0;
static double RADIAN(double x) {return x*3.14159265/180.0;}
double dist_km(const double *p1, const double *p2) {

  double lon1 = p1[0];
  double lat1 = p1[1];
  double lon2 = p2[0];
  double lat2 = p2[1];
  double rlon1 = RADIAN(lon1);
  double rlat1 = RADIAN(lat1);
  double rlon2 = RADIAN(lon2);
  double rlat2 = RADIAN(lat2);

  /*
  double dist_rad = acos(sin(rlat1)
			* sin(rlat2)
			+ cos(rlat1)
			* cos(rlat2)
			* cos(rlon2 - rlon1));
			*/

  double dist_rad = asin(sqrt(
			     pow(sin((rlat1-rlat2)/2.0),2) +
			     cos(rlat1)*cos(rlat2)*
			     pow(sin((rlon1-rlon2)/2.0),2)
			     ));

  //printf("returning C:dist_km of (%f, %f) and (%f, %f) is %f\n", lon1, lat1, lon2, lat2, dist_rad * AVG_EARTH_RADIUS_KM);
  return dist_rad * AVG_EARTH_RADIUS_KM;
}

double dist_3d_km(const point p1, const point p2, double BOUND_IGNORED, const double *scales, void * extra) {
  return sqrt(distsq_3d_km(p1.p, p2.p, BOUND_IGNORED, scales, extra));
}

double dist_3d_km(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales, const void * extra) {
  return sqrt(distsq_3d_km(p1, p2, BOUND_IGNORED, scales, extra));
}

double distsq_3d_km(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales, const void * extra) {
  double distkm = dist_km(p1, p2) / scales[0];
  double dist_d = (p2[2] - p1[2]) / scales[1];
  //printf("dist3d returning sqrt(%f^2 + %f^2) = %f\n", distkm, dist_d, sqrt(pow(distkm, 2) + pow(dist_d, 2)));
  return pow(distkm, 2) + pow(dist_d, 2);
}



double dist_6d_km(const point p1, const point p2, double BOUND_IGNORED, const double *scales, void *extra) {
  return sqrt(distsq_6d_km(p1.p, p2.p, BOUND_IGNORED, scales, extra));  
}

double dist_6d_km(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales, const void *extra) {
  return sqrt(distsq_6d_km(p1, p2, BOUND_IGNORED, scales, extra));
}

double distsq_6d_km(const double * p1, const double * p2, double BOUND_IGNORED, const double *scales,const void *extra) {
  double sta_distkm = dist_km(p1, p2) / scales[0];
  double sta_dist_d = (p2[2] - p1[2]) / scales[1];
  double ev_distkm = dist_km(p1+3, p2+3) / scales[2];
  double ev_dist_d = (p2[5] - p1[5]) / scales[3];

  return pow(sta_distkm, 2) + pow(sta_dist_d, 2) + pow(ev_distkm, 2) + pow(ev_dist_d, 2);
}

double pair_dist_6d_km(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales, void * extra) {
  double sta_distkm1 = dist_km(p1.pt1, p2.pt1) / scales[0];
  double sta_distkm2 = dist_km(p1.pt2, p2.pt2) / scales[0];
  double ev_distkm1 = dist_km(p1.pt1, p2.pt1) / scales[2];
  double ev_distkm2 = dist_km(p1.pt2, p2.pt2) / scales[2];  

  double sta_dist_d1 = (p2.pt1[2] - p1.pt1[2]) / scales[1];
  double sta_dist_d2 = (p2.pt2[2] - p1.pt2[2]) / scales[1];
  double ev_dist_d1 = (p2.pt1[2] - p1.pt1[2]) / scales[1];
  double ev_dist_d2 = (p2.pt2[2] - p1.pt2[2]) / scales[1];

  return sqrt(pow(sta_distkm1, 2) + pow(sta_distkm2, 2) + pow(sta_dist_d1, 2) + pow(sta_dist_d2,2) + pow(ev_distkm1, 2) + pow(ev_distkm2, 2) + pow(ev_dist_d1, 2) + pow(ev_dist_d2, 2));
}



double pair_dist_3d_km(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales, void  * extra) {
  double distkm1 = dist_km(p1.pt1, p2.pt1) / scales[0];
  double distkm2 = dist_km(p1.pt2, p2.pt2) / scales[0];

  double dist_d1 = (p2.pt1[2] - p1.pt1[2]) / scales[1];
  double dist_d2 = (p2.pt2[2] - p1.pt2[2]) / scales[1];
  return sqrt(pow(distkm1, 2) + pow(distkm2, 2) + pow(dist_d1, 2) + pow(dist_d2,2));
}



double dist3d_se_deriv_wrt_i(int i, const double * p1, const double * p2,  const double *variance, const double *scales, const double *EXTRA_IGNORED) {
  double distkm = dist_km(p1, p2) / scales[0];
  double dist_d = (p2[2] - p1[2]) / scales[1];
  //printf("dist3d returning sqrt(%f^2 + %f^2) = %f\n", distkm, dist_d, sqrt(pow(distkm, 2) + pow(dist_d, 2)));
  double sqd =  distkm*distkm + dist_d * dist_d;

  if (i==0) { // deriv wrt variance
    return exp(-1 * sqd);
  } else if (i == 1) {
    return variance[0] * exp(-1 * sqd) * 2 * (distkm*distkm / scales[0] );
  } else if (i == 2) {
    return variance[0] * exp(-1 * sqd) * 2 * (dist_d * dist_d / scales[1] );
  } else {
    printf("taking derivative wrt unrecognized parameter %d!\n", i);
    exit(-1);
    return 0;
  }
}

double dist6d_se_deriv_wrt_i(int i, const double * p1, const double * p2, const double * variance, const double *scales, const double *EXTRA_IGNORED) {
  double sta_distkm = dist_km(p1, p2) / scales[0];
  double sta_dist_d = (p2[2] - p1[2]) / scales[1];
  double ev_distkm = dist_km(p1+3, p2+3) / scales[2];
  double ev_dist_d = (p2[5] - p1[5]) / scales[3];

  double sqd =  pow(sta_distkm, 2) + pow(sta_dist_d, 2) + pow(ev_distkm, 2) + pow(ev_dist_d, 2);

  if (i==0) {
    return exp(-1*sqd);
  } else if (i == 1) {
    return variance[0] * exp(-1 * sqd) * 2 * (sta_distkm*sta_distkm / scales[0]); 
  } else if (i == 2) {
    return variance[0] * exp(-1 * sqd) * 2 * (sta_dist_d*sta_dist_d / scales[1]);
  } else if (i == 3) {
    return variance[0] * exp(-1 * sqd) * 2 * (ev_distkm*ev_distkm / scales[2]);
  } else if (i == 4) {
    return variance[0] * exp(-1 * sqd) * 2 * (ev_dist_d*ev_dist_d / scales[3]);
  } else{
    printf("taking derivative wrt unrecognized parameter %d!\n", i);
    return 0;
  }
}




double w_se(double d, const double * variance) {
  return variance[0] * exp(-1 * d*d);
}

double w_e(double d, const double * variance) {
  return variance[0] * exp(-1 * d);
}

double w_matern32(double d, const double * variance) {
  double sqrt3 = 1.73205080757;
  return variance[0] * (1 + sqrt3*d) * exp(-sqrt3 * d);
}

double w_matern32_lower(double d, const double *variance) {
  double sqrt3 = 1.73205080757;
  return variance[0] * (1 + sqrt3*d) * exp(-sqrt3 * d);
}

double w_matern32_upper(double d, const double *variance) {
  double sqrt3 = 1.73205080757;
  return variance[0] * (1 + sqrt3*d + .75*d*d) * exp(-sqrt3 * d);
}
