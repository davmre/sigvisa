
#include "cover_tree.hpp"
#include "vector_mult.hpp"

#include <cmath>


double pair_distance(const point pt1, const point pt2, double BOUND_IGNORED, const double * PARAMS_IGNORED) {
  double a = sqrt(pow(pt2.p[1] - pt1.p[1], 2) + pow(pt2.p[0] - pt1.p[0], 2));
  return a;
}


const double AVG_EARTH_RADIUS_KM = 6371.0;
inline double RADIAN(double x) {return x*3.14159265f/180.0f;}
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

double dist_3d_km(const point p1, const point p2, double BOUND_IGNORED, const double *scales) {
  double distkm = dist_km(p1.p, p2.p) * scales[0];
  double dist_d = (p2.p[2] - p1.p[2]) * scales[1];
  //printf("dist3d returning sqrt(%f^2 + %f^2) = %f\n", distkm, dist_d, sqrt(pow(distkm, 2) + pow(dist_d, 2)));
  return sqrt(pow(distkm, 2) + pow(dist_d, 2));
}

double pair_dist_3d_km(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales) {
  double distkm1 = dist_km(p1.pt1, p2.pt1) * scales[0];
  double distkm2 = dist_km(p1.pt2, p2.pt2) * scales[0];

  double dist_d1 = (p2.pt1[2] - p1.pt1[2]) * scales[1];
  double dist_d2 = (p2.pt2[2] - p1.pt2[2]) * scales[1];
  return sqrt(distkm1*distkm1 + distkm2*distkm2 + dist_d1*dist_d1 + dist_d2*dist_d2);
}

double w_se(double d, const double * variance) {
  return variance[0] * exp(-1 * pow(d,2));
}
