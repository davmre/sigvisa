#include "cover_tree.h"
#include "vector_mult.h"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>

#include <cmath>
#include <memory>
#include <vector>
#include <limits.h>
#include <values.h>

using namespace std;
namespace bp = boost::python;

const double AVG_EARTH_RADIUS_KM = 6371.0;
inline double RADIAN(double x) {return x*3.14159265f/180.0f;}
double dist_km(const point &p1, const point &p2, double BOUND_IGNORED, const double *PARAMS_IGNORED) {

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

double dist_3d_km(const point &p1, const point &p2, double BOUND_IGNORED, const double *scales) {
  double distkm = dist_km(p1, p2, -1, NULL) * scales[0];
  double dist_d = (p2[2] - p1[2]) * scales[1];
  //printf("dist3d returning sqrt(%f^2 + %f^2) = %f\n", distkm, dist_d, sqrt(pow(distkm, 2) + pow(dist_d, 2)));
  return sqrt(pow(distkm, 2) + pow(dist_d, 2));
}

double CoverTree::weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps, string wfn_str, const pyublas::numpy_vector<double> &weight_params) {
  point qp(query_pt.size2());
  for (unsigned int i = 0; i < query_pt.size2(); ++i) {
    qp[i] = query_pt(0,i);
  }

  wfn w;
  if (wfn_str.compare("se") == 0) {
    w = w_se;
  } else{
    printf("error: unrecognized weight function %s\n", wfn_str.c_str());
    exit(1);
  }

  double * wp = NULL;
  if (weight_params.size() > 0) {
    wp = new double[weight_params.size()];
    for (unsigned i = 0; i < weight_params.size(); ++i) {
      wp[i] = weight_params(i);
    }
  }

  double weight_sofar = 0;
  int fcalls = 0;
  this->root.distance_to_query = this->dfn(qp, this->root.p, MAXDOUBLE, this->dist_params);
  double ws = weighted_sum_node(this->root, v_select, 8,
			       qp, eps, weight_sofar,
			       fcalls, w, this->dfn, this->dist_params, wp);
  if (wp != NULL) {
    delete wp;
    wp = NULL;
  }
  this->fcalls = fcalls;
  return ws;
}

void CoverTree::set_v(int v_select, const pyublas::numpy_vector<double> &v) {
  if (v.ndim() != 1) {
    printf("error: tree can only hold 1D arrays! (array passed has %lu dimensions)\n", v.ndim());
    exit(1);
  }
  npy_intp item_stride = v.strides()[0] / v.itemsize();
  vector<double> new_v(v.size());
  for (pyublas::numpy_vector<double>::const_iterator a = v.begin(); a < v.end(); a += item_stride) {
    new_v[ (a - v.begin())/item_stride] = *a;
  }
  set_v_node(this->root, v_select, new_v);
}

pyublas::numpy_vector<double> CoverTree::get_v(int v_select) {
  vector<double> v(this->n);
  get_v_node(this->root, v_select, v);

  pyublas::numpy_vector<double> pv(this->n);
  for (unsigned int i = 0; i < this->n; ++i) {
    pv(i) = v[i];
  }

  return pv;
}

CoverTree::CoverTree (const pyublas::numpy_matrix<double> &pts,
			const string &distfn_str,
			const pyublas::numpy_vector<double> &dist_params) {
  vector< point > points(pts.size1(), point(pts.size2()));
  for (unsigned i = 0; i < pts.size1 (); ++ i)
    for (unsigned j = 0; j < pts.size2 (); ++ j)
      points[i][j] = pts (i, j);
  this->n = pts.size1();
  if (distfn_str.compare("pair") == 0) {
    this->dfn = pair_distance;
  } else if (distfn_str.compare("lld") == 0) {
    this->dfn = dist_3d_km;
  } else{
    printf("error: unrecognized distance function %s\n", distfn_str.c_str());
    exit(1);
  }

  this->dist_params = NULL;
  this->set_dist_params(dist_params);

  this->root = batch_create(points, this->dfn, this->dist_params);
}

void CoverTree::set_dist_params(const pyublas::numpy_vector<double> &dist_params) {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
  this->dist_params = new double[dist_params.size()];
  for (unsigned i = 0; i < dist_params.size(); ++i) {
    this->dist_params[i] = dist_params(i);
  }
}


pyublas::numpy_matrix<double> CoverTree::debug_kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, string wfn_str, const pyublas::numpy_vector<double> &weight_params, bool distance_only) {

  wfn w;
  if (wfn_str.compare("se") == 0) {
    w = w_se;
  } else{
    printf("error: unrecognized weight function %s\n", wfn_str.c_str());
    exit(1);
  }

  double * wp = NULL;
  if (weight_params.size() > 0) {
    wp = new double[weight_params.size()];
    for (unsigned i = 0; i < weight_params.size(); ++i) {
      wp[i] = weight_params(i);
    }
  }

  pyublas::numpy_matrix<double> K(pts1.size1(), pts2.size1());
  for (unsigned i = 0; i < pts1.size1 (); ++ i) {
    point p1(pts1.size2());
    for (unsigned int pi=0; pi < pts1.size2(); pi++) {
      p1[pi] = pts1(i, pi);
    }

    for (unsigned j = 0; j < pts2.size1 (); ++ j) {
      point p2(pts2.size2());
      for (unsigned int pi=0; pi < pts2.size2(); pi++) {
	p2[pi] = pts2(j, pi);
      }

      double d = this->dfn(p1, p2, MAXDOUBLE, this->dist_params);
      K(i,j) = distance_only ? d : w(d, wp);
    }
  }

  if (wp != NULL) {
    delete wp;
    wp = NULL;
  }
  return K;
}


CoverTree::~CoverTree() {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
}

BOOST_PYTHON_MODULE(cover_tree) {
  bp::class_<CoverTree>("CoverTree", bp::init< pyublas::numpy_matrix< double > const &, string const &, pyublas::numpy_vector< double > const &>())
    .def("set_v", &CoverTree::set_v)
    .def("get_v", &CoverTree::get_v)
    .def("weighted_sum", &CoverTree::weighted_sum)
    .def("set_dist_params", &CoverTree::weighted_sum)
    .def("debug_kernel_matrix", &CoverTree::debug_kernel_matrix)
    .def_readonly("fcalls", &CoverTree::fcalls);
}
