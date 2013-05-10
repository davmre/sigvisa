#include "cover_tree.hpp"
#include "vector_mult.hpp"

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

void set_v_node (node<point> &n, int v_select, const std::vector<double> &v) {
  if (n.num_children == 0) {
      n.unweighted_sums[v_select] = v[n.p.idx];
  } else {
    n.unweighted_sums[v_select] = 0;
    for(int i=0; i < n.num_children; ++i) {
      set_v_node(n.children[i], v_select, v);
      n.unweighted_sums[v_select] += n.children[i].unweighted_sums[v_select];
    }
  }
}

void get_v_node(node<point> &n, int v_select, std::vector<double> &v) {
  if (n.num_children == 0) {
    v[n.p.idx] = n.unweighted_sums[v_select];
  } else {
    for(int i=0; i < n.num_children; ++i) {
      get_v_node(n.children[i], v_select, v);
    }
  }
}


double VectorTree::weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps, string wfn_str, const pyublas::numpy_vector<double> &weight_params) {
  point qp = {&query_pt(0,0), 0};

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
  this->root.distance_to_query = this->dfn(qp, this->root.p, MAXDOUBLE, this->dist_params, NULL);
  double ws = weighted_sum_node(this->root, v_select,
				qp, eps, weight_sofar,
				fcalls, w, this->dfn, this->dist_params, NULL, wp);
  if (wp != NULL) {
    delete wp;
    wp = NULL;
  }
  this->fcalls = fcalls;
  return ws;
}

void VectorTree::set_v(int v_select, const pyublas::numpy_vector<double> &v) {
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

pyublas::numpy_vector<double> VectorTree::get_v(int v_select) {
  vector<double> v(this->n);
  get_v_node(this->root, v_select, v);

  pyublas::numpy_vector<double> pv(this->n);
  for (unsigned int i = 0; i < this->n; ++i) {
    pv(i) = v[i];
  }

  return pv;
}

VectorTree::VectorTree (const pyublas::numpy_matrix<double> &pts,
			const unsigned int narms,
			const string &distfn_str,
			const pyublas::numpy_vector<double> &dist_params) {
  vector< point > points(pts.size1());
  for (unsigned i = 0; i < pts.size1 (); ++ i) {
    point p = {&pts (i, 0), i};
    points[i] = p;
  }
  this->n = pts.size1();
  if (distfn_str.compare("lld") == 0) {
    this->dfn = dist_3d_km;
  } else{
    printf("error: unrecognized distance function %s\n", distfn_str.c_str());
    exit(1);
  }

  this->dist_params = NULL;
  this->set_dist_params(dist_params);

  this->root = batch_create(points, this->dfn, this->dist_params, NULL);
  this->root.alloc_arms(narms);
}

void VectorTree::set_dist_params(const pyublas::numpy_vector<double> &dist_params) {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
  this->dist_params = new double[dist_params.size()];
  for (unsigned i = 0; i < dist_params.size(); ++i) {
    this->dist_params[i] = dist_params(i);
  }
}


pyublas::numpy_matrix<double> VectorTree::kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, string wfn_str, const pyublas::numpy_vector<double> &weight_params, bool distance_only) {

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
    point p1 = {&pts1(i, 0), 0};
    for (unsigned j = 0; j < pts2.size1 (); ++ j) {
      point p2 = {&pts2(j, 0), 0};
      double d = this->dfn(p1, p2, MAXDOUBLE, this->dist_params, NULL);
      K(i,j) = distance_only ? d : w(d, wp);
    }
  }

  if (wp != NULL) {
    delete wp;
    wp = NULL;
  }
  return K;
}

pyublas::numpy_matrix<double> VectorTree::kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, string wfn_str, const pyublas::numpy_vector<double> &weight_params, int param_i) {
  if (wfn_str.compare("se") != 0) {
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
    point p1 = {&pts1(i, 0), 0};
    for (unsigned j = 0; j < pts2.size1 (); ++ j) {
      point p2 = {&pts2(j, 0), 0};
      K(i,j) = dist3d_se_deriv_wrt_i(param_i, p1, p2, wp, this->dist_params);
    }
  }

  if (wp != NULL) {
    delete wp;
    wp = NULL;
  }
  return K;
}



VectorTree::~VectorTree() {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
}

BOOST_PYTHON_MODULE(cover_tree) {
  bp::class_<VectorTree>("VectorTree", bp::init< pyublas::numpy_matrix< double > const &, int const, string const &, pyublas::numpy_vector< double > const &>())
    .def("set_v", &VectorTree::set_v)
    .def("get_v", &VectorTree::get_v)
    .def("weighted_sum", &VectorTree::weighted_sum)
    .def("set_dist_params", &VectorTree::set_dist_params)
    .def("kernel_matrix", &VectorTree::kernel_matrix)
    .def("kernel_deriv_wrt_i", &VectorTree::kernel_deriv_wrt_i)
    .def_readonly("fcalls", &VectorTree::fcalls);

  bp::class_<MatrixTree>("MatrixTree", bp::init< pyublas::numpy_matrix< double > const &, pyublas::numpy_vector< int > const &, pyublas::numpy_vector< int > const &, string const &, pyublas::numpy_vector< double > const &>())
    .def("set_m", &MatrixTree::set_m)
    .def("get_m", &MatrixTree::get_m)
    .def("quadratic_form", &MatrixTree::quadratic_form)
    .def("set_dist_params", &MatrixTree::set_dist_params)
    .def_readonly("fcalls", &MatrixTree::fcalls);

}
