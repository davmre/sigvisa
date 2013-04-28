#include "cover_tree.h"
#include "vector_mult.h"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>

#include <cmath>
#include <memory>
#include <vector>
using namespace std;
namespace bp = boost::python;


float VectorTree::weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, float eps, string wfn_str) {
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

  float weight_sofar = 0;
  int fcalls = 0;
  float ws = weighted_sum_node(this->root, v_select, 8,
			       qp, eps, weight_sofar,
			       fcalls, w, this->dfn);
  this->fcalls = fcalls;
  return ws;
}

void VectorTree::set_v(int v_select, const pyublas::numpy_vector<double> &v) {
  vector<float> new_v(v.size());
  for (pyublas::numpy_vector<double>::const_iterator a = v.begin(); a < v.end(); ++a) {
    new_v[a - v.begin()] = *a;
  }
  set_v_node(this->root, v_select, new_v);
}

VectorTree::VectorTree (const pyublas::numpy_matrix<double> &pts, const string &distfn_str) {
  vector< point > points(pts.size1(), point(pts.size2()));
  for (unsigned i = 0; i < pts.size1 (); ++ i)
    for (unsigned j = 0; j < pts.size2 (); ++ j)
      points[i][j] = pts (i, j);
  if (distfn_str.compare("pair") == 0) {
    this->dfn = pair_distance;
  } else{
    printf("error: unrecognized distance function %s\n", distfn_str.c_str());
    exit(1);
  }
  this->root = batch_create(points, this->dfn);
}


BOOST_PYTHON_MODULE(cover_tree) {
  bp::class_<VectorTree>("CoverTree", bp::init< pyublas::numpy_matrix< double > const &, string const &>())
    .def("set_v", &VectorTree::set_v)
    .def("weighted_sum", &VectorTree::weighted_sum)
    .def_readonly("fcalls", &VectorTree::fcalls);
}
