#include "cover_tree.hpp"
#include <limits.h>
#include <values.h>
#include <stdint.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>







typedef std::pair<double,int> mypair;
template<class T>
double weighted_sum_node(node<T> &n, int v_select,
			 const T &query_pt, double eps,
			 double &weight_sofar,
			 int &fcalls,
			 wfn w,
			 typename distfn<T>::Type dist,
			 const double * dist_params,
			 const double* weight_params) {
  double ws = 0;
  double d = n.distance_to_query; // avoid duplicate distance
				    // calculations by assuming this
				    // distance has already been
				    // computed by the parent, in the
				    // recursive expansion below. Note
				    // this calculation must be done
				    // explicitly at the root before
				    // this function is called.
  fcalls += 1;
  bool cutoff = false;
  if (n.num_children == 0) {
    // if we're at a leaf, just do the multiplication

    double weight = w(d, weight_params);
    ws = weight * n.unweighted_sums[v_select];
    weight_sofar += weight;
    //printf("at leaf: ws = %lf*%lf = %lf\n", weight, n.unweighted_sums[v_select], ws);
    cutoff = true;
  } else {
    bool query_in_bounds = (d <= n.max_dist);
    if (!query_in_bounds) {
      double min_weight = w(d + n.max_dist, weight_params);
      double max_weight = w(max(0.0, d - n.max_dist), weight_params);
      double cutoff_threshold = 2 * eps * (weight_sofar + n.num_children * min_weight);
      cutoff = (max_weight - min_weight) <= cutoff_threshold;
      if (cutoff) {
	// if we're cutting off, just compute an estimate of the sum
	// in this region
	ws = .5 * (max_weight + min_weight) * n.unweighted_sums[v_select];
	//printf("cutting off: ws = %lf*%lf = %lf\n", .5 * (max_weight + min_weight), n.unweighted_sums[v_select], ws);
	weight_sofar += min_weight * n.num_children;
      }
    }
    if (!cutoff) {
      // if not cutting off, we expand the sum recursively at the
      // children of this node, from nearest to furthest.
      std::vector<mypair> v(n.num_children);
      for(int i=0; i < n.num_children; ++i) {
	n.children[i].distance_to_query = dist(query_pt, n.children[i].p, MAXDOUBLE, dist_params);
	v[i] = mypair(n.children[i].distance_to_query, i);
      }
      sort(v.begin(), v.end());
      for(int i=0; i < n.num_children; ++i) {
	int child_i = v[i].second;
	ws +=weighted_sum_node(n.children[child_i], v_select,
			  query_pt, eps, weight_sofar, fcalls,
			       w, dist, dist_params, weight_params);
      }
    }
  }
  return ws;
}



class VectorTree {
  node<point> root;
  distfn<point>::Type dfn;

public:
  unsigned int n;
  int fcalls;
  VectorTree (const pyublas::numpy_matrix<double> &pts, const unsigned int narms, const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params);
  void set_v(int v_select, const pyublas::numpy_vector<double> &v);
  pyublas::numpy_vector<double> get_v(int v_select);

  double weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps,
		     std::string wfn_str, const pyublas::numpy_vector<double> &weight_params);


  pyublas::numpy_matrix<double> debug_kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, std::string wfn_str, const pyublas::numpy_vector<double> &weight_params, bool distance_only);

  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);
  double *dist_params;

  ~VectorTree();
};

class MatrixTree {
  node<pairpoint> root;
  distfn<pairpoint>::Type dfn;

public:
  unsigned int n;
  int fcalls;
  MatrixTree (const pyublas::numpy_matrix<double> &pts,
	      const pyublas::numpy_strided_vector<int> &nonzero_rows,
	      const pyublas::numpy_strided_vector<int> &nonzero_cols,
	      const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params);
  void set_m(const pyublas::numpy_matrix<double> &m);
  pyublas::numpy_matrix<double> get_m();

  double quadratic_form(const pyublas::numpy_matrix<double> &query_pt, double eps,
		     std::string wfn_str, const pyublas::numpy_vector<double> &weight_params);


  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);
  double *dist_params;

  ~MatrixTree();
};
