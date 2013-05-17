#include "cover_tree.hpp"
#include <limits.h>
#include <pthread.h>
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


#include <google/dense_hash_map>





typedef std::pair<double,int> mypair;
template<class T>
double weighted_sum_node(node<T> &n, int v_select,
			 const T &query_pt, double eps,
			 double &weight_sofar,
			 int &fcalls,
			 wfn w,
			 typename distfn<T>::Type dist,
			 const double * dist_params,
			 void * dist_extra,
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
      for(int i=0; i < n.num_children; ++i) {
	n.children[i].distance_to_query = dist(query_pt, n.children[i].p, MAXDOUBLE, dist_params, dist_extra);
      }
      int permutation[20];
      if (n.num_children > 20){ printf("error: too many (%d) children!\n", n.num_children); exit(1); }
      bubblesort_nodes(n.children, n.num_children, permutation);
      for(int i=0; i < n.num_children; ++i) {
	ws +=weighted_sum_node(n.children[permutation[i]], v_select,
			  query_pt, eps, weight_sofar, fcalls,
			       w, dist, dist_params, dist_extra, weight_params);
      }
    }
  }
  return ws;
}

template <class T>
void bubblesort_nodes(node<T> * children, int num_children, int (&permutation)[20]) {
  /*printf("bubble sorting... ");
  for(int i=0; i < num_children; ++i) {
    printf(" %.3f ", children[i].distance_to_query);
  }
  printf("\n");*/

  for(int i=0; i < num_children; ++i) {
    permutation[i] = i;
  }

  while(1) {
    int i;
    for(i=1; i < num_children; ++i) {
      int current = permutation[i];
      int prev = permutation[i-1];
      if (children[current].distance_to_query < children[prev].distance_to_query) {
	permutation[i] = prev;
	permutation[i-1] = current;
	break;
      }
    }
    if (i == num_children) break;
  }

  /*
  printf("done... ");
  for(int i=0; i < num_children; ++i) {
    printf(" %.3f ", children[permutation[i]].distance_to_query);
  }
  printf("\n");
  sleep(1);*/

}

class VectorTree {
  node<point> root;
  distfn<point>::Type dfn;
  void * dfn_extra;

public:
  unsigned int n;
  int fcalls;
  VectorTree (const pyublas::numpy_matrix<double> &pts, const unsigned int narms, const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params);
  void set_v(int v_select, const pyublas::numpy_vector<double> &v);
  pyublas::numpy_vector<double> get_v(int v_select);

  double weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps,
		     std::string wfn_str, const pyublas::numpy_vector<double> &weight_params);


  pyublas::numpy_matrix<double> kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, std::string wfn_str, const pyublas::numpy_vector<double> &weight_params, bool distance_only);
  pyublas::numpy_matrix<double> kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, std::string wfn_str, const pyublas::numpy_vector<double> &weight_params, int param_i);


  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);
  double *dist_params;

  ~VectorTree();
};



struct pair_dfn_extra {
  partial_dfn dfn;
  google::dense_hash_map<long, double> *cache;
  void * dfn_extra;
  int NPTS;
  int hits;
  int misses;
};

class MatrixTree {
  node<pairpoint> root;
  distfn<pairpoint>::Type dfn;
  pair_dfn_extra * dfn_extra;
  double * distance_cache;

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
