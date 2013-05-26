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
#include <boost/numeric/ublas/matrix_sparse.hpp>


#include <google/dense_hash_map>






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
  wfn w;
  double * wp;
  double *dist_params;
  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);


public:
  unsigned int n;
  int fcalls;
  VectorTree (const pyublas::numpy_matrix<double> &pts, const unsigned int narms,
	      const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params,
	      const std::string wfn_str,
	      const pyublas::numpy_vector<double> &weight_params);
  void set_v(int v_select, const pyublas::numpy_vector<double> &v);
  pyublas::numpy_vector<double> get_v(int v_select);

  double weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps);


  pyublas::numpy_matrix<double> kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, bool distance_only);
  pyublas::numpy_matrix<double> sparse_training_kernel_matrix(const pyublas::numpy_matrix<double> &pts, double max_distance);
  pyublas::numpy_matrix<double> kernel_deriv_wrt_i(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, int param_i);

  void dump_tree(const std::string &fname);


  ~VectorTree();
};



struct pair_dfn_extra {
  partial_dfn dfn;
  partial_dfn dfn_orig;
  partial_dfn dfn_sq;
  google::dense_hash_map<long, double> *build_cache;
  google::dense_hash_map<int, double> *query1_cache;
  google::dense_hash_map<int, double> *query2_cache;
  void * dfn_extra;
  int NPTS;
  int hits;
  int misses;
};

class MatrixTree {
  node<pairpoint> root;
  distfn<pairpoint>::Type raw_pair_dfn;
  distfn<pairpoint>::Type factored_build_dist;
  distfn<pairpoint>::Type factored_query_dist;
  pair_dfn_extra * dfn_extra;
  double * wp_pair;
  double * wp_point;
  wfn w_point;
  wfn w_upper;
  wfn w_lower;
  double max_weight;

  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);
  double *dist_params;


public:
  unsigned int n;
  int fcalls;
  int dfn_evals;
  MatrixTree (const pyublas::numpy_matrix<double> &pts,
	      const pyublas::numpy_strided_vector<int> &nonzero_rows,
	      const pyublas::numpy_strided_vector<int> &nonzero_cols,
	      const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params,
	      std::string wfn_str, const pyublas::numpy_vector<double> &weight_params);
  void set_m(const pyublas::numpy_matrix<double> &m);
  void set_m_sparse(const pyublas::numpy_strided_vector<int> &nonzero_rows,
		    const pyublas::numpy_strided_vector<int> &nonzero_cols,
		    const pyublas::numpy_strided_vector<double> &nonzero_vals);

  pyublas::numpy_matrix<double> get_m();

  double quadratic_form(const pyublas::numpy_matrix<double> &query_pt1,
			const pyublas::numpy_matrix<double> &query_pt2,
			double eps);

  void print_hierarchy(const pyublas::numpy_matrix<double> &query_pt1, const pyublas::numpy_matrix<double> &query_pt2);

  ~MatrixTree();
};
