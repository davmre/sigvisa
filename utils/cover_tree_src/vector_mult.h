#include "cover_tree.h"
#include <pyublas/numpy.hpp>

typedef double (*wfn)(double, const double *);
class CoverTree {
  node root;
  distfn dfn;

public:
  unsigned int n;
  int fcalls;
  CoverTree (const pyublas::numpy_matrix<double> &pts, const std::string &distfn_str, const pyublas::numpy_vector<double> &dist_params);
  void set_v(int v_select, const pyublas::numpy_vector<double> &v);
  pyublas::numpy_vector<double> get_v(int v_select);

  double weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, double eps,
		     std::string wfn_str, const pyublas::numpy_vector<double> &weight_params);


  pyublas::numpy_matrix<double> debug_kernel_matrix(const pyublas::numpy_matrix<double> &pts1, const pyublas::numpy_matrix<double> &pts2, std::string wfn_str, const pyublas::numpy_vector<double> &weight_params, bool distance_only);

  void set_dist_params(const pyublas::numpy_vector<double> &dist_params);
  double *dist_params;

  ~CoverTree();
};


double pair_distance(const point &pt1, const point &pt2, double BOUND_IGNORED, const double * PARAMS_IGNORED);
double w_se(double d, const double * PARAMS_IGNORED);
void set_v_node(node &n, int v_select, const std::vector<double> &v);
void get_v_node(node &n, int v_select, std::vector<double> &v);
double weighted_sum_node(node &n, int v_select, int pt_len,
			const point &query_pt, double eps,
			double &weight_sofar,
			int &fcalls,
			wfn w,
			distfn dist,
			const double * dist_params,
			const double * weight_params);
