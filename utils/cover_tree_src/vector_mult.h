#include "cover_tree.h"
#include <pyublas/numpy.hpp>

typedef float (*wfn)(float);
class VectorTree {
  node root;
  distfn dfn;

public:
  int fcalls;
  VectorTree (const pyublas::numpy_matrix<double> &pts, const std::string &distfn_str);
  void set_v(int v_select, const pyublas::numpy_vector<double> &v);
  float weighted_sum(int v_select, const pyublas::numpy_matrix<double> &query_pt, float eps,
		     std::string wfn_str);


};


float pair_distance(const point &pt1, const point &pt2, float BOUND_IGNORED);
float w_se(float d);
void set_v_node(node &n, int v_select, const std::vector<float> &v);
float weighted_sum_node(node &n, int v_select, int pt_len,
			const point &query_pt, float eps,
			float &weight_sofar,
			int &fcalls,
			wfn w,
			distfn dist);
