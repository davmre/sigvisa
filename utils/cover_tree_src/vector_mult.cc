#include "cover_tree.h"
#include "vector_mult.h"
#include <limits.h>
#include <values.h>
#include <stdint.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
using namespace std;

float pair_distance(const point &pt1, const point &pt2, float BOUND_IGNORED, const double * PARAMS_IGNORED) {
  float a = sqrt(pow(pt2[1] - pt1[1], 2) + pow(pt2[0] - pt1[0], 2));
  return a;
}

float w_se(float d, const double * variance) {
  return variance[0] * exp(-1 * pow(d,2));
}


void set_v_node(node &n, int v_select, const vector<float> &v) {
  if (n.num_children == 0) {
      n.unweighted_sum_v[v_select] = v[n.point_idx];
  } else {
    n.unweighted_sum_v[v_select] = 0;
    for(int i=0; i < n.num_children; ++i) {
      set_v_node(n.children[i], v_select, v);
      n.unweighted_sum_v[v_select] += n.children[i].unweighted_sum_v[v_select];
    }
  }
}

typedef pair<float,int> mypair;
float weighted_sum_node(node &n, int v_select, int pt_len,
			const point &query_pt, float eps,
			float &weight_sofar,
			int &fcalls,
			wfn w,
			distfn dist,
			const double * dist_params,
			const double* weight_params) {
  float ws = 0;
  float d = n.distance_to_query; // avoid duplicate distance
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

    float weight = w(d, weight_params);
    ws = weight * n.unweighted_sum_v[v_select];
    weight_sofar += weight;
    cutoff = true;
  } else {
    bool query_in_bounds = (d <= n.max_dist);
    if (!query_in_bounds) {
      float min_weight = w(d + n.max_dist, weight_params);
      float max_weight = w(max(0.0f, d - n.max_dist), weight_params);
      float cutoff_threshold = 2 * eps * (weight_sofar + n.num_children * min_weight);
      cutoff = (max_weight - min_weight) <= cutoff_threshold;
      if (cutoff) {
	// if we're cutting off, just compute an estimate of the sum
	// in this region
	ws = .5 * (max_weight + min_weight) * n.unweighted_sum_v[v_select];
	weight_sofar += min_weight * n.num_children;
      }
    }
    if (!cutoff) {
      // if not cutting off, we expand the sum recursively at the
      // children of this node, from nearest to furthest.
      vector<mypair> v(n.num_children);
      for(int i=0; i < n.num_children; ++i) {
	n.children[i].distance_to_query = dist(query_pt, n.children[i].p, MAXFLOAT, dist_params);
	v[i] = mypair(n.children[i].distance_to_query, i);
      }
      sort(v.begin(), v.end());
      for(int i=0; i < n.num_children; ++i) {
	int child_i = v[i].second;
	ws +=weighted_sum_node(n.children[child_i], v_select, pt_len,
			  query_pt, eps, weight_sofar, fcalls,
			       w, dist, dist_params, weight_params);
      }
    }
  }
  n.ws[v_select] = ws;
  n.cutoff[v_select] = cutoff;
  return ws;
}
