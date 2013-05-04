#include "cover_tree.hpp"
#include "vector_mult.hpp"

#include <cmath>
#include <memory>
#include <vector>
#include <limits.h>
#include <values.h>

using namespace std;
namespace bp = boost::python;

void set_m_node (node<pairpoint> &n, const pyublas::numpy_matrix<double> &m) {
  if (n.num_children == 0) {
    n.unweighted_sum = m(n.p.idx1, n.p.idx2);
  } else {
    n.unweighted_sum = 0;
    for(int i=0; i < n.num_children; ++i) {
      set_m_node(n.children[i], m);
      n.unweighted_sum += n.children[i].unweighted_sum;
    }
  }
}

void get_m_node(node<pairpoint> &n, pyublas::numpy_matrix<double> &m) {
  if (n.num_children == 0) {
    //printf("%du %du = %lf\n" n.p.idx1, n.p.idx2, n.unweighted_sum);
    m(n.p.idx1, n.p.idx2) = n.unweighted_sum;
  } else {
    for(int i=0; i < n.num_children; ++i) {
      get_m_node(n.children[i], m);
    }
  }
}


double MatrixTree::quadratic_form(const pyublas::numpy_matrix<double> &query_pt, double eps, string wfn_str, const pyublas::numpy_vector<double> &weight_params) {
  pairpoint qp = double_point(&query_pt(0,0));
  wfn w;
  if (wfn_str.compare("se") == 0) {
    w = w_se;
  } else{
    printf("error: unrecognized weight function %s\n", wfn_str.c_str());
    exit(1);
  }

  /* GIANT HACK: we're assuming the weight function is SE,
     so w(d1)*w(d2) = w'(sqrt(d1**2 + d2**2))
     where w' is w with a squared leading coefficient.
   */
  double * wp = NULL;
  if (weight_params.size() > 0) {
    wp = new double[weight_params.size()];
    for (unsigned i = 0; i < weight_params.size(); ++i) {
      wp[i] = weight_params(i)*weight_params(i);
    }
  }

  double weight_sofar = 0;
  int fcalls = 0;
  this->root.distance_to_query = this->dfn(qp, this->root.p, MAXDOUBLE, this->dist_params);
  double ws = weighted_sum_node(this->root, 0,
				qp, eps, weight_sofar,
				fcalls, w, this->dfn, this->dist_params, wp);
  if (wp != NULL) {
    delete wp;
    wp = NULL;
  }
  this->fcalls = fcalls;
  return ws;
}

void MatrixTree::set_m(const pyublas::numpy_matrix<double> &m) {
  if (m.size1() != m.size2()) {
    printf("error: matrixtree can only hold square matrices! (matrix passed has dimensions %lu x %lu)\n", m.size1(), m.size2());
    exit(1);
  }
  set_m_node(this->root, m);
}

pyublas::numpy_matrix<double> MatrixTree::get_m() {
  vector<double> v(this->n * this->n);
  pyublas::numpy_matrix<double> pm(this->n, this->n);
  get_m_node(this->root, pm);
  return pm;
}

MatrixTree::MatrixTree (const pyublas::numpy_matrix<double> &pts,
			const pyublas::numpy_strided_vector<int> &nonzero_rows,
			const pyublas::numpy_strided_vector<int> &nonzero_cols,
			const string &distfn_str,
			const pyublas::numpy_vector<double> &dist_params) {


  unsigned int nzero = nonzero_rows.size();
  vector< pairpoint > pairs(nzero);
  unsigned int n = pts.size1();
  unsigned int i=0;
  for(unsigned int i=0; i < nzero; ++i) {
    int r = nonzero_rows(i);
    int c = nonzero_cols(i);
    pairs[i].pt1 = &pts(r, 0);
    pairs[i].pt2 = &pts(c, 0);
    pairs[i].idx1 = r;
    pairs[i].idx2 = c;
  }
  this->n = n;

if (distfn_str.compare("lld") == 0) {
    this->dfn = pair_dist_3d_km;
  } else{
    printf("error: unrecognized distance function %s\n", distfn_str.c_str());
    exit(1);
  }

  this->dist_params = NULL;
  this->set_dist_params(dist_params);
  this->root = batch_create(pairs, this->dfn, this->dist_params);
  this->root.alloc_arms(1);
}

void MatrixTree::set_dist_params(const pyublas::numpy_vector<double> &dist_params) {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
  this->dist_params = new double[dist_params.size()];
  for (unsigned i = 0; i < dist_params.size(); ++i) {
    this->dist_params[i] = dist_params(i);
  }
}

MatrixTree::~MatrixTree() {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
}
