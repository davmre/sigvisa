#include "cover_tree.hpp"
#include "vector_mult.hpp"
#include <cstdlib>
#include <cmath>
#include <memory>
#include <vector>
#include <limits.h>
#include <values.h>
#include <google/dense_hash_map>
using google::dense_hash_map;

using namespace std;
namespace bp = boost::python;

double gt(void)
{
  struct timespec tv;

  if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

  return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}

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

double factored_distance(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *params, void * extra) {

  pair_dfn_extra * p = (pair_dfn_extra *) extra;
  dense_hash_map<long, double> &distance_cache = *(p->cache);

  const int NPTS = (const int) p->NPTS;

  double d1;
  long pair1_idx = p1.idx1 * NPTS + p2.idx1;
  dense_hash_map<long, double>::iterator i = distance_cache.find(pair1_idx);
  if (i == distance_cache.end()) {
    d1 = p->dfn(p1.pt1, p2.pt1, BOUND_IGNORED, params, p->dfn_extra);
    distance_cache[pair1_idx] = d1;
    p->misses += 1;
  } else {
    d1 = distance_cache[pair1_idx];
    p->hits += 1;
  }

  double d2;
  long pair2_idx = p1.idx2 * NPTS + p2.idx2;
  i = distance_cache.find(pair2_idx);
  if (i == distance_cache.end()) {
    d2 = p->dfn(p1.pt2, p2.pt2, BOUND_IGNORED, params, p->dfn_extra);
    distance_cache[pair2_idx] = d2;
    p->misses += 1;
  } else {
    d2 = distance_cache[pair2_idx];
    p->hits += 1;
  }

  return sqrt(d1 + d2);
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

  this->distance_cache[0] = 0;
  for(unsigned int i=1; i < this->n+1; ++i) {
    this->distance_cache[i] = -1;
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
  this->root.distance_to_query = this->dfn(qp, this->root.p, MAXDOUBLE, this->dist_params, (void*)this->distance_cache);
  double ws = weighted_sum_node(this->root, 0,
				qp, eps, weight_sofar,
				fcalls, w, this->dfn, this->dist_params, (void *)this->dfn_extra, wp);
  this->fcalls = fcalls;
  //printf("quadratic form did %.0lf distance calculations for %d fcalls\n", ((double *)(this->distance_cache))[0], this->fcalls);

  if (wp != NULL) {
    delete wp;
    wp = NULL;
  }

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

  pair_dfn_extra * p = new pair_dfn_extra;

  p->dfn_extra = NULL;
  if (distfn_str.compare("lld") == 0) {
    p->dfn = distsq_3d_km;
  } else if (distfn_str.compare("euclidean") == 0) {
    p->dfn = sqdist_euclidean;
    p->dfn_extra = malloc(sizeof(int));
    *((int *) p->dfn_extra) = pts.size2();
  } else{
    printf("error: unrecognized distance function %s\n", distfn_str.c_str());
    exit(1);
  }
  p->cache = new dense_hash_map<long, double>(nzero*pts.size2());
  p->cache->set_empty_key(-1);
  p->hits = 0;
  p->misses = 0;
  p->NPTS = n;

  this->dfn_extra = p;

  this->dist_params = NULL;
  this->set_dist_params(dist_params);

  // next block contains some benchmarking code, not needed
  /*
  double a;
 int k = 0;
 double tt0 = gt();
 for(unsigned int i=0; i < nzero; i += 93) {
   for(unsigned int j=1; j < nzero; j += 107) {
     a = factored_build_distance(pairs[i], pairs[j], 0, this->dist_params, (void *) build_cache);
     ++k;
   }
 }
 double tt1 = gt();
 printf("did %d hash distance computations in %lfs: %d cache hits and %d cache misses\n", k, tt1-tt0, (int)((*build_cache)[-3]), (int)((*build_cache)[-2]));
 (*build_cache)[-2] = 0;
 (*build_cache)[-3] = 0;
 k=0;
 for(unsigned int i=0; i < nzero; i += 93) {
   for(unsigned int j=1; j < nzero; j += 107) {
     a = factored_build_distance(pairs[i], pairs[j], 0, this->dist_params, (void *) build_cache);
     ++k;
   }
 }
 double tt2 = gt();
 printf("did the same %d hash distance computations in %lfs: %d cache hits and %d cache misses\n", k, tt2-tt1, (int)((*build_cache)[-3]), (int)((*build_cache)[-2]));
 k = 0;
 for(unsigned int i=0; i < nzero; i += 93) {
   for(unsigned int j=1; j < nzero; j += 107) {
     a = this->dfn(pairs[i], pairs[j], 0, this->dist_params, NULL);
     ++k;
   }
 }
 double tt3 = gt();
 printf("did  %d naive distance computations in %lfs.\n", k, tt3-tt2);
 exit(0);*/

  double t0 = gt();
  this->root = batch_create(pairs, factored_distance, this->dist_params, this->dfn_extra);
  double t1 = gt();
  printf("built tree in %lfs: %d cache hits and %d cache misses\n", t1-t0, p->hits, p->misses);

  this->root.alloc_arms(1);
  delete p->cache;
}

void MatrixTree::set_dist_params(const pyublas::numpy_vector<double> &dist_params) {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
  this->dist_params = new double[dist_params.size()+1];
  for (unsigned i = 0; i < dist_params.size(); ++i) {
    this->dist_params[i] = dist_params(i);
  }
  // HACK ALERT
  this->dist_params[dist_params.size()] = (double) this->n;
}

MatrixTree::~MatrixTree() {
  if (this->dist_params != NULL) {
    delete this->dist_params;
    this->dist_params = NULL;
  }
  delete this->dfn_extra;
}
