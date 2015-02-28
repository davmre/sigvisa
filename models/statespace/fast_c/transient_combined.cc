#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <cmath>
#include <memory>
#include <vector>
#include <limits>
#include <algorithm>
#include <utility>
#include <iterator>

#include "statespace.hpp"

using namespace boost::numeric::ublas;

class ChangeEvent {
  /* struct (supporting comparison) representing
     a change-of-active-SSMs event. */
public:
  int t;
  int i_ssm;
  bool is_start;
  friend bool operator< (const ChangeEvent &c1, const ChangeEvent &c2);
};

bool operator< (const ChangeEvent &c1, const ChangeEvent &c2) {
    if (c1.t < c2.t) {
      return true;
    } else if (c1.t == c2.t) {
      // ends come before starts
      return c1.is_start > c2.is_start;
    }
    return false;
  }



int active_set_dimension(const std::vector<StateSpaceModel *> & ssms, vector<int> & active_set) {
  vector<int>::const_iterator it;
  int dimension = 0;
  for (it = active_set.begin(); it < active_set.end(); ++it) {
    if (*it < 0) continue;
    StateSpaceModel *ssm = ssms[*it];
    if (!ssm) continue;
    dimension += ssm->max_dimension;
  }
  return dimension;
}


TransientCombinedSSM::TransientCombinedSSM(\
       std::vector<StateSpaceModel *> & ssms, const vector<int> & start_idxs,
       const vector<int> & end_idxs, const std::vector<const double * > & scales,
       double obs_noise) : ssms(ssms), start_idxs(start_idxs),
			   end_idxs(end_idxs), scales(scales),
			   obs_noise(obs_noise), n_ssms(ssms.size()),
			   active_ssm_cache1_k(-1), active_ssm_cache2_k(-1) {

  this->ssms_tmp.resize(this->n_ssms);
  this->n_steps = *(max_element(end_idxs.begin(), end_idxs.end()));

  /* Compute a list of ChangeEvents, each representing the
   * activation or deactivation of a component SSM. The
   * list is sorted by the timestep at which the event
   * occurs (and secondarily, with deactivation events at
   * a timestep before activation events). */
  std::vector<ChangeEvent> events(start_idxs.size() + end_idxs.size());
  for (unsigned i=0; i < n_ssms; ++i) {
    events[2*i].t = start_idxs[i];
    events[2*i].i_ssm = i;
    events[2*i].is_start = true;

    events[2*i+1].t = end_idxs[i];
    events[2*i+1].i_ssm = i;
    events[2*i+1].is_start = false;
  }
  std::sort(events.begin(), events.end());

  /*
     Build a sorted list of changepoints (timesteps at which the set of active
     SSMs changes), along with the set of active SSMs at each changepoint.
     Also compute the max total dimension of the state space, by computing the
     dimension of each active set as we add it to the active_sets matrix.
   */
  // allocate the active_sets matrix by first computing the largest number of
  // SSMs that might be active at one time
  unsigned int n_active = 0;
  unsigned int max_active = 0;
  for(std::vector<ChangeEvent>::const_iterator idx = events.begin(); idx < events.end(); ++idx) {
    if (idx->is_start) n_active++; else n_active--;
    if (n_active > max_active) max_active = n_active;
  }
  active_sets.resize(2*n_ssms, max_active);

  vector<int> active_set(max_active);
  for (unsigned i=0; i < max_active; ++i) active_set(i) = -1;
  int t_prev = 0, max_dimension = 0, i=0;
  std::vector<ChangeEvent>::const_iterator idx;
  for(i=0, idx = events.begin();
      idx < events.end(); ++idx) {

    // printf("event t %d i_ssm %d start %s\n", idx->t, idx->i_ssm, idx->is_start ? "true" : "false");

    // if this represents a change, add a new changepoint
    if (idx->t != t_prev) {
      changepoints.push_back(t_prev);

      // printf(" change detected (t_prev %d), copying to active_set %d\n", t_prev, i);

      // copy the current active set of SSMs
      // into the active_sets matrix (terminated with -1)
      for (unsigned k=0, j=0; k < max_active; ++k) {
	if (active_set(k) >= 0) {
	  active_sets(i, j++) = active_set(k);
	}
	if (j < max_active) {
	  active_sets(i, j) = -1;
	}
      }

      /*printf("matr active set(%d, ...) is", i);
      for(int tmpi=0; tmpi < max_active; ++tmpi) {
	printf(" %d", active_sets(i, tmpi));
      }
      printf("\n");*/


      t_prev = idx->t;
      i++;

      int current_dim = active_set_dimension(this->ssms, active_set);
      if (current_dim > max_dimension) max_dimension = current_dim;
    }

    // update the active set: add the new SSM if this
    // is a start event, otherwise remove it from
    // the active set.
    vector<int>::iterator it;
    if (idx->is_start) {
      it = std::find (active_set.begin(), active_set.end(), -1);
      if (it == active_set.end()) {
	printf("ERROR: need to start new SSM but no room in active_set vector!\n");
	exit(-1);
      } else {
	*it = idx->i_ssm;
      }
    } else {
      it = std::find (active_set.begin(), active_set.end(), idx->i_ssm);
      if (it == active_set.end()) {
	printf("ERROR: trying to remove SSM %d but it is not present in the active_set vector!\n", idx->i_ssm);
	exit(-1);
      } else {
	*it = -1;
      }
    }


  }

  this->max_dimension = max_dimension;

}

TransientCombinedSSM::~TransientCombinedSSM() {
  return;
};

int TransientCombinedSSM::active_set_idx(int k) {
  if (k == this->active_ssm_cache1_k)
    return this->active_ssm_cache1_v;
  else if (k == this->active_ssm_cache2_k)
    return this->active_ssm_cache2_v;

  std::vector<int>::iterator it = std::upper_bound(this->changepoints.begin(), this->changepoints.end(), k);
  int i = it - this->changepoints.begin() - 1;

  /*printf("active set at time k=%d is %d, changepoints %d %d %d %d\n", k, i,
	 i-1 >= 0 ? this->changepoints[i-1] : -1,
	 i >= 0 ? this->changepoints[i] : -1,
	 i+1 < this->changepoints.size() ? this->changepoints[i+1] : -1,
	 i+2 < this->changepoints.size() ? this->changepoints[i+2] : -1);*/

  this->active_ssm_cache2_k = this->active_ssm_cache1_k;
  this->active_ssm_cache2_v = this->active_ssm_cache1_v;
  this->active_ssm_cache1_k = k;
  this->active_ssm_cache1_v = i;
  return i;
}

int TransientCombinedSSM::apply_transition_matrix(const double * x, int k, double * result) {

  // first, loop over ssms active at the *previous*
  // timestep in order to cache the location of each
  // ssm in the previous state space.
  int j = 0;
  int asidx_prev = this->active_set_idx(k-1);
  int asidx = this->active_set_idx(k);
  bool same_active_set = (asidx==asidx_prev);

  if (k == 0) {
    printf("ERROR: requested transition INTO timestep 0 (invalid)!");
    exit(-1);
  }

  if (!same_active_set) {
    matrix_row < matrix<int> > old_ssm_indices = row(this->active_sets, asidx_prev);
    for (matrix_row < matrix<int> >::const_iterator it = old_ssm_indices.begin();
	 it < old_ssm_indices.end() && *it >= 0; ++it) {

      // skip any null SSMs
      int i_ssm = *it;
      StateSpaceModel * ssm = this->ssms[i_ssm];
      if (!ssm) continue;

      this->ssms_tmp[i_ssm] = j;
      j += ssm->max_dimension;
    }
  }

  //printf("transition to time %d (asidx %d, %d):\n",  k, asidx_prev, asidx);

  // now apply the transition to the current time
  int i=0;
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, asidx);
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    // skip any null SSMs
    int i_ssm = *it;
    StateSpaceModel * ssm = this->ssms[i_ssm];
    if (!ssm) continue;

    unsigned int state_size = ssm->max_dimension;
    if (this->start_idxs[i_ssm] == k) {
      /* new ssms just get filled in as zero
         (prior means will be added by the
         transition_bias operator) */
      for (unsigned j=i; j < i+state_size; ++j) {
	result[j] = 0;
      }
      //printf("   new ssm %d active from %d to %d\n", i_ssm, i, i+state_size);

    } else {
      /* this ssm is persisting from the
       * previous timestep, so just run the
       * transition */
      unsigned int j = same_active_set ? i : this->ssms_tmp[i_ssm];

      ssm->apply_transition_matrix(x+j, k-this->start_idxs[i_ssm], result+i);
      //printf("   transitioning ssm %d, prev %d, in state %d to %d (sidx %d eidx %d)\n", i_ssm, j, i, i+state_size, this->start_idxs[i_ssm], this->end_idxs[i_ssm]);
    }
    i += state_size;
  }
  return i;
}

int TransientCombinedSSM::apply_transition_matrix( const matrix<double,column_major> &X,
						   unsigned int x_row_offset,
						   int k,
						   matrix<double,column_major> &result,
						   unsigned int r_row_offset,
						   unsigned int n) {
  int j = x_row_offset;
  int asidx_prev = this->active_set_idx(k-1);
  int asidx = this->active_set_idx(k);
  bool same_active_set = (asidx==asidx_prev);

  if (!same_active_set) {
    matrix_row < matrix<int> > old_ssm_indices = row(this->active_sets, asidx_prev);
    for (matrix_row < matrix<int> >::const_iterator it = old_ssm_indices.begin();
	 it < old_ssm_indices.end() && *it >= 0; ++it) {

      // skip any null SSMs
      int i_ssm = *it;
      StateSpaceModel * ssm = this->ssms[i_ssm];
      if (!ssm) continue;

      this->ssms_tmp[i_ssm] = j;
      j += ssm->max_dimension;
    }
  }

  int i=x_row_offset;
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, asidx);
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    int i_ssm = *it;
    StateSpaceModel * ssm = this->ssms[i_ssm];
    if (!ssm) continue;

    unsigned int state_size = ssm->max_dimension;
    if (this->start_idxs[i_ssm] == k) {
      /* new ssms just get filled in as zero
         (prior means will be added by the
         transition_bias operator) */
      for (unsigned j=i; j < i+state_size; ++j) {
	for (unsigned jj=0; jj < n; ++jj) result(j, jj) = 0;
      }
    } else {
      unsigned int j = same_active_set ? i : this->ssms_tmp[i_ssm];
      //printf("MATR transitioning ssm %d, prev %d, in state %d to %d (sidx %d eidx %d)\n", i_ssm, j, i, i+state_size, this->start_idxs[i_ssm], this->end_idxs[i_ssm]);
      ssm->apply_transition_matrix(X, j, k-this->start_idxs[i_ssm], result, i, n);
    }
    i += state_size;
  }
  return i-x_row_offset;
}


void TransientCombinedSSM::transition_bias(int k, double *result) {

  int asidx = this->active_set_idx(k);
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, asidx);
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    // skip any null SSMs
    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    if (!ssm) continue;

    if (this->start_idxs[j] == k) {
      ssm->prior_mean(result);
    } else {
      ssm->transition_bias(k-this->start_idxs[j], result);
    }
    result += ssm->max_dimension;
  }
}


void TransientCombinedSSM::transition_noise_diag(int k, double *result) {
  int asidx = this->active_set_idx(k);
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, asidx);
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    // skip any null SSMs
    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    if (!ssm) continue;

    if (this->start_idxs[j] == k) {
      ssm->prior_vars(result);
    } else {
      ssm->transition_noise_diag(k-this->start_idxs[j], result);
    }
    result += ssm->max_dimension;
  }
}

double TransientCombinedSSM::apply_observation_matrix(const double *x, int k) {

  double r = 0;

  int asidx = this->active_set_idx(k);
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, asidx);
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    if (!ssm) continue;
    const double * scale = this->scales[j];
    double ri = ssm->apply_observation_matrix(x, k - this->start_idxs[j]);
    if (scale) ri *= scale[k - this->start_idxs[j]];
    r += ri;
    if (ssm) x += ssm->max_dimension;
  }
  return r;
}

void TransientCombinedSSM::apply_observation_matrix(const matrix<double,column_major> &X,
						    unsigned int row_offset, int k,
						    double *result, double *result_tmp, unsigned int n) {

  for (unsigned i=0; i < n; ++i) {
    result[i] = 0;
    result_tmp[i] = 0;
  }

  int asidx = this->active_set_idx(k);
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, asidx);
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {
    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    if (!ssm) continue;
    const double * scale = this->scales[j];

    unsigned int state_size = ssm->max_dimension;
    ssm->apply_observation_matrix(X, row_offset,
				  k-this->start_idxs[j],
				  result_tmp, NULL, n);
    //printf("TSSM step %d applying obs matrix on ssm %d at row_offset %d n %d\n", k, j, row_offset, n);
    if (scale) {
      for (unsigned ii=0; ii < n; ++ii) {
	result[ii] += scale[k-this->start_idxs[j]] * result_tmp[ii];
      }
    } else {
      for (unsigned ii=0; ii < n; ++ii) {
	result[ii] += result_tmp[ii];
      }
    }
    row_offset += state_size;
  }


}

double TransientCombinedSSM::observation_bias(int k) {
  double bias = 0;

  int asidx = this->active_set_idx(k);
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, asidx);
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    // skip any null SSMs
    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    const double * scale = this->scales[j];
    int kk = k - this->start_idxs[j];

    double b = ssm ? ssm->observation_bias(kk) : 1.0;
    if (scale) b *= scale[kk];
    bias += b;
  }
  return bias;
}

double TransientCombinedSSM::observation_noise(int k) {
  return this->obs_noise;
}

bool TransientCombinedSSM::stationary(int k) {

  matrix_row < matrix<int> > s1 = row(this->active_sets, this->active_set_idx(k));
  if (k > 0) {
    matrix_row < matrix<int> > s2 = row(this->active_sets, this->active_set_idx(k-1));
    for (unsigned i=0; i < s1.size() && !(s1(i) == -1 && s2(i) == -1); ++i) {
      if (s1(i) != s2(i)) return false;
    }
  }

  for (matrix_row < matrix<int> >::const_iterator it = s1.begin();
       it < s1.end() && *it >= 0; ++it) {
    // skip any null SSMs
    int j = *it;
    if (this->scales[j]) return false;

    StateSpaceModel * ssm = this->ssms[j];
    if (ssm && !ssm->stationary(k-this->start_idxs[j])) return false;
  }
  return true;
}

int TransientCombinedSSM::prior_mean(double *result) {

  /*
     TODO: propagate through the forward model whenever
   */

  double * r1 = result;
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, this->active_set_idx(0));
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    if (!ssm) continue;

    int state_size = ssm->max_dimension;
    for(int i=0; i < state_size; ++i) {
      result[i] = 0;
    }
    ssm->prior_mean(result);
    result += state_size;
  }
  return result-r1;
}

int TransientCombinedSSM::prior_vars(double *result) {
  double * r1 = result;
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, this->active_set_idx(0));
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    if (!ssm) continue;

    ssm->prior_vars(result);
    result += ssm->max_dimension;
  }
  return result-r1;
}

void TransientCombinedSSM::init_coef_priors(std::vector<vector<double> > & cmeans,
					    std::vector<vector<double> > & cvars) {
  for (unsigned j=0; j < this->n_ssms; ++j) {
    StateSpaceModel * ssm = this->ssms[j];
    if (ssm && ssm->is_cssm) {
      CompactSupportSSM *cssm = (CompactSupportSSM *) ssm;
      vector<double> mean(cssm->coef_means);
      vector<double> var(cssm->coef_vars);
      cmeans.push_back(mean);
      cvars.push_back(var);
    } else {
      vector<double> mean;
      vector<double> var;
      cmeans.push_back(mean);
      cvars.push_back(var);
    }

  }
}


void TransientCombinedSSM::extract_all_coefs(FilterState &cache, int k,
					     std::vector<vector<double> > & cmeans,
					     std::vector<vector<double> > & cvars) {
  /*
    Assumes cache has a valid, current P matrix.
   */
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, this->active_set_idx(k));
  unsigned int state_offset = 0;
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    if (!ssm) continue;
    if (ssm->is_cssm) {
      CompactSupportSSM *cssm = (CompactSupportSSM *) ssm;
      cssm->extract_coefs(cache.xk, cache.P,
			 state_offset, k - this->start_idxs[j],
			 cmeans[j],
			 cvars[j]);
    }
    state_offset += ssm->max_dimension;
  }
}

void TransientCombinedSSM::extract_component_means(double *xk, int k,
						   std::vector<vector<double> > & means) {
  matrix_row < matrix<int> > ssm_indices = row(this->active_sets, this->active_set_idx(k));
  unsigned int state_offset = 0;
  for (matrix_row < matrix<int> >::const_iterator it = ssm_indices.begin();
       it < ssm_indices.end() && *it >= 0; ++it) {

    int j = *it;
    StateSpaceModel * ssm = this->ssms[j];
    int kk = k - this->start_idxs[j];
    means[j](kk) = ssm ? ssm->apply_observation_matrix(xk + state_offset, kk) : 1.0;
    if (ssm) state_offset += ssm->max_dimension;
  }
}
