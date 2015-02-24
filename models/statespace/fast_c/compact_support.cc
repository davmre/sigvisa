#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <cmath>
#include <memory>
#include <vector>
#include <limits>
#include <algorithm>
#include <utility>
#include <iterator>
#include "statespace.hpp"

using namespace boost::numeric::ublas;

double gdb_get( matrix<double> & m, int i, int j) {
  return m(i,j);
}

CompactSupportSSM::CompactSupportSSM(\
       const vector<int> & start_idxs, const vector <int > & end_idxs,
       const vector<int> & identities, const matrix<double> & basis_prototypes,
       const vector<double> & coef_means,  const vector<double> & coef_vars,
       double obs_noise, double bias) : start_idxs(start_idxs), end_idxs(end_idxs),
					identities(identities), basis_prototypes(basis_prototypes),
					coef_means(coef_means), coef_vars(coef_vars),
					obs_noise(obs_noise), bias(bias) {
  this->n_basis = start_idxs.size();
  this->n_steps = *(std::max_element(end_idxs.begin(), end_idxs.end()));


  std::vector< std::vector<int> > tmp_active_basis(this->n_steps);

  this->active_indices.set_empty_key(std::make_pair(-1,-1) );

  int i;
  vector<int>::const_iterator st;
  vector<int>::const_iterator et;
  for( i = 0, st = this->start_idxs.begin(), et = this->end_idxs.begin() ;
       i < this->n_basis && st < this->start_idxs.end() && et < this->end_idxs.end();
       ++i, ++st, ++et ) {

    for (int j= std::max(*st, 0); j < std::min(*et, this->n_steps); ++j) {
      tmp_active_basis[j].push_back(i);
      (this->active_indices)[std::make_pair(i,j)] = int(tmp_active_basis[j].size()-1);
    }
  }

  int max_dimension = 0;
  for (int i=0; i < this->n_steps; ++i) {
    max_dimension = std::max(max_dimension, int(tmp_active_basis[i].size()));
  }
  this->max_dimension = max_dimension;

  this->active_basis = matrix<int>(this->n_steps, max_dimension);
  for (int i=0; i < this->n_steps; ++i) {
    int j = 0;
    for(j=0; j < int(tmp_active_basis[i].size()); ++j) {
      this->active_basis(i,j) = tmp_active_basis[i][j];
    }
    if (j < max_dimension) {
      this->active_basis(i,j) = -1;
    }
  }

  // printf("identities len %d\n", this->identities.size());


}

CompactSupportSSM::~CompactSupportSSM() {
  return;
};

int CompactSupportSSM::apply_transition_matrix(const vector<double> &x, int k, vector<double> &result){
  /* compute F_k*x for the given x, where
     F_k is the transition matrix from time
     k-1 to k. The result is stored in
     the provided vector x_new. If x_new
     has extra dimensions, they are not
     touched (so may still contain garbage).*/

  const matrix_row< matrix<int> > active = row(this->active_basis, k);
  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;
  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {



    // if this basis is new at this timestep, initialize to zero.
    // otherwise, use its previous value.
    std::pair<int,int> key = std::make_pair(int(*idx), k-1);
    dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  >::iterator contains = this->active_indices.find(key);
    if (contains == this->active_indices.end()) {
      result(i) = 0;
      // printf("transition k %d i %d idx %d prev_idx %d x_new[i] %.4f\n", k, i, *idx, -1, 0.0);
    } else {
      int prev_idx = this->active_indices[key];
      result(i) =x(prev_idx);
      // printf("transition k %d i %d idx %d prev_idx %d x_new[i] %.4f\n", k, i, *idx, prev_idx, x(prev_idx));
    }
  }
  return i;
}

void CompactSupportSSM::transition_bias(int k, vector<double> &result) {

  const matrix_row< matrix<int> > active = row(this->active_basis, k);
  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;
  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {

    // if this basis is new at this timestep, initialize to zero.
    // otherwise, use its previous value.
    std::pair<int,int> key = std::make_pair(*idx, k-1);
    dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  >::iterator contains = this->active_indices.find(key);
    if (contains == this->active_indices.end()) {
      result(i) += this->coef_means(*idx);
    }
  }
}


void CompactSupportSSM::transition_noise_diag(int k, vector<double> &result) {
  result.clear();

  const matrix_row< matrix<int> > active = row(this->active_basis, k);
  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;
  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {

    // if this basis is new at this timestep, initialize to zero.
    // otherwise, use its previous value.
    std::pair<int,int> key = std::make_pair(*idx, k-1);
    dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  >::iterator contains = this->active_indices.find(key);
    if (contains == this->active_indices.end()) {
      result(i) = this->coef_vars(*idx);
    } else {
      result(i) = 0.0;
    }
  }
}

double CompactSupportSSM::apply_observation_matrix(const vector<double> &x, int k) {
  // const matrix_row< matrix<int> > active = row(this->active_basis, k);

  double result = 0;

  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;
  for (i=0; i < this->active_basis.size2() && this->active_basis(k, i) >= 0; ++i) {
    int basis = this->active_basis(k, i);

    int prototype = (this->identities)(basis);
    int st = (this->start_idxs)(basis);
    int et = this->end_idxs(basis);

    result += x(i) * (this->basis_prototypes)(prototype, k-st);
  }
  return result;
}

void CompactSupportSSM::apply_observation_matrix(const matrix<double> &X, int k, vector<double> &result, int n) {
  //const matrix_row< matrix<int> > active = row(this->active_basis, k);
  //matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;
  result.clear();
  for (i=0; i < this->active_basis.size2() && this->active_basis(k, i) >= 0; ++i) {
    int basis = this->active_basis(k, i);

    int prototype = this->identities(basis);
    int st = this->start_idxs(basis);
    int et = this->end_idxs(basis);

    for (int j=0; j < n; ++j) {
      result(j) += X(i,j) * this->basis_prototypes(prototype, k-st);
    }
  }
}


double CompactSupportSSM::observation_bias(int k) {
  return this->bias;
}

double CompactSupportSSM::observation_noise(int k) {
  return this->obs_noise;
}

bool CompactSupportSSM::stationary(int k) {
  return false;
}

int CompactSupportSSM::prior_mean(vector<double> &result) {
  const matrix_row< matrix<int> > active = row(this->active_basis, 0);
  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;

  int d = std::distance(active.begin(), active.end());

  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {
    int d1 = std::distance(idx, active.end());
    // printf("prior mean i %d *idx %d coef_mean %f distance %d %d\n", i, *idx, this->coef_means(*idx), d, d1);
    result(i) = this->coef_means(*idx);
  }
  return i;
}

int CompactSupportSSM::prior_vars(vector<double> &result) {
  const matrix_row< matrix<int> > active = row(this->active_basis, 0);
  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;
  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {
    result(i) = this->coef_vars(*idx);
  }
  return i;
}

/*
    def extract_coefs(self, x, P, k, coef_means, coef_vars):
        # given a state estimate at some time k, extract marginals for
        # whatever coefficients we can reasonably do so for at the
        # current time. Earlier estimates will always be overwritten
        # by later ones.
        for i, basis_idx in enumerate(self.active_basis[k]):
            state_idx = self.active_indices[basis_idx, k]
            coef_means[basis_idx] = x[state_idx-1]
            coef_vars[basis_idx] = P[state_idx-1,state_idx-1]

    def filtered_coef_marginals(self, z):
        coef_means = np.empty((self.n_basis,))
        coef_vars = np.empty((self.n_basis,))


        for k, (x, U, d) in enumerate(self.filtered_states(z)):
            P = np.dot(d*U, U.T)
            self.extract_coefs(x, P, k, coef_means, coef_vars)
            if k % 100 == 0:
                print "filtering step", k

        return coef_means, coef_vars
*/
