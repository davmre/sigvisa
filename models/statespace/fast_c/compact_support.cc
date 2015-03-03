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


CompactSupportSSM::CompactSupportSSM(\
       const vector<int> & start_idxs, const vector <int > & end_idxs,
       const vector<int> & identities, const matrix<double> & basis_prototypes,
       const vector<double> & coef_means,  const vector<double> & coef_vars,
       double obs_noise, double bias) : start_idxs(start_idxs), end_idxs(end_idxs),
					identities(identities), basis_prototypes(basis_prototypes),
					coef_means(coef_means), coef_vars(coef_vars),
					obs_noise(obs_noise), bias(bias) {

  this->is_cssm = true;
  this->n_basis = start_idxs.size();
  this->n_steps = *(std::max_element(end_idxs.begin(), end_idxs.end()));

  if (coef_means.size() != this->n_basis) {
    printf("ERROR: coef prior mean length must match size of basis\n");
    exit(-1);
  }
  if (coef_vars.size() != this->n_basis) {
    printf("ERROR: coef prior vars length must match size of basis\n");
    exit(-1);
  }

  std::vector< std::vector<int> > tmp_active_basis(this->n_steps);

  this->active_indices.set_empty_key(std::make_pair(-1,-1) );

  unsigned int i;
  vector<int>::const_iterator st;
  vector<int>::const_iterator et;
  for( i = 0, st = this->start_idxs.begin(), et = this->end_idxs.begin() ;
       i < this->n_basis && st < this->start_idxs.end() && et < this->end_idxs.end();
       ++i, ++st, ++et ) {

    for (unsigned j= std::max(*st, 0); j < std::min((unsigned int)*et, this->n_steps); ++j) {
      tmp_active_basis[j].push_back(i);
      (this->active_indices)[std::make_pair(i,j)] = int(tmp_active_basis[j].size()-1);
    }
  }

  int max_dimension = 0;
  for (unsigned i=0; i < this->n_steps; ++i) {
    max_dimension = std::max(max_dimension, int(tmp_active_basis[i].size()));
  }
  this->max_dimension = max_dimension;

  this->active_basis = matrix<int>(this->n_steps, max_dimension);
  for (unsigned i=0; i < this->n_steps; ++i) {
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


int CompactSupportSSM::apply_transition_matrix( const double * x, int k, double * result) {
  /* compute F_k*x for the given x, where
     F_k is the transition matrix from time
     k-1 to k. The result is stored in
     the provided vector x_new. If x_new
     has extra dimensions, they are not
     touched (so may still contain garbage).*/

  if (k <= 0 || k >= this->n_steps) {
    printf("error: applying CSSM transition at invalid index %d.\n", k);
    exit(-1);
  }

  for (int i=0; i < this->max_dimension; ++i) {
    result[i] = 0;
  }

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
      result[i] = 0;
      // printf("transition k %d basis %d idx %d prev_idx %d x_new[i] %.4f\n", k, *idx, i, -1, 0.0);
    } else {
      int prev_idx = this->active_indices[key];
      result[i] =x[prev_idx];
      // printf("transition k %d basis %d idx %d prev_idx %d x_new[i] %.4f\n", k, *idx, i, prev_idx, x[prev_idx]);
    }
  }
  return i;
}

/* Each column of the input X gives a column of the result matrix */
int CompactSupportSSM::apply_transition_matrix( const matrix<double,column_major> &X,
						unsigned int x_row_offset,
						int k,
						matrix<double,column_major> &result,
						unsigned int r_row_offset,
						unsigned int n) {
  if (k <= 0 || k >= this->n_steps) {
    printf("error: applying CSSM transition at invalid index %d.\n", k);
    exit(-1);
  }


  for (int i=r_row_offset; i < r_row_offset+this->max_dimension; ++i) {
    for (int j =0; j < n; ++j) {
      result(i,j) = 0;
    }
  }

  const matrix_row< matrix<int> > active = row(this->active_basis, k);
  matrix_row< matrix<int> >::const_iterator idx;
  unsigned i = r_row_offset;

  int dummy= 0 ;
  if (k == 4) {
    dummy +=1 ;
  }

  for (i=r_row_offset, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {

    // if this basis is new at this timestep, initialize to zero.
    // otherwise, use its previous value.
    std::pair<int,int> key = std::make_pair(int(*idx), k-1);
    dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  >::iterator contains = this->active_indices.find(key);

    if (contains == this->active_indices.end()) {
      for (unsigned j=0; j < n; ++j) result(i, j) = 0;
      // printf("MATR0 transition k %d i %d idx %d prev_idx %d x_new[i,i] %.4f\n", k, i, *idx, -1, 0.0);
    } else {
      int prev_idx = this->active_indices[key] + x_row_offset;
      for (unsigned j=0; j < n; ++j) result(i, j) = X(prev_idx, j);
      //printf("MATR1 transition k %d i %d idx %d prev_idx %d x_new[i, i] %.4f\n", k, i, *idx, prev_idx, X(prev_idx, prev_idx));
    }
  }
  return i-r_row_offset;
}


void CompactSupportSSM::transition_bias(int k, double * result) {

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
      result[i] += this->coef_means(*idx);
    }
  }
}


void CompactSupportSSM::transition_noise_diag(int k, double * result) {
  //result.clear();

  unsigned int i = 0;
  for (i; i < this->max_dimension; ++i) {
    result[i] = 0.0;
  }


  const matrix_row< matrix<int> > active = row(this->active_basis, k);
  matrix_row< matrix<int> >::const_iterator idx;
  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {
    std::pair<int,int> key = std::make_pair(*idx, k-1);
    dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  >::iterator contains = this->active_indices.find(key);
    if (contains == this->active_indices.end()) {
      result[i] = this->coef_vars(*idx);
      // printf("time %d instantiating coef %d into state %d with noise variance %f\n", k, *idx, i, result[i]);
    } else {
      result[i] = 0.0;
    }
  }
}

double CompactSupportSSM::apply_observation_matrix(const double * x, int k) {
  // const matrix_row< matrix<int> > active = row(this->active_basis, k);

  double result = 0;

  matrix_row< matrix<int> >::const_iterator idx;
  unsigned int i = 0;
  for (i=0; i < this->active_basis.size2() && this->active_basis(k, i) >= 0; ++i) {
    int basis = this->active_basis(k, i);

    int prototype = (this->identities)(basis);
    int st = (this->start_idxs)(basis);

    result += x[i] * (this->basis_prototypes)(prototype, k-st);
  }

  return result;
}

void CompactSupportSSM::apply_observation_matrix(const matrix<double,column_major> &X,
						 unsigned int row_offset, int k,
						 double *result, double *result_tmp, unsigned int n) {
  //const matrix_row< matrix<int> > active = row(this->active_basis, k);
  //matrix_row< matrix<int> >::const_iterator idx;
  unsigned int i = 0;

  for (unsigned j=0; j < n; ++j) {
    result[j] = 0;
  }

  for (i=0; i < this->active_basis.size2() && this->active_basis(k, i) >= 0; ++i) {
    int basis = this->active_basis(k, i);

    int prototype = this->identities(basis);
    int st = this->start_idxs(basis);

    for (unsigned j=0; j < n; ++j) {
      result[j] += X(row_offset+i,j) * this->basis_prototypes(prototype, k-st);
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

int CompactSupportSSM::prior_mean(double *result) {
  const matrix_row< matrix<int> > active = row(this->active_basis, 0);
  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;

  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {
    result[i] = this->coef_means(*idx);
  }
  for (unsigned ii=i; ii < this->max_dimension; ++ii) {
    result[ii] = 0.0;
  }
  return i;
}

int CompactSupportSSM::prior_vars(double *result) {
  const matrix_row< matrix<int> > active = row(this->active_basis, 0);
  matrix_row< matrix<int> >::const_iterator idx;
  int i = 0;
  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {
    result[i] = this->coef_vars(*idx);
  }
  for (unsigned ii=i; ii < this->max_dimension; ++ii) {
    result[ii] = 0.0;
  }

  return i;
}


void CompactSupportSSM::extract_coefs(const vector<double> &x,
				      const matrix<double> &P,
				      unsigned int state_offset,
				      int k,
				      vector<double> & coef_means,
				      vector<double> & coef_vars) {
  /* given a state estimate at some time k, extract marginals for
   * whatever coefficients we can reasonably do so for at the
   * current time. Earlier estimates will always be overwritten
   * by later ones. */
  const matrix_row< matrix<int> > active = row(this->active_basis, k);
  matrix_row< matrix<int> >::const_iterator idx;
  unsigned int i;


  for (i=0, idx = active.begin();
       idx < active.end() && *idx >= 0;
       ++i, ++idx) {
    int basis_idx = *idx;
    std::pair<int,int> key = std::make_pair(basis_idx, k);

    dense_hash_map< std::pair<int, int>, int, boost::hash< std::pair< int,int> >  >::iterator contains = this->active_indices.find(key);
    if (contains == this->active_indices.end()) {
      printf("\nWAAT, no key found for basis %d\n", basis_idx);
      exit(-1);
    } else {
      int state_idx = this->active_indices[key] + state_offset;
      coef_means(basis_idx) = x(state_idx);
      coef_vars(basis_idx) = P(state_idx,state_idx);
    }
  }
}
