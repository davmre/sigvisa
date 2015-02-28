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


ARSSM::ARSSM(const vector<double> & params, double error_var,
	     double obs_noise, double bias) : params(params), error_var(error_var),
					      obs_noise(obs_noise), bias(bias) {

  this->is_cssm = false;
  this->max_dimension = params.size();
}

ARSSM::~ARSSM() {
  return;
};


int ARSSM::apply_transition_matrix(const double * x, int k, double * result) {
  double ip = 0;
  for (unsigned i=0; i < this->params.size(); ++i) {
    ip += x[i] * this->params(i);
  }
  result[0] = ip;
  for (unsigned i=1; i < this->max_dimension; ++i) {
    result[i] = x[i-1];
  }
  return this->max_dimension;
}

int ARSSM::apply_transition_matrix(const matrix <double,column_major> &X,
				    unsigned int x_row_offset, int k,
				    matrix <double,column_major> &result,
				    unsigned int r_row_offset,
				    unsigned int n) {
  for (unsigned j=0; j < n; ++j) {
    double ip = 0;
    for (unsigned i=0; i < this->params.size(); ++i) {
      ip += X(i+x_row_offset,j) * this->params(i);
    }
    result(r_row_offset,j) = ip;
    for (unsigned i=1; i < this->max_dimension; ++i) {
      result(i+r_row_offset,j) = X(i-1+x_row_offset,j);
    }
  }
  return this->max_dimension;
}


void ARSSM::transition_bias(int k, double *result) {
  return;
}

void ARSSM::transition_noise_diag(int k, double *result) {
  result[0] = this->error_var;
  for (unsigned i=1; i < this->max_dimension; ++i) {
    result[i] = 0.0;
  }
}

double ARSSM::apply_observation_matrix(const double *x, int k) {
  return x[0];
}

void ARSSM::apply_observation_matrix(const matrix <double,column_major> &X,
				     unsigned int row_offset, int k,
				     double *result, double * result_tmp, unsigned int n) {
  for (unsigned i=0; i < n; ++i) {
    result[i] = X(row_offset, i);
  }
}


double ARSSM::observation_bias(int k) {
  return this->bias;
}

double ARSSM::observation_noise(int k) {
  return this->obs_noise;
}

bool ARSSM::stationary(int k) {
  return true;
}

int ARSSM::prior_mean(double *result) {
  for (unsigned i=0; i < this->max_dimension; ++i) {
    result[i] = 0;
  }
  return this->max_dimension;
}

int ARSSM::prior_vars(double *result) {
  for(unsigned i=0; i < this->max_dimension; ++i) {
    result[i] = this->error_var;
  }
  return this->max_dimension;
}
