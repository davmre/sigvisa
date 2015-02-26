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
  for (int i=1; i < this->max_dimension; ++i) {
    result[i] = x[i-1];
  }
  return this->max_dimension;
}

void ARSSM::transition_bias(int k, double *result) {
  return;
}

void ARSSM::transition_noise_diag(int k, double *result) {
  result[0] = this->error_var;
  for (int i=1; i < this->max_dimension; ++i) {
    result[i] = 0.0;
  }
}

double ARSSM::apply_observation_matrix(const double *x, int k) {
  return x[0];
}

void ARSSM::apply_observation_matrix(const matrix <double,column_major> &X,
				     int row_offset, int k,
				     double *result, double * result_tmp, int n) {
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
  for (int i=0; i < this->max_dimension; ++i) {
    result[i] = 0;
  }
  return this->max_dimension;
}

int ARSSM::prior_vars(double *result) {
  for(int i=0; i < this->max_dimension; ++i) {
    result[i] = this->error_var;
  }
  return this->max_dimension;
}
