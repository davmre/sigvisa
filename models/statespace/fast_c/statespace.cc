#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <cmath>
#include <memory>
#include <vector>
#include <limits>
#include <random>
#include <cmath>

#include "statespace.hpp"

using namespace boost::numeric::ublas;

void udu(matrix<double> &M, matrix<double,column_major> &U, vector<double> &d, unsigned int state_size) {
  // this method stolen from pykalman.sqrt.bierman by Daniel Duckworth (BSD license)
  /*Construct the UDU' decomposition of a positive, semidefinite matrix M

    Parameters
    ----------
    M : [n, n] array
        Matrix to factorize

    Returns
    -------
    UDU : UDU_decomposition of size n
        UDU' representation of M
  */

  unsigned int n = state_size;

  // make M upper triangular: not sure if this is necessary?
  for (unsigned i=0; i < n; ++i) {
    for (unsigned j=0; j < i; ++j) {
      M(i,j) = 0;
    }
  }

  d.clear();
  U.clear();
  for (unsigned i=0; i < n; ++i) {
    U(i,i) = 1;
  }

  for (unsigned j=n; j >= 2; --j) {
    d(j - 1) = M(j - 1, j - 1);
    double alpha = 0.0;
    double beta = 0.0;
    if (d(j - 1) > 0) {
      alpha = 1.0 / d(j - 1);
    } else {
      d(j-1) = 0.0;
      alpha = 0.0;
    }
    for (unsigned k=1; k < j; ++k) {
      beta = M(k - 1, j - 1);
      U(k - 1, j - 1) = alpha * beta;

      // M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]
      for (unsigned kk=0; kk < k; ++kk) {
	M(kk, k - 1) = M(kk, k - 1) - beta * U(kk, j - 1);
      }
    }
  }
  d(0) = M(0, 0);

  if (d(0) < 0) {
    printf("ERROR: udu decomposition on non-posdef matrix with M(0,0)=%f\n", M(0,0));
    exit(-1);
  }

}

FilterState::FilterState(int max_dimension, double eps_stationary) {
  this->eps_stationary = eps_stationary;
  this->at_fixed_point=false;
  this->alpha = 0;
  this->wasnan = false;

  this->obs_U = matrix<double,column_major>(max_dimension, max_dimension);
  this->pred_U = matrix<double,column_major>(max_dimension, max_dimension);
  this->tmp_U1 = matrix<double,column_major>(max_dimension, max_dimension);
  this->tmp_U2 = matrix<double,column_major>(max_dimension, max_dimension);

  // we only ever write to the upper triangle of this matrix, so
  // zeroing the full matrix now guarantees that it will always be zero.
  this->obs_U.clear();
  this->pred_U.clear();
  this->tmp_U1.clear();
  this->tmp_U2.clear();

  this->P = matrix<double>(max_dimension, max_dimension);

  this->obs_d = vector<double>(max_dimension);
  this->pred_d = vector<double>(max_dimension);
  this->obs_d.clear();
  this->pred_d.clear();

  this->gain = vector<double>(max_dimension);
  this->f = vector<double>(max_dimension);
  this->v = vector<double>(max_dimension);

  this->xk = vector<double>(max_dimension);
}

void print_vec(const vector<double> & v) {
  for(unsigned i=0; i < v.size(); ++i) {
    printf("%.3f ", v(i));
  }
  printf("\n");
}

void print_mat(const matrix<double> & m) {
  for(unsigned i=0; i < m.size1(); ++i) {
    for(unsigned j=0; j < m.size2(); ++j) {
      printf("%.3f ", m(i,j));
    }
    printf("\n");
  }
  printf("\n");
}

void print_mat_col(const matrix<double,column_major> & m) {
  for(unsigned i=0; i < m.size1(); ++i) {
    for(unsigned j=0; j < m.size2(); ++j) {
      printf("%.3f ", m(i,j));
    }
    printf("\n");
  }
  printf("\n");
}

void compute_explicit_cov(FilterState &cache,
			  matrix<double,column_major> &U_tmp,
			  vector<double> &d_tmp) {
  matrix<double,column_major> &mtmp = cache.tmp_U2;
  matrix<double> &P = cache.P;
  unsigned state_size = cache.state_size;

  // construct the cov matrix
  for (unsigned i=0; i < state_size; ++i) {
    subrange(mtmp, 0, state_size, i, i+1) = subrange(U_tmp, 0, state_size, i, i+1);
    subrange(mtmp, 0, state_size, i, i+1) *= d_tmp(i);
  }
  noalias(subrange(P, 0, state_size, 0, state_size))		\
    = prod(subrange(mtmp, 0, state_size, 0, state_size),
	   trans(subrange(U_tmp, 0, state_size, 0, state_size)));
}



void FilterState::init_priors(StateSpaceModel &ssm) {
  this->pred_d.clear();
  this->xk.clear();

  this->state_size = ssm.prior_mean(&(this->xk(0)));
  this->state_size = ssm.prior_vars(&(this->pred_d(0)));

  this->pred_U.clear();
  for (unsigned i=0; i < ssm.max_dimension; ++i) {
    this->pred_U(i,i) = 1;
  }

  return;
}


double kalman_observe_sqrt(StateSpaceModel &ssm, FilterState &cache, int k, double zk) {

  // if needed, use this for isnan: #include <boost/math/special_functions/fpclassify.hpp>
  if (isnan(zk)) {
    if (cache.at_fixed_point && !cache.wasnan) {
       cache.at_fixed_point = false;
    }
    cache.wasnan = true;
    return 0;
  } else {
    if (cache.at_fixed_point && cache.wasnan) {
       cache.at_fixed_point = false;
    }
    cache.wasnan = false;
  }

  double alpha = cache.alpha;
  unsigned int state_size = cache.state_size;
  if (!cache.at_fixed_point || !ssm.stationary(k)) {
    cache.at_fixed_point = false;

    matrix<double,column_major> &U_old = cache.pred_U;
    vector<double> &d_old = cache.pred_d;

    matrix<double,column_major> &U = cache.obs_U;
    vector<double> &d = cache.obs_d;
    vector<double> &K = cache.gain;
    vector<double> &f = cache.f;
    vector<double> &v = cache.v;
    double r = ssm.observation_noise(k);

    K.clear();

    ssm.apply_observation_matrix(U_old, 0,
				 k, &(f(0)), &(v(0)), state_size);


    for (unsigned i=0; i < state_size; ++i) {
      v(i) = d_old(i)*f(i);
    }

    //printf("predicted obs ");
    //print_vec(f);

    // printf("got v ");
    // print_vec(v);

    alpha = r + v(0)*f(0);
    if (alpha > 1e-20) {
       d(0) = d_old(0) * r/alpha;
    } else {
      d(0) = d_old(0);
      alpha = 1e-20;
    }
    // printf("   alpha C: %f\n", alpha);
    K(0)=v(0);

    U(0, 0) = U_old(0,0);
    for (unsigned j=1; j < state_size; ++j) {
      double old_alpha = alpha;
      alpha += v(j)*f(j);
      // printf("   alpha C %d: %f\n", j, alpha);
      if (alpha > 1e-20) {
         d(j) = d_old(j) * old_alpha/alpha;
      } else {
	d(j) = d_old(j);
	alpha = 1e-20;
      }

      for (unsigned i=0; i < state_size; ++i) {
	U(i, j) = U_old(i,j) - f(j)/old_alpha*K(i);
        K(i) += v(j) * U_old(i,j);
      }

    }
    cache.alpha = alpha;
  }

  // given the Kalman gain from the covariance update, compute
  // the updated mean vector.
  vector<double> &xk = cache.xk;
  double pred_z = ssm.apply_observation_matrix(&(xk(0)), k) + ssm.observation_bias(k);
  double yk = zk - pred_z;
  for (unsigned i=0; i < state_size; ++i) {
    xk(i) += cache.gain(i) * yk/alpha;
  }

  // also compute log marginal likelihood for this observation
  double step_ell = -.5 * log(2*PI*alpha) - .5 * yk*yk / alpha;

  //printf("step %d (C) pred %.4f alpha %.4f z %.4f y %.4f ell %.4f\n", k, pred_z, alpha, zk, yk, step_ell);

  return step_ell;
}

void kalman_predict_sqrt(StateSpaceModel &ssm, FilterState &cache, int k, bool force_P) {

  unsigned int prev_state_size = cache.state_size;

  vector<double> &tmp = cache.f;

  unsigned int state_size = ssm.apply_transition_matrix( &(cache.xk(0)), k,  &(tmp(0)));
  cache.state_size = state_size;

  vector<double> &xk = cache.xk;
  subrange(xk, 0, state_size) = subrange(tmp, 0, state_size);
  ssm.transition_bias(k, &(xk(0)));

  if (cache.at_fixed_point and ssm.stationary(k)) {
    return;
  }

  cache.at_fixed_point = false;

  matrix<double,column_major> &U_old = cache.obs_U;
  vector<double> &d_old = cache.obs_d;

  matrix<double,column_major> &U_tmp = cache.tmp_U1;
  vector<double> &d_tmp = cache.v;

  // get transition noise into temporary storage
  ssm.transition_noise_diag(k, &(tmp(0)));

  /* pushing the covariance P through the transition model F yields
     FPF'. In a factored representation, this is FUDU'F', so we just need
     to compute FU. */
  unsigned int min_size = std::min(prev_state_size, state_size);


  // COMMENTED OUT: this loop is equivalent to the matrix-valued transition call
  // directly below. I'm leaving it in for debugging and to run speed comparisons.
  /*for (int i=0; i < min_size; ++i) {
    // THIS ONLY WORKS IF U_old is in column-major order
    ssm.apply_transition_matrix(&(column(U_old, i)(0) ), k, &(d_tmp(0)) );
    noalias(column(U_tmp, i)) = d_tmp;
    }*/
  ssm.apply_transition_matrix(U_old, 0, k, U_tmp, 0, min_size);

  subrange(d_tmp, 0, state_size) = subrange(d_old, 0, state_size);
  for (unsigned i=prev_state_size; i < state_size; ++i) {
    d_tmp(i) = 0;
    for (unsigned j=0; j < state_size; ++j) {
      U_tmp(j, i) = 0;
    }
  }

  // if there is transition noise, do the expensive reconstruction/factoring step
  if (force_P || norm_2(subrange(tmp, 0, state_size)) > 0) {
    matrix<double> &P = cache.P;

    compute_explicit_cov(cache, U_tmp, d_tmp);

    // add transition noise
    for (unsigned i=0; i < state_size; ++i) {
      P(i,i) += tmp(i);
    }

    // get the new factored representation
    udu(P, U_tmp, d_tmp, state_size);
  }

  // if our factored representation is (almost) the same as the previous invocation,
  // we've reached a stationary state
  matrix<double,column_major> &U_cached = cache.pred_U;
  vector<double> &d_cached = cache.pred_d;
  if (ssm.stationary(k)) {
    if (k > 0 && ssm.stationary(k-1)) {
      bool potential_fixed_point = true;
      for (unsigned i=0; i < state_size; ++i) {
	if (std::abs(d_tmp(i) - d_cached(i)) > cache.eps_stationary) {
	  potential_fixed_point=false;
	  break;
	}
	for (unsigned j=0; j < state_size; ++j) {
	  if (std::abs(U_tmp(i,j) - U_cached(i,j)) > cache.eps_stationary) {
	    potential_fixed_point=false;
	    break;
	  }
	}
	if (!potential_fixed_point) {
	  break;
	}
      }
      if (potential_fixed_point) {
	cache.at_fixed_point = true;
      }
    }
  }
  if (!cache.at_fixed_point) {
    subrange(U_cached, 0, state_size, 0, state_size) = subrange(U_tmp, 0, state_size, 0, state_size);
    subrange(d_cached, 0, state_size) = subrange(d_tmp, 0, state_size);
  }
}

double filter_likelihood(StateSpaceModel &ssm, const vector<double> &z) {
  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);
  unsigned int N = z.size();
  double ell = 0;
  ell += kalman_observe_sqrt(ssm, cache, 0, z(0));

  /*printf("post observe(0) obs_U ");
  print_mat(cache.obs_U);
  printf("post observe(0) obs_d ");
  print_vec(cache.obs_d);*/

  for (unsigned k=1; k < N; ++k) {


    kalman_predict_sqrt(ssm, cache, k, false);

    /*printf("post pred(%d) U ", k);
      print_mat(cache.pred_U);

  printf("post pred(%d) d ", k);
  print_vec(cache.pred_d);

  printf("post pred(%d) xk ", k);
  print_vec(cache.xk);*/

    ell += kalman_observe_sqrt(ssm, cache, k, z(k));

    /*printf("post observe(%d) U ", k);
      print_mat(cache.obs_U);
  printf("post obs(%d) d ", k);
  print_vec(cache.obs_d);

  printf("post obs(%d) xk ", k);
  print_vec(cache.xk);*/


  }
  return ell;
}

void mean_obs(StateSpaceModel &ssm, vector<double> & result) {
  vector<double> x(ssm.max_dimension);
  vector<double> x2(ssm.max_dimension);
  x.clear();
  x2.clear();

  ssm.prior_mean(&(x(0)));

  for (unsigned k = 0; k < result.size(); ++k) {
    result[k] = ssm.apply_observation_matrix(&(x(0)), k);
    result[k] += ssm.observation_bias(k);

    if (k+1 < result.size()) {
      ssm.apply_transition_matrix(&(x(0)), k+1, &(x2(0)));
    }
    x = x2; // this copy is unnecessary, we could swap
            // pointers instead, but it doesn't
            // matter cause this method is never the
            // performance bottleneck.
  }
}



void resample_state(FilterState &cache) {
  /* Given a cache representing a mean state and covariance matrix at
   * time k, replace the state xk with a sample from N(mean, cov).
   * That is, suppose we have already pushed the previous state through
   * the transition model to get an expected state at the current
   * timestep; this method samples the new, random state.*/

  vector<double> &x = cache.xk;
  vector<double> &d = cache.pred_d;
  vector<double> &tmp = cache.f;
  matrix<double,column_major> &U = cache.pred_U;

  // we have P = UDU'
  // so U*sqrt(d) is a matrix square root of P
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> randn(0,1);

  for (unsigned i=0; i < tmp.size(); ++i) {
    tmp(i) = randn(rd) * sqrt(d(i));
  }
  tmp = prod(U, tmp);
  x += tmp;

}

void prior_sample(StateSpaceModel &ssm, vector<double> & result) {
  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);

  for (unsigned k = 0; k < result.size(); ++k) {
    resample_state(cache);
    result(k) = ssm.apply_observation_matrix(&(cache.xk(0)), k);
    result(k) += ssm.observation_bias(k);
    if (k+1 < result.size()) {
      kalman_predict_sqrt(ssm, cache, k+1, true);
    }
  }
}


void all_filtered_cssm_coef_marginals(TransientCombinedSSM &ssm,
				      const vector<double> &z,
				      std::vector<vector<double> > & cmeans,
				      std::vector<vector<double> > & cvars) {
  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);
  ssm.init_coef_priors(cmeans, cvars);

  unsigned int N = z.size();
  double ell = 0;
  ell += kalman_observe_sqrt(ssm, cache, 0, z(0));
  ssm.extract_all_coefs(cache, 0, cmeans, cvars);
  for (unsigned k=1; k < N; ++k) {
    kalman_predict_sqrt(ssm, cache, k, false);
    ell += kalman_observe_sqrt(ssm, cache, k, z(k));

    compute_explicit_cov(cache, cache.obs_U, cache.obs_d);
    ssm.extract_all_coefs(cache, k, cmeans, cvars);
  }
}

void tssm_component_means(TransientCombinedSSM &ssm,
			  const vector<double> &z,
			  std::vector<vector<double> > & means) {

  if (means.size() != ssm.n_ssms) {
    printf("component_means() needs exactly one vector for each component SSM\n");
    exit(-1);
  }

  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);

  unsigned int N = z.size();
  double ell = 0;
  ell += kalman_observe_sqrt(ssm, cache, 0, z(0));
  ssm.extract_component_means(&(cache.xk(0)), 0, means);
  for (unsigned k=1; k < N; ++k) {
    kalman_predict_sqrt(ssm, cache, k, false);
    ell += kalman_observe_sqrt(ssm, cache, k, z(k));
    ssm.extract_component_means(&(cache.xk(0)), k, means);
  }
}
