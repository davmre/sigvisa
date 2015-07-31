#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <stdio.h>

#include <cmath>
#include <memory>
#include <vector>
#include <limits>
#include <random>
#include <cmath>

#include "statespace.hpp"

using namespace boost::numeric::ublas;


void udu(matrix<double> &M_mat, matrix<double> &U_mat, vector<double> &d_vec, unsigned int state_size) {

  unsigned int n = M_mat.size1();
  double *M = &(M_mat(0,0));
  double *U = &(U_mat(0,0));
  double *d = &(d_vec(0));

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



  // make M upper triangular
  for (unsigned i=0; i < state_size; ++i) {
    for (unsigned idx=i*n; idx < i*n+i; ++idx) {
      M[idx] = 0;
    }
  }

  // initialize U as the identity
  for (unsigned i=0; i < state_size; ++i) {
    for (unsigned idx=i*n; idx < i*n+state_size; ++idx) {
      U[idx] = 0;
    }
  }
  for (unsigned i=0; i < state_size; ++i) {
    U[i*n+i] = 1;
  }

  // note j cannot be unsigned, to handle state_size=0
  for (int j=state_size-1; j >= 1; --j) {
    d[j] = M[j*n+j];
    double alpha = 0.0;
    double beta = 0.0;
    if (d[j] > 0) {
      alpha = 1.0 / d[j];
    } else {
      if (fabs(d[j] > 1e-5) ) {
	printf("WARNING: nonpositive d[%d] %f in udu decomp\n", j, d[j]);
	exit(-1);
      }
      d[j] = 0.0;
      alpha = 0.0;
    }
    for (unsigned k=0; k < j; ++k) {
      beta = M[k*n+ j ];
      U[k*n+ j] = alpha * beta;
      for (unsigned kk=0; kk <= k; ++kk) {
	M[kk*n+ k] = M[kk*n+k] - beta * U[kk*n+j];
      }
    }
  }
  d[0] = M[0];

  if (d[0] < 0) {
    printf("ERROR: udu decomposition on non-posdef matrix with M(0,0)=%f\n", M[0]);
    exit(-1);
  }
}


FilterState::FilterState(int max_dimension, double eps_stationary) {
  this->eps_stationary = eps_stationary;
  this->at_fixed_point=false;
  this->alpha = 0;
  this->wasnan = false;

  this->obs_U = matrix<double>(max_dimension, max_dimension);
  this->pred_U = matrix<double>(max_dimension, max_dimension);
  this->tmp_U1 = matrix<double>(max_dimension, max_dimension);
  this->tmp_U2 = matrix<double>(max_dimension, max_dimension);

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

void write_vec(const char *fname, const vector<double> & v) {
  FILE *f = fopen(fname, "w");
  if (f == NULL)
    {
      printf("Error opening file!\n");
      exit(1);
    }
  for(unsigned i=0; i < v.size(); ++i) {
    fprintf(f, "%.12f ", v(i));
  }
  fprintf(f, "\n");
  fclose(f);

}


void write_mat_col(const char *fname, const matrix<double> & m) {

  FILE *f = fopen(fname, "w");
  if (f == NULL)
    {
      printf("Error opening file!\n");
      exit(1);
    }

  for(unsigned i=0; i < m.size1(); ++i) {
    for(unsigned j=0; j < m.size2(); ++j) {
      fprintf(f, "%.12f ", m(i,j));
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");

  fclose(f);
}

void write_stuff(const char *item, unsigned int k, const vector<double> & v) {
  char fname[100];
  snprintf(fname, 100, "matrices/%s_c_%d.txt", item, k);
  write_vec(fname, v);
}

void write_stuff(const char *item, unsigned int k, const matrix<double> & m) {
  char fname[100];
  snprintf(fname, 100, "matrices/%s_c_%d.txt", item, k);
  write_mat_col(fname, m);
}


void print_mat_col(const matrix<double> & m) {
  for(unsigned i=0; i < m.size1(); ++i) {
    for(unsigned j=0; j < m.size2(); ++j) {
      printf("%.3f ", m(i,j));
    }
    printf("\n");
  }
  printf("\n");
}


#include <cblas.h>
void compute_explicit_cov_atlas(FilterState &cache,
				matrix<double> &U,
				vector<double> &d,
				int prev_state_size) {

  matrix<double> &mtmp = cache.tmp_U2;
  matrix<double> &P = cache.P;
  unsigned state_size = cache.state_size;

  if (prev_state_size == -1) {
    prev_state_size = state_size;
  }

  // U has dimension (ss x pss)
  for (unsigned i=0; i < state_size; ++i) {
    for (unsigned j=0; j < prev_state_size; ++j) {
      mtmp(i,j) = U(i,j) * d(j);
    }
  }

  cblas_dgemm(CblasRowMajor,
	      CblasNoTrans,
	      CblasTrans,
	      state_size,  state_size, prev_state_size, 1.0,
	      &(mtmp(0,0)), mtmp.size2(),
	      &(U(0,0)), U.size2(),
	      0.0, &(P(0,0)), P.size2());

}

/* void compute_explicit_cov(FilterState &cache,
			  matrix<double> &U_tmp,
			  vector<double> &d_tmp,
			  int prev_state_size) {

  // this is the original version of this function in ublas.
  // it's slow and superceded by the atlas version above.

  matrix<double> &mtmp = cache.tmp_U2;
  matrix<double> &P = cache.P;
  unsigned state_size = cache.state_size;

  if (prev_state_size == -1) {
    prev_state_size = state_size;
  }

  // construct the cov matrix
  for (unsigned i=0; i < prev_state_size; ++i) {
    subrange(mtmp, 0, state_size, i, i+1) = subrange(U_tmp, 0, state_size, i, i+1);
    subrange(mtmp, 0, state_size, i, i+1) *= d_tmp(i);
  }
  noalias(subrange(P, 0, state_size, 0, state_size))		\
    = prod(subrange(mtmp, 0, state_size, 0, prev_state_size),
	   trans(subrange(U_tmp, 0, state_size, 0, prev_state_size)));
} */



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
    // printf("declaring nan at timestep %d, val %f\n", k, zk);
    if (!cache.wasnan && cache.at_fixed_point) {
       cache.at_fixed_point = false;
    }
    cache.wasnan = true;

    // no observation, so the predicted state becomes the "observed" state
    cache.obs_d = cache.pred_d;
    cache.obs_U = cache.pred_U;

    return 0;
  } else {
    if (cache.wasnan && cache.at_fixed_point) {
       cache.at_fixed_point = false;
    }
    cache.wasnan = false;
  }

  double alpha = cache.alpha;
  unsigned int state_size = cache.state_size;
  if (!cache.at_fixed_point || !ssm.stationary(k)) {
    cache.at_fixed_point = false;

    matrix<double> &U_old = cache.pred_U;
    vector<double> &d_old = cache.pred_d;

    matrix<double> &U = cache.obs_U;
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


    D(write_stuff("U_obs_old", k, U_old);)
      D(write_stuff("d_obs_old", k, d_old);)
      D(write_stuff("f", k, f);)
      D(write_stuff("v", k, v);)

    alpha = r + v(0)*f(0);
    if (alpha > 1e-20) {
       d(0) = d_old(0) * r/alpha;
    } else {
      //printf("step %d correcting initial alpha from %f to 1e-20\n", k, alpha);
      d(0) = d_old(0);
      alpha = 1e-20;
    }
    // printf("   alpha C: %f\n", alpha);
    K(0)=v(0);

    U(0, 0) = U_old(0,0);
    for (unsigned j=1; j < state_size; ++j) {
      double old_alpha = alpha;
      alpha += v(j)*f(j);
      //printf("   alpha C %d: %f\n", j, alpha);
      if (alpha > 1e-20) {
	d(j) = d_old(j) * (old_alpha/alpha);
	//printf("d = %f * %f = %f\n" , d_old(j), old_alpha/alpha, d(j));
      } else {
	//printf("step %d correcting alpha from %f to 1e-20\n", k, alpha);
	d(j) = d_old(j);
	alpha = 1e-20;
      }

      for (unsigned i=0; i < state_size; ++i) {
	U(i, j) = U_old(i,j) - (K(i)/old_alpha)*f(j);
        K(i) += v(j) * U_old(i,j);
      }

    }
    cache.alpha = alpha;

      D(write_stuff("U_obs", k, U);)
      D(write_stuff("d_obs", k, d);)

  }

  // given the Kalman gain from the covariance update, compute
  // the updated mean vector.
  vector<double> &xk = cache.xk;
  double pred_z = ssm.apply_observation_matrix(&(xk(0)), k) + ssm.observation_bias(k);
  cache.pred_z = pred_z;
  double yk = zk - pred_z;
  for (unsigned i=0; i < state_size; ++i) {
    xk(i) += cache.gain(i) * yk/alpha;
  }

  // also compute log marginal likelihood for this observation
  double step_ell = -.5 * log(2*PI*alpha) - .5 * yk*yk / alpha;

  //printf("step %d (C) pred %.4f alpha %.4f z %.4f y %.4f ell %.4f\n", k, pred_z, alpha, zk, yk, step_ell);

  if (isnan(step_ell)) {
    printf("step %d (C) pred %.4f alpha %.4f z %.4f y %.4f ell %.4f\n", k, pred_z, alpha, zk, yk, step_ell);
    print_vec(cache.obs_d);
    printf("\n");
    print_vec(cache.xk);
    exit(-1);
  }

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

  D(write_stuff("xk_posttransit", k, xk);)

  if (cache.at_fixed_point and ssm.stationary(k)) {
    return;
  }

  cache.at_fixed_point = false;

  matrix<double> &U_old = cache.obs_U;
  vector<double> &d_old = cache.obs_d;

  matrix<double> &U_tmp = cache.tmp_U1;
  vector<double> &d_tmp = cache.v;

  // get transition noise into temporary storage
  ssm.transition_noise_diag(k, &(tmp(0)));

  /* pushing the covariance P through the transition model F yields
     FPF'. In a factored representation, this is FUDU'F', so we just need
     to compute FU. */
  unsigned int min_size = std::min(prev_state_size, state_size);



  D(write_stuff("U_pretransit", k, U_old);)


  // COMMENTED OUT: this loop is equivalent to the matrix-valued transition call
  // directly below. I'm leaving it in for debugging and to run speed comparisons.

    /*
  for (int i=0; i < min_size; ++i) {
    // THIS ONLY WORKS IF U_old is in column-major order
    ssm.apply_transition_matrix(&(column(U_old, i)(0) ), k, &(d_tmp(0)) );
    noalias(column(U_tmp, i)) = d_tmp;
    }*/
  subrange(d_tmp, 0, prev_state_size) = subrange(d_old, 0, prev_state_size);
  ssm.apply_transition_matrix(U_old, 0, k, U_tmp, 0, prev_state_size);
  for (unsigned i=prev_state_size; i < state_size; ++i) {
    d_tmp(i) = 0;
    for (unsigned j=0; j < state_size; ++j) {
      U_tmp(j, i) = 0;
    }
  }

    D(write_stuff("d_posttransit", k, d_tmp);)
    D(write_stuff("U_posttransit", k, U_tmp);)

  // if there is transition noise, do the expensive reconstruction/factoring step
  if (force_P || state_size != prev_state_size || norm_2(subrange(tmp, 0, state_size)) > 0) {

    compute_explicit_cov_atlas(cache, U_tmp, d_tmp, prev_state_size);
    // add transition noise
    matrix<double> &P = cache.P;


    D(write_stuff("P_prenoise", k, P);)


    for (unsigned i=0; i < state_size; ++i) {
      P(i,i) += tmp(i);
    }
    //printf("noise at time %d ", k);
    //print_vec(tmp);


      D(write_stuff("P", k, P);)


    // printf("step %d state size %d\n", k, state_size);

    // udu overwrites the cov matrix, so we need to
    // save it if we're going to explicitly use it
    // later on.
    matrix<double> & mtmp = cache.tmp_U2;
    if (force_P) {
      mtmp = P;
    }

    // get the new factored representation
    udu(P, U_tmp, d_tmp, state_size);

      D(write_stuff("d_decomp", k, d_tmp);)
      D(write_stuff("U_decomp", k, U_tmp);)

    if (force_P) {
      P = mtmp;
    }

  }

  // if our factored representation is (almost) the same as the previous invocation,
  // we've reached a stationary state
  matrix<double> &U_cached = cache.pred_U;
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

  D(write_stuff("U_prior", 0, cache.pred_U);)
  D(write_stuff("d_prior", 0, cache.pred_d);)
  D(write_stuff("xk_prior", 0, cache.xk);)


  ell += kalman_observe_sqrt(ssm, cache, 0, z(0));



  D(write_stuff("U_post_obs", 0, cache.obs_U);)
    D(write_stuff("d_post_obs", 0, cache.obs_d);)
    D(write_stuff("xk_post_obs", 0, cache.xk);)

  for (unsigned k=1; k < N; ++k) {
    kalman_predict_sqrt(ssm, cache, k, false);

    D(write_stuff("U_post_predict", k, cache.pred_U);)
      D(write_stuff("d_post_predict", k, cache.pred_d);)
      D(write_stuff("xk_post_predict", k, cache.xk);)

    ell += kalman_observe_sqrt(ssm, cache, k, z(k));


      D(write_stuff("U_post_obs", k, cache.obs_U);)
      D(write_stuff("d_post_obs", k, cache.obs_d);)
      D(write_stuff("xk_post_obs", k, cache.xk);)



  }
  return ell;
}


void step_obs_likelihoods(StateSpaceModel &ssm, const vector<double> &z,
			  vector<double> & ells,
			  vector<double> & preds,
			  vector<double> & alphas) {
  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);
  unsigned int N = z.size();
  double ell = 0;
  ells(0) = kalman_observe_sqrt(ssm, cache, 0, z(0));
  preds(0) = cache.pred_z;
  alphas(0) = cache.alpha;
  for (unsigned k=1; k < N; ++k) {
    kalman_predict_sqrt(ssm, cache, k, false);
    ells(k) = kalman_observe_sqrt(ssm, cache, k, z(k));
    preds(k) = cache.pred_z;
    alphas(k) = cache.alpha;
    //printf("got ell %f at step %d\n", ells(k), k);
  }
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
      ssm.transition_bias(k+1, &(x2(0)));
    }
    x = x2; // this copy is unnecessary, we could swap
            // pointers instead, but it doesn't
            // matter cause this method is never the
            // performance bottleneck.
  }
}

void obs_var(StateSpaceModel &ssm, vector<double> & result) {
  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);
  compute_explicit_cov_atlas(cache, cache.pred_U, cache.pred_d, -1);

  matrix<double> P = cache.P;

  for (unsigned k = 0; k < result.size(); ++k) {
    ssm.apply_observation_matrix(P, 0,
    				 k, &(cache.f(0)), &(cache.v(0)), cache.f.size());
    result(k) = ssm.apply_observation_matrix(&(cache.f(0)), k);
    result(k) += ssm.observation_noise(k);
    if (k+1 < result.size()) {

      ssm.apply_transition_matrix(P, 0, k+1, cache.obs_U, 0, ssm.max_dimension);
      cache.obs_U = trans(cache.obs_U);
      ssm.apply_transition_matrix(cache.obs_U, 0, k+1, P, 0, ssm.max_dimension);

      vector<double> &tmp = cache.f;
      ssm.transition_noise_diag(k+1, &(tmp(0)));
      for (unsigned i=0; i < ssm.max_dimension; ++i) {
	P(i,i) += tmp(i);
      }
    }
  }
}

void prior_sample(StateSpaceModel &ssm, vector<double> & result, unsigned long seed) {
  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);

  std::mt19937 gen(seed);
  std::normal_distribution<double> randn(0,1);
  /* auto randn = std::bind(std::normal_distribution<double>(0,1),
			     ); */

  printf("seed with %lu, first randn %f\n", seed, randn(gen));

  // sample initial state from the prior
  vector<double> &d = cache.pred_d;
  for (unsigned i=0; i < ssm.max_dimension; ++i) {
    cache.xk(i) += randn(gen) * sqrt(d(i));
  }


  unsigned k = 0;
  result(k) = ssm.apply_observation_matrix(&(cache.xk(0)), k);
  result(k) += ssm.observation_bias(k);

  for (k=1; k < result.size(); ++k) {
    cache.obs_d = cache.pred_d;
    cache.obs_U = cache.pred_U;

    vector<double> &tmp = cache.f;
    unsigned int state_size = ssm.apply_transition_matrix( &(cache.xk(0)), k,  &(tmp(0)));
    vector<double> &xk = cache.xk;
    subrange(xk, 0, state_size) = subrange(tmp, 0, state_size);
    ssm.transition_bias(k, &(xk(0)));

    ssm.transition_noise_diag(k, &(tmp(0)));
    for (unsigned i=0; i < state_size; ++i) {
      cache.xk(i) += randn(gen) * sqrt(tmp(i));
    }

    result(k) = ssm.apply_observation_matrix(&(cache.xk(0)), k);
    result(k) += ssm.observation_bias(k);
    result(k) += randn(gen) * sqrt(ssm.observation_noise(k));

  }
}




double all_filtered_cssm_coef_marginals(TransientCombinedSSM &ssm,
				      const vector<double> &z,
				      vector<double> & step_ells,
				      std::vector<vector<double> > & cmeans,
				      std::vector<vector<double> > & cvars) {
  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);
  ssm.init_coef_priors(cmeans, cvars);

  unsigned int N = z.size();
  double ell = 0;
  step_ells[0] = kalman_observe_sqrt(ssm, cache, 0, z(0));
  ell += step_ells[0];
  compute_explicit_cov_atlas(cache, cache.obs_U, cache.obs_d, -1);
  ssm.extract_all_coefs(cache, 0, cmeans, cvars);
  for (unsigned k=1; k < N; ++k) {
    kalman_predict_sqrt(ssm, cache, k, false);
    step_ells[k] = kalman_observe_sqrt(ssm, cache, k, z(k));
    ell += step_ells[k];

    compute_explicit_cov_atlas(cache, cache.obs_U, cache.obs_d, -1);
    ssm.extract_all_coefs(cache, k, cmeans, cvars);
  }
  return ell;
}

double tssm_component_means(TransientCombinedSSM &ssm,
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
  return ell;
}


double tssm_component_vars(TransientCombinedSSM &ssm,
			  const vector<double> &z,
			  std::vector<vector<double> > & vars) {

  if (vars.size() != ssm.n_ssms) {
    printf("component_vars() needs exactly one vector for each component SSM\n");
    exit(-1);
  }

  FilterState cache(ssm.max_dimension, 1e-10);
  cache.init_priors(ssm);
  compute_explicit_cov_atlas(cache, cache.pred_U, cache.pred_d, -1);

  unsigned int N = z.size();
  double ell = 0;
  ell += kalman_observe_sqrt(ssm, cache, 0, z(0));
  compute_explicit_cov_atlas(cache, cache.obs_U, cache.obs_d, -1);
  ssm.extract_component_vars(cache.P, cache.tmp_U2, 0, vars);
  for (unsigned k=1; k < N; ++k) {
    kalman_predict_sqrt(ssm, cache, k, false);
    ell += kalman_observe_sqrt(ssm, cache, k, z(k));
    compute_explicit_cov_atlas(cache, cache.obs_U, cache.obs_d, -1);
    ssm.extract_component_vars(cache.P, cache.tmp_U2, k, vars);
  }
  return ell;
}
