#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdarg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../sigvisa.h"
#include "matrix_util.h"
#include "kalman_filter.h"

/* Augments the mean vector and covariance matrix with space for a new order-m AR process, and fills in the transition matrix with the coefficients ar_coeffs. */
int kalman_add_AR_process(KalmanState_t * k, ARProcess_t * p) {

  int m = p->order;

  /* fill in the index of the new process */
  resize_vector(&k->p_process_indices, (k->np)+1);
  gsl_vector_set(k->p_process_indices, k->np, k->n);

  resize_vector(&k->p_const, (k->np)+1);
  resize_vector(&k->p_means, (k->n)+m);
  resize_vector(&k->p_process_noise, (k->n)+m);
  resize_vector(&k->p_sample_state, (k->n)+m);
  resize_matrix(&k->p_covars, (k->n)+m, (k->n)+m, FALSE);
  resize_matrix(&k->p_transition, (k->n)+m, (k->n)+m, FALSE);

  gsl_vector_set(k->p_process_noise, k->n, p->sigma2);

  /* fill in the new entries for the transition matrix */
  for (int i=0; i < m; ++i) {
    gsl_matrix_set(k->p_transition, k->n, (k->n)+i, p->coeffs[i]);

    if (i <m-1) {
      gsl_matrix_set(k->p_transition, (k->n)+i+1, (k->n)+i, 1);
    }
  }

  gsl_vector_set(k->p_const, k->np, p->mean);

  k->n += m;
  k->np++;

  // now resize all the temp matrices
  int L = k->np;
  realloc_matrix(&k->P, L, L);
  realloc_matrix(&k->p_sigma_points, L, 2*L+1);
  realloc_matrix(&k->p_obs_points, k->obs_n, 2*L+1);
  realloc_vector(&k->p_collapsed_means, k->np);
  realloc_matrix(&k->p_collapsed_covars, k->np, k->np);
  realloc_matrix(&k->K, k->np, k->obs_n);
  realloc_matrix(&k->Ktmp, k->np, k->obs_n);
  realloc_vector(&k->p_weights, 2*L+1);
  realloc_vector(&k->p_collapsed_mean_update, L);
  realloc_vector(&k->p_mean_update, (k->n));
  realloc_matrix(&k->p_covars_tmp, (k->n), (k->n));

  return k->np - 1;

}

/* Removes the order-m AR process with specified index arridx from the mean vector and covariance matrix. */
void kalman_remove_AR_process(KalmanState_t * k, int m, int process_idx) {

  int arridx = gsl_vector_get(k->p_process_indices, process_idx);
  remove_vector_slice(&k->p_process_indices, process_idx, 1);

  remove_vector_slice(&k->p_const, process_idx, 1);
  remove_vector_slice(&k->p_means, arridx, m);
  remove_vector_slice(&k->p_process_noise, arridx, m);
  remove_vector_slice(&k->p_sample_state, arridx, m);
  remove_matrix_slice(&k->p_covars, arridx, m);
  remove_matrix_slice(&k->p_transition, arridx, m);

  k->n -= m;
  k->np--;
  // now resize all the temp matrices
  int L = k->np;
  realloc_matrix(&k->P, L, L);
  realloc_matrix(&k->p_sigma_points, L, 2*L+1);
  realloc_matrix(&k->p_obs_points, k->obs_n, 2*L+1);
  realloc_vector(&k->p_collapsed_means, k->np);
  realloc_matrix(&k->p_collapsed_covars, k->np, k->np);
  realloc_matrix(&k->K, k->np, k->obs_n);
  realloc_matrix(&k->Ktmp, k->np, k->obs_n);
  realloc_vector(&k->p_weights, 2*L+1);
  realloc_vector(&k->p_collapsed_mean_update, L);
  realloc_vector(&k->p_mean_update, (k->n));
  realloc_matrix(&k->p_covars_tmp, (k->n), (k->n));
}

/* constructs the observation matrix for the current set of AR processes */
/* void AR_observation(int n, int k,
		    int n_arrs, ArrivalWaveform_t * active_arrivals,
		    Segment_t * p_segment,
		    gsl_matrix ** pp_obs) {
  if (*pp_obs != NULL) {
    gsl_matrix_free(*pp_obs);
  }

  if (n_arrs == 0) {
    *pp_obs = NULL;
    return;
  }

  *pp_obs = gsl_matrix_alloc(k, n*n_arrs);
  gsl_matrix_set_zero(*pp_obs);

  for (int arr=0; arr < n_arrs; ++arr) {
    ArrivalWaveform_t * p_a = active_arrivals;
   int curr_chan=0;
    for (int i=0; i < NUM_CHANS; ++i) {
      if (p_segment->p_channels[i] != NULL) {
	switch (i) {
	case CHAN_BHZ:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]);
	  break;
	case CHAN_BHN:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]);
	  break;
	case CHAN_BHE:
	  gsl_matrix_set(*pp_obs, curr_chan, arr*n+n-1, p_a->bhz_coeff * p_a->p_envelope[p_a->idx]);
	  break;
	}
	curr_chan++;
      }
    }

    active_arrivals=active_arrivals->next_active;
  }


} */

/* predict the distribution at the next timestep */
void kalman_predict(KalmanState_t * k) {

  /* propagate the means through the transition model */
  gsl_blas_dgemv (CblasNoTrans, 1, k->p_transition, k->p_means, 0, k->p_mean_update);
  gsl_vector_memcpy(k->p_means, k->p_mean_update);


  // propagate the covariance matrix
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, k->p_covars, k->p_transition, 0, k->p_covars_tmp);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, k->p_transition, k->p_covars_tmp, 0, k->p_covars);

  /* add the process noise variance */
  gsl_vector_view covar_diag = gsl_matrix_diagonal(k->p_covars);
  gsl_vector_add(&covar_diag.vector, k->p_process_noise);

}


void collapse_vector(KalmanState_t *k, gsl_vector * p_orig, gsl_vector * p_collapsed) {
  for(int i=0; i < k->np; ++i) {
    int i_full = gsl_vector_get(k->p_process_indices, i);
    double m = gsl_vector_get(p_orig, i_full);
    gsl_vector_set(p_collapsed, i, m);
  }
}

void collapse_state(KalmanState_t *k) {
  for(int i=0; i < k->np; ++i) {
    int i_full = gsl_vector_get(k->p_process_indices, i);
    double m = gsl_vector_get(k->p_means, i_full);
    gsl_vector_set(k->p_collapsed_means, i, m);

    for(int j=0; j < k->np; ++j) {
      int j_full = gsl_vector_get(k->p_process_indices, j);
      double c = gsl_matrix_get(k->p_covars, i_full, j_full);
      gsl_matrix_set(k->p_collapsed_covars, i, j, c);
    }
  }
}

void uncollapse_state(KalmanState_t *k) {
for(int i=0; i < k->np; ++i) {
    int i_full = gsl_vector_get(k->p_process_indices, i);
    double m = gsl_vector_get(k->p_collapsed_means, i);
    gsl_vector_set(k->p_means, i_full, m);

    for(int j=0; j < k->np; ++j) {
      int j_full = gsl_vector_get(k->p_process_indices, j);
      double c = gsl_matrix_get(k->p_collapsed_covars, i, j);
      gsl_matrix_set(k->p_covars, i_full, j_full, c);
    }
  }
}

/* Uses the unscented transform to compute the measurement mean and covariance; then uses these to update the Kalman filtering distribution. */
double kalman_nonlinear_update(KalmanState_t *k,  gsl_vector * p_true_obs, ...) {

  /* for the update, we work only with the current-time components of the state, and we add in the process mean */
  collapse_state(k);
  gsl_vector_add(k->p_collapsed_means, k->p_const);

  int L = k->np;

  gsl_matrix_memcpy(k->P, k->p_collapsed_covars);

  double alpha = 0.01, kappa = 0, beta = 2;
  double lambda = alpha * alpha * (L + kappa) - L;

  /* First step in the unscented transform:
     compute the matrix sqrt( (L+\lambda) * P ) */
  gsl_matrix_scale(k->P, L + lambda);

  gsl_linalg_cholesky_decomp(k->P);
  for (int i=0; i < L; ++i) {
    for (int j=i+1; j < L; ++j) {
      gsl_matrix_set(k->P, i, j, 0);
    }
  }

  /* Generate sigma points by adding and subtracting the columns of
     the above matrix from the augmented mean vector. */
  for (int i=0; i < 2*L+1; ++i) {
    gsl_vector_view col = gsl_matrix_column(k->p_sigma_points, i);
    gsl_vector_memcpy(&col.vector, k->p_collapsed_means);
  }

  gsl_matrix_view add_points = gsl_matrix_submatrix(k->p_sigma_points, 0, 1, L, L);
  gsl_matrix_add(&add_points.matrix, k->P);
  gsl_matrix_view sub_points = gsl_matrix_submatrix(k->p_sigma_points, 0, L+1, L, L);
  gsl_matrix_sub(&sub_points.matrix, k->P);

  /* Pass the sigma points through the observation function, and
     compute their associated weights. */
  for (int i=0; i < 2*L+1; ++i) {
     gsl_vector_view state_col = gsl_matrix_column(k->p_sigma_points, i);
     gsl_vector_view obs_col = gsl_matrix_column(k->p_obs_points, i);

     // provide the observation function with any extra arguments we were passed
     va_list args;
     va_start(args, p_true_obs);
     (*k->p_obs_fn)(&state_col.vector, &obs_col.vector, &args);
     va_end(args);

     gsl_vector_set(k->p_weights, i, 1/(2*(L+lambda)));
  }

  /* Now, compute the weighted mean and covariance of the sigma points. */
  gsl_vector_set(k->p_weights, 0, lambda/(L+lambda));
  weighted_mean(k->p_obs_points, k->p_weights, k->y);
  gsl_vector_set(k->p_weights, 0, lambda/(L+lambda) + (1-alpha*alpha+beta));
  weighted_covar(k->p_obs_points, k->y, k->p_weights, k->S);
  /* add in the covariance of the observation noise */
  matrix_add_to_diagonal(k->S, k->p_obs_noise);

  /* finally, the state/measurement cross-covariance is used to get the Kalman gain */
  weighted_cross_covar(k->p_sigma_points, k->p_collapsed_means, k->p_obs_points, k->y, k->p_weights, k->Ktmp);
  double log_det_S = psdmatrix_inv_logdet(k->S, k->Sinv);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, k->Ktmp, k->Sinv, 0, k->K);

  // find the observation residual
  gsl_vector_sub(k->y, p_true_obs);
  gsl_vector_scale(k->y, -1);

  double pred_mean = gsl_vector_get(k->p_collapsed_means, 0);

  gsl_blas_dgemv(CblasNoTrans, 1, k->K, k->y, 0, k->p_collapsed_mean_update);
  gsl_vector_add(k->p_collapsed_means, k->p_collapsed_mean_update);

  // update the filtering covariance: p_covars = p_covars - K*S*K^T
  // (uses k->Ktmp as temp matrix)
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, k->K, k->S, 0, k->Ktmp); // Ktmp = (S*K^T)^T
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, -1, k->K, k->Ktmp, 1, k->p_collapsed_covars);
  matrix_stabilize_zeros(k->p_collapsed_covars); /* beat down those
						    pesky
						    slightly-nonzero
						    covariances, for
						    numerical
						    stability */

  gsl_vector_sub(k->p_collapsed_means, k->p_const);
  uncollapse_state(k);

  gsl_blas_dgemv(CblasNoTrans, 1, k->Sinv, k->y, 0, k->ytmp);
  double ex;
  gsl_blas_ddot(k->y, k->ytmp, &ex);
  double thisll = 0.5 * k->obs_n*log(2*PI) + .5 * log_det_S + 0.5 * ex;

  /*double r = gsl_vector_get(k->y, 0);
  double s2 = gsl_matrix_get(k->S, 0, 0);

  double t1 = .5*log(s2) + 0.5 * log(2 * PI);
  double t2 = 0.5 * r*r /s2;

  printf("error likelihood %f of residual %f (logstd %f, .5*ex %f)", -1*(t1+t2), r, .5*log(s2), t2);

  printf("thisll %f for residual %f (obs_n %d, lds %f, .5*ex %f)\n", thisll, gsl_vector_get(k->y, 0), k->obs_n, log_det_S, .5*ex);*/

  return thisll;

}

/* sample a zero-mean Gaussian vector with the given variances */
void sample_indep_gaussians(KalmanState_t *k, gsl_vector * p_variances, gsl_vector * p_result) {
  gsl_vector_set_zero(p_result);
  for(int i=0; i < p_result->size; ++i) {
    double std = sqrt(gsl_vector_get(p_variances, i));
    double sample = (std > 1e-10) ? gsl_ran_gaussian(k->r, std) : 0;
    gsl_vector_set(p_result, i, sample);
  }
}


void kalman_sample_forward(KalmanState_t *k, gsl_vector * p_output, ...) {
  // sample a new hidden state
  gsl_blas_dgemv (CblasNoTrans, 1, k->p_transition, k->p_sample_state, 0, k->p_mean_update);
  sample_indep_gaussians(k, k->p_process_noise, k->p_sample_state);
  gsl_vector_add (k->p_sample_state, k->p_mean_update);

  // use p_collapsed_means as a temp vector for the collapsed state
  collapse_vector(k, k->p_sample_state, k->p_collapsed_means);
  //  printf("collapsed means %.3f, %.3f, %.3f\n", gsl_vector_get(k->p_collapsed_means, 0), gsl_vector_get(k->p_collapsed_means, 1), gsl_vector_get(k->p_collapsed_means, 2));

  gsl_vector_add(k->p_collapsed_means, k->p_const);

  // compute the deterministic observation (using optional extra arguments)
  va_list args;
  va_start(args, p_output);
  (*k->p_obs_fn)(k->p_collapsed_means, p_output, &args);
  va_end(args);

  // add observation noise
  sample_indep_gaussians(k, k->p_obs_noise, k->ytmp);
  gsl_vector_add(p_output, k->ytmp);
}




/*
  This function should work, but is untested and unproven since we're
  currently using a nonlinear observation model.

  (and should be updated not to alloc its own memory; use temp variables in KalmanState)
 */
/*
void kalman_update_linear(KalmanState *k,  gsl_vector * p_true_obs, gsl_vector ** y, gsl_matrix ** S) {

  int obs_n = k->p_obs->size1;
  int state_n = k->p_means->size;

  // calculate the predicted observations
  *y = gsl_vector_alloc(obs_n);
  gsl_blas_dgemv (CblasNoTrans, 1, k->p_obs, k->p_means, 0, *y);

  // calculate the predicted observation covariance
  gsl_matrix * PHt = gsl_matrix_alloc(state_n, obs_n);
  gsl_matrix * S = gsl_matrix_alloc(obs_n, obs_n);
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, k->p_covars, k->p_obs, 0, PHt);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, k->p_obs, PHt, 0, *S);
  matrix_add_to_diagonal(*S, k->p_obs_noise);

  // and the Kalman gain

  gsl_blas_dtrsm (CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, 1, *S, *PHt);
  gsl_matrix * K = PHt;

  // find the observation residual
  gsl_vector_sub(*y, p_true_obs);
  gsl_vector_scale(*y, -1);

  gsl_vector *mean_update = gsl_vector_alloc(state_n);
  gsl_blas_dgemv(CblasNoTrans, 1, K, *y, 0, mean_update);
  gsl_vector_add(k->p_means, mean_update);

  gsl_matrix * I = gsl_matrix_alloc(state_n,state_n);
  gsl_matrix_set_identity (I);

  gsl_matrix * new_covar = gsl_matrix_alloc(state_n,state_n);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, -1, K, k->p_obs, 1, I);
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, I, k->p_covars, 0, new_covar);
  gsl_matrix_memcpy(p_covars, new_covar);

  gsl_vector_free(mean_update);
  gsl_matrix_free(I);
  gsl_matrix_free(K);
  gsl_matrix_free(new_covar);

  }*/

void kalman_state_init(KalmanState_t *k, int obs_n, int linear_obs, gsl_matrix * p_linear_obs, kalman_obs_fn p_obs_fn) {

  // initialize random number generator
  const gsl_rng_type * T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  k->r = gsl_rng_alloc (T);
  gsl_rng_set(k->r, time(NULL));

  k->n = 0;
  k->np = 0;
  k->obs_n = obs_n;

  k->linear_obs = linear_obs;
  k->p_linear_obs = p_linear_obs;
  k->p_obs_fn = p_obs_fn;

  // allocate the temp matrices that have a known size
  k->y = gsl_vector_alloc(obs_n);
  k->ytmp = gsl_vector_alloc(obs_n);
  k->S = gsl_matrix_alloc(obs_n, obs_n);
  k->Sinv = gsl_matrix_alloc(obs_n, obs_n);

  k->p_obs_noise = gsl_vector_calloc(obs_n);
  for(int i=0; i < k->obs_n; ++i) {
    gsl_vector_set(k->p_obs_noise, i, 0);
  }
  /* the other state matrices have varying sizes depending on the
     current state-space dimension, so they get (re)allocated when
     needed using kalman_add_AR_process and
     kalman_remove_AR_process. */

}

void kalman_state_free(KalmanState_t * k) {

  gsl_rng_free(k->r);

  if (k->p_const != NULL) gsl_vector_free(k->p_const);
  if (k->p_means != NULL) gsl_vector_free(k->p_means);
  if (k->p_covars != NULL) gsl_matrix_free(k->p_covars);

  if (k->p_transition != NULL) gsl_matrix_free(k->p_transition);
  if (k->p_process_noise != NULL) gsl_vector_free(k->p_process_noise);
  if (k->p_sample_state != NULL) gsl_vector_free(k->p_sample_state);
  if (k->p_process_indices != NULL) gsl_vector_free(k->p_process_indices);

  if (k->p_linear_obs != NULL) gsl_matrix_free(k->p_linear_obs);
  if (k->p_obs_noise != NULL) gsl_vector_free(k->p_obs_noise);


  // free all the temp matrices
  if (k->y != NULL) gsl_vector_free(k->y);
  if (k->ytmp != NULL) gsl_vector_free(k->ytmp);
  if (k->S != NULL) gsl_matrix_free(k->S);
  if (k->Sinv != NULL) gsl_matrix_free(k->Sinv);

  if (k->P != NULL) gsl_matrix_free(k->P);
  if (k->p_sigma_points != NULL) gsl_matrix_free(k->p_sigma_points);
  if (k->p_obs_points != NULL) gsl_matrix_free(k->p_obs_points);
  if (k->p_weights != NULL) gsl_vector_free(k->p_weights);
  if (k->p_mean_update != NULL) gsl_vector_free(k->p_mean_update);
  if (k->p_collapsed_mean_update != NULL) gsl_vector_free(k->p_collapsed_mean_update);
  if (k->p_covars_tmp != NULL) gsl_matrix_free(k->p_covars_tmp);

  if (k->p_collapsed_means != NULL) gsl_vector_free(k->p_collapsed_means);
  if (k->p_collapsed_covars != NULL) gsl_matrix_free(k->p_collapsed_covars);

  if (k->K != NULL) gsl_matrix_free(k->K);
  if (k->Ktmp != NULL) gsl_matrix_free(k->Ktmp);

}

void kalman_state_print(KalmanState_t * k) {

  printf("p_means:\n");
  pretty_print_vector(k->p_means,"%.2f ");

  printf("p_covars:\n");
  pretty_print_matrix(k->p_covars,"%.2f ");

  printf("p_transition:\n");
  pretty_print_matrix(k->p_transition,"%.2f ");

  printf("p_process_noise:\n");
  pretty_print_vector(k->p_process_noise,"%.2f ");

  printf("p_sample_state:\n");
  pretty_print_vector(k->p_sample_state,"%.2f ");


}
