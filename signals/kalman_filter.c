#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
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
void kalman_add_AR_process(KalmanState_t * k, int m, double * ar_coeffs, double noise_sigma2) {

  expand_vector(&k->p_means, (k->n)+m);
  expand_vector(&k->p_process_noise, (k->n)+m);
  expand_matrix(&k->p_covars, (k->n)+m, (k->n)+m, TRUE);
  expand_matrix(&k->p_transition, (k->n)+m, (k->n)+m, FALSE);

  gsl_vector_set(k->p_process_noise, (k->n)+m-1, noise_sigma2);

  /* fill in the new entries for the transition matrix */
  gsl_matrix_set(k->p_transition, (k->n)+m-1, (k->n), ar_coeffs[0]);
  for (int i=1; i < m; ++i) {
    gsl_matrix_set(k->p_transition, (k->n)+i-1, (k->n)+i, 1);
    gsl_matrix_set(k->p_transition, (k->n)+m-1, (k->n)+i, ar_coeffs[i]);
  }

  k->n += m;

}

/* Removes the order-m AR process with specified index arridx from the mean vector and covariance matrix. */
void kalman_remove_AR_process(KalmanState_t * k, int m, int arridx) {

  remove_vector_slice(&k->p_means, arridx, m);
  remove_vector_slice(&k->p_process_noise, arridx, m);
  remove_matrix_slice(&k->p_covars, arridx, m);
  remove_matrix_slice(&k->p_transition, arridx, m);

  k->n -= m;

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
void kalman_predict(KalmanState * k) {

  /* propagate the means through the transition model */
  gsl_vector * tmp = gsl_vector_alloc(p_means->size);
  gsl_blas_dgemv (CblasNoTrans, 1, p_transition, p_means, 0, tmp);
  gsl_vector_memcpy(p_means, tmp);
  gsl_vector_free(tmp);

  /* propagate the covariances through the transition model */
  gsl_matrix * mtmp = gsl_matrix_alloc(p_covars->size1, p_covars->size2);
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, p_covars, p_transition, 0, mtmp);  
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, p_transition, mtmp, 0, p_covars);

  /* add the process noise variance */
  gsl_vector_view covar_diag = gsl_matrix_diagonal(p_covars);
  gsl_vector_add(&covar_diag.vector, k->p_process_noise);
}


/* Uses the unscented transform to compute the measurement mean and covariance; then uses these to update the Kalman filtering distribution. */
void kalman_nonlinear_update(KalmanState *k,  gsl_vector * p_true_obs, gsl_vector ** y, gsl_matrix ** S) {

  int L = k->p_mean->size;
  gsl_matrix * P = gsl_matrix_alloc(L, L);
  gsl_matrix_memcpy(P, k->p_covar);

  double alpha = 0.001, kappa = 0, beta = 2;
  double lambda = alpha * alpha * (L + kappa) - L;

  /* First step in the unscented transform: 
     compute the matrix sqrt( (L+\lambda) * P ) */
  gsl_matrix_scale(P, L + lambda);
  gsl_linalg_cholesky_decomp(P);
  for (i=0; i < L; ++i) {
    for (j=i+1; j < L; ++j) {
      gsl_matrix_set(P, i, j, 0);
    }
  }

  /* Generate sigma points by adding and subtracting the columns of
     the above matrix from the augmented mean vector. */
  gsl_matrix * p_sigma_points = gsl_matrix_alloc(L, 2L+1);
  for (int i=0; i < 2L+1; ++i) {
    gsl_vector_view col = gsl_matrix_column(p_sigma_points, i);
    gsl_vector_memcpy(&col.vector, k->p_mean);
  }
  gsl_matrix_view add_points = gsl_matrix_submatrix(p_sigma_points, 1, 1, L, L);
  gsl_matrix_add(&add_points.matrix, P);
  gsl_matrix_view sub_points = gsl_matrix_submatrix(p_sigma_points, L+1, L+1, L, L);
  gsl_matrix_sub(&sub_points.matrix, P);

  /* Pass the sigma points through the observation function, and
     compute their associated weights. */
  gsl_matrix * p_obs_points = gsl_matrix_alloc(L, 2L+1);
  gsl_vector * p_weights = gsl_vector_alloc(2L+1);
  for (int i=0; i < 2L+1; ++i) {
     gsl_vector_view state_col = gsl_matrix_column(p_sigma_points, i);
     gsl_vector_view obs_col = gsl_matrix_column(p_sigma_points, i);
     (*k->p_obs_fn)(&state_col.vector, &obs_col.vector);
     gsl_vector_set(p_weights, i, 1/(2*(L+lambda)));
  }

  /* Now, compute the weighted mean and covariance of the sigma points. */
  gsl_vector_set(p_weights, 0, lambda/(L+lambda));
  *y = weighted_mean(p_obs_points, p_weights);
  gsl_vector_set(p_weights, 0, lambda/(L+lambda) + (1-alpha*alpha+beta));
  *S = weighted_covar(p_obs_points, *y, p_weights);
  /* add in the covariance of the observation noise */
  matrix_add_to_diagonal(*S, k->p_obs_noise);

  /* finally, the state/measurement cross-covariance is used to get the Kalman gain */
  gsl_matrix * K = weighted_cross_covar(p_sigma_points, k->p_mean, p_obs_points, *y, p_weights);
  gsl_blas_dtrsm (CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, 1, *S, *K);

  // find the observation residual
  gsl_vector_sub(*y, p_true_obs);
  gsl_vector_scale(*y, -1); 

  gsl_vector *mean_update = gsl_vector_alloc(state_n);
  gsl_blas_dgemv(CblasNoTrans, 1, K, *y, 0, mean_update);
  gsl_vector_add(k->p_means, mean_update);

  // update the filtering covariance: p_covars = p_covars - K*S*K^(-1)
  gsl_matrix * tmp = gsl_matrix_alloc(state_n,state_n);
  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1, *S, K, 0, tmp); // tmp = S*K^(-1)
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, -1, K, tmp, 1, k->p_covars); 

  gsl_matrix_free(P);
  gsl_matrix_free(p_sigma_points);
  gsl_matrix_free(p_obs_points);
  gsl_vector_free(p_weights_m);
  gsl_vector_free(p_weights_c);
  gsl_matrix_free(K);
  gsl_vector_free(mean_update);
  gsl_matrix_free(tmp);
}


/*
  This function should work, but is untested and unproven since we're
  currently using a nonlinear observation model.
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

