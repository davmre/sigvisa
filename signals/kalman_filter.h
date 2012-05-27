#ifndef KALMAN_FILTER
#define KALMAN_FILTER


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <stdarg.h>

typedef void (*kalman_obs_fn)(const gsl_vector * state, gsl_vector * obs, va_list * args);
typdef struct KalmanState {
  /* KalmanState invariants:
     -- n must equal the dimension of p_means, p_covars, p_transition, and p_process_noise (note that the matrices are all square), if they are non-NULL.
     -- either p_linear_obs or p_obs_fn must be non-NULL, depending on whether linear_obs is set. 
     -- p_obs_noise must be defined (non-NULL).
  */
  
  int n; // number of current state variables
  int obs_n; // number of current output variables

  gsl_vector * p_means;
  gsl_matrix * p_covars;

  gsl_matrix * p_transition;
  gsl_vector * p_process_noise; /* gives the variance of the process
				   noise for each of the indep AR
				   processes (i.e., this is the
				   diagonal of the overall process
				   noise covariance matrix). */

  gsl_vector * p_sample_state; /* The true hidden state of the
				    process; used when sampling from
				    the model. (n) */


  int linear_obs; /* determines whether observations are a linear
		     transformation, given by the matrix p_linear_obs,
		     or a nonlinear transformation given by
		     p_obs_fn. The nonlinear case is handled through
		     an unscented Kalman filter. */
  gsl_matrix * p_linear_obs;
  kalman_obs_fn p_obs_fn;
  gsl_vector * p_obs_noise;

  gsl_rng *r;

  /* begin temp variables: these are not really part of the state, but are used repeatedly (mostly when updating) and so we keep them around to avoid allocating/deallocating */ 

  gsl_vector * y; // the predicted observation (or obs residual)
  gsl_vector * ytmp; // temp vector, same size as y
  gsl_matrix * S; // predicted observation covariance
  gsl_matrix * Sinv; // inverse of S

  gsl_matrix * P; // unscented transform tmp matrix (n x n)
  gsl_matrix * p_sigma_points; // original sigma points (n x 2n+1)
  gsl_matrix * p_obs_points; // sigma point observations (obs_n x 2n+1)
  gsl_vector * p_weights; // weights for unscented transform (2n+1)
  gsl_vector * p_mean_update; // tmp vector for mean update (n)

  gsl_matrix * K; // optimal Kalman gain matrix
  gsl_matrix * Ktmp; // temp matrix, same size as K

} KalmanState_t ;

void kalman_state_init(KalmanState *k, int obs_n, int linear_obs, gsl_matrix * p_linear_obs, kalman_obs_fn p_obs_fn);
void kalman_state_free(KalmanState * k);

void kalman_add_AR_process(KalmanState_t * k, int m, double * ar_coeffs, double noise_sigma2);
void kalman_remove_AR_process(KalmanState_t * k, int m, int arridx);
void kalman_predict(KalmanState * k);
void kalman_nonlinear_update(KalmanState *k,  gsl_vector * p_true_obs, gsl_vector ** y, gsl_matrix ** S, ...);

void kalman_sample_forward(KalmanState *k, gsl_vector * p_output, ...);

/* void kalman_observation(int n, int k, 
		    int n_arrs, ArrivalWaveform_t * active_arrivals, 
		    Segment_t * p_segment,
		    gsl_matrix ** pp_obs); 

void kalman_update_linear(KalmanState *k,  gsl_vector * p_true_obs, gsl_vector ** y, gsl_matrix ** S);
*/


#endif
