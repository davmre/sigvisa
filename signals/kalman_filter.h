#ifndef KALMAN_FILTER
#define KALMAN_FILTER


#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

typedef void (*kalman_obs_fn)(const gsl_vector * state, gsl_vector * obs);
typdef struct KalmanState {
  /* KalmanState invariants:
     -- n must equal the dimension of p_means, p_covars, p_transition, and p_process_noise (note that the matrices are all square), if they are non-NULL.
     -- either p_linear_obs or p_obs_fn must be non-NULL, depending on whether linear_obs is set. 
     -- p_obs_noise must be defined (non-NULL).
  */
  
  int n; // number of current state variables

  gsl_vector * p_means;
  gsl_matrix * p_covars;

  gsl_matrix * p_transition;
  gsl_vector * p_process_noise; /* gives the variance of the process
				   noise for each of the indep AR
				   processes (i.e., this is the
				   diagonal of the overall process
				   noise covariance matrix). */

  int linear_obs; /* determines whether observations are a linear
		     transformation, given by the matrix p_linear_obs,
		     or a nonlinear transformation given by
		     p_obs_fn. The nonlinear case is handled through
		     an unscented Kalman filter. */
  gsl_matrix * p_linear_obs;
  kalman_obs_fn p_obs_fn;
  gsl_vector * p_obs_noise;

} KalmanState_t ;

void kalman_add_AR_process(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n);
void kalman_remove_AR_process(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n, int arridx);

/* void kalman_observation(int n, int k, 
		    int n_arrs, ArrivalWaveform_t * active_arrivals, 
		    Segment_t * p_segment,
		    gsl_matrix ** pp_obs); */


void kalman_predict(gsl_vector * p_means, gsl_matrix * p_covars, gsl_matrix * p_transition, double noise_sigma2, int n);

void kalman_nonlinear_update(KalmanState *k,  gsl_vector * p_true_obs, gsl_vector ** y, gsl_matrix ** S);

/* void kalman_observation(int n, int k, 
		    int n_arrs, ArrivalWaveform_t * active_arrivals, 
		    Segment_t * p_segment,
		    gsl_matrix ** pp_obs); 

void kalman_update_linear(KalmanState *k,  gsl_vector * p_true_obs, gsl_vector ** y, gsl_matrix ** S);
*/


#endif
