#ifndef KALMAN_FILTER
#define KALMAN_FILTER



#include <stdarg.h>
#include "signal_structures.h"

typedef void (*kalman_obs_fn)(const gsl_vector * state, gsl_vector * obs, va_list *args);

typedef struct KalmanState {
  /* KalmanState invariants:
     -- n must equal the dimension of p_means, p_covars, p_transition, and p_process_noise (note that the matrices are all square), if they are non-NULL.
     -- either p_linear_obs or p_obs_fn must be non-NULL, depending on whether linear_obs is set.
     -- p_obs_noise must be defined (non-NULL).
  */

  int n; // number of current state variables
  int np; // number of current AR processes
  int obs_n; // number of current output variables

  gsl_vector * p_const;

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
  gsl_vector * p_process_indices; /* indices of the current AR processes (np) */


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

  gsl_vector * y; // the predicted observation (or obs residual) (obs_n)
  gsl_vector * ytmp; // temp vector (obs_n)
  gsl_matrix * S; // predicted observation covariance (obs_n * obs_n)
  gsl_matrix * Sinv; // inverse of S (obs_n * obs_n)

  gsl_matrix * P; // unscented transform tmp matrix (np x np)
  gsl_matrix * p_sigma_points; // original sigma points (np x 2np+1)
  gsl_matrix * p_obs_points; // sigma point observations (obs_n x 2np+1)
  gsl_vector * p_weights; // weights for unscented transform (2np+1)
  gsl_vector * p_mean_update; // tmp vector for mean update (n)
  gsl_vector * p_collapsed_mean_update; // tmp vector for mean update (np)
  gsl_matrix * p_covars_tmp; //tmp vector for covar update (n x n)

  gsl_vector * p_collapsed_means; // collapsed mean vector (np)
  gsl_matrix * p_collapsed_covars; // collapsed covariance matrix (np x np)

  gsl_matrix * K; // optimal Kalman gain matrix (np x obs_n)
  gsl_matrix * Ktmp; // temp matrix, (np x obs_n)


  int verbose;

  FILE * debug_res_fp;
  FILE * debug_var_fp;
  FILE * debug_gain_fp;
  FILE * debug_state_fp;

} KalmanState_t ;

void kalman_state_init(KalmanState_t *k, int obs_n, int linear_obs, gsl_matrix * p_linear_obs, kalman_obs_fn p_obs_fn, double obs_noise, char * debug_dir);
void kalman_state_free(KalmanState_t * k);

int kalman_add_AR_process(KalmanState_t * k, ARProcess_t * p);
void kalman_remove_AR_process(KalmanState_t * k, int m, int arridx);
void kalman_predict(KalmanState_t * k);
double kalman_nonlinear_update(KalmanState_t *k,  gsl_vector * p_true_obs, ...);

void kalman_sample_forward(KalmanState_t *k, gsl_vector * p_output, ...);

void kalman_state_print(KalmanState_t * k);


void matrix_stabilize(gsl_matrix *m);
void matrix_stabilize_zeros(gsl_matrix *m);
/* void kalman_observation(int n, int k,
		    int n_arrs, ArrivalWaveform_t * active_arrivals,
		    Segment_t * p_segment,
		    gsl_matrix ** pp_obs);

void kalman_update_linear(KalmanState_t *k,  gsl_vector * p_true_obs, gsl_vector ** y, gsl_matrix ** S);
*/


#endif
