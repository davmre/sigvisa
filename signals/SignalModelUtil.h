#ifndef SIGNAL_MODEL_UTIL
#define SIGNAL_MODEL_UTIL

typedef struct ArrivalWaveform {
  double start_time;
  long idx;
  double end_time;
  long len;
  double * p_envelope;

  struct ArrivalWaveform * next_start;
  struct ArrivalWaveform * next_end;
  struct ArrivalWaveform * next_active;
  int active_id;

  double bhe_coeff;
  double bhn_coeff;
  double bhz_coeff;

  // used for sampling
  double * last_perturbs;

} ArrivalWaveform_t;


ArrivalWaveform_t * insert_st(ArrivalWaveform_t * p_head, 
			      ArrivalWaveform_t * p_arr);
ArrivalWaveform_t * insert_et(ArrivalWaveform_t * p_head, 
			      ArrivalWaveform_t * p_arr);
ArrivalWaveform_t * remove_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr);
ArrivalWaveform_t * append_active(ArrivalWaveform_t * p_head, 
				  ArrivalWaveform_t * p_arr);

int AR_add_arrival(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n);
void AR_remove_arrival(gsl_vector ** pp_means, gsl_matrix ** pp_covars, int n, int arridx);
void AR_transitions(int n, int k, double * ar_coeffs, 
		    int n_arrs, 
		    gsl_matrix ** pp_transition);
void AR_observation(int n, int k, 
		    int n_arrs, ArrivalWaveform_t * active_arrivals, 
		    Segment_t * p_segment,
		    gsl_matrix ** pp_obs);
void AR_predict(gsl_vector * p_means, gsl_matrix * p_covars, gsl_matrix * p_transition, double noise_sigma2, int n);
double matrix_inv_det(gsl_matrix * A, gsl_matrix * invA);
void AR_update(gsl_vector * p_means, gsl_matrix * p_covars, 
	       gsl_matrix * p_obs, gsl_vector * obs_perturb, 
	       gsl_matrix * obs_covar, int n, int k, 
	       gsl_vector * y, gsl_matrix * S);

#endif
