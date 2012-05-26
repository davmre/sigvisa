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

#endif
