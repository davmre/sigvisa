#ifndef SIGNAL_MODEL_UTIL
#define SIGNAL_MODEL_UTIL

typedef struct ARProcess {
  
  double mean;
  int order;
  double * params;
  double noise_sigma2;

} ARProcess_t;


typedef struct ARWLists_t {

  struct ArrivalWaveform * st_head;
  struct ArrivalWaveform * et_head;
  
  struct ArrivalWaveform * st_ptr;
  struct ArrivalWaveform * et_ptr;
  struct ArrivalWaveform * active_arrivals;

} ARWLists_t;

typedef struct ArrivalWaveform {
  double start_time;
  long idx;
  double end_time;
  long len;

  Trace * p_abstract_trace;

  struct ArrivalWaveform * next_start;
  struct ArrivalWaveform * next_end;
  struct ArrivalWaveform * next_active;
  int active_id;

  double projection_coeffs[NUM_CHANS];
  ARProcess_t ar_process;

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
