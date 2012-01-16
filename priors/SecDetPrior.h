typedef struct SecDetPrior_t
{
  int num_logamp_bins;
  double low_logamp;
  double step_logamp;
  double * p_detprob;                        /* X num_logamp_bins */
  double time_shape;
  double min_delay;
  double time_scale;
  double max_delay;
  double az_loc;
  double az_scale;
  double slo_loc;
  double slo_scale;
  double logamp_loc;
  double logamp_scale;
  double snr_pos_prob;
  double snr_pos_lambda;
  double snr_neg_lambda;
  int numphases;
  double * phaseprob;                        /* numphases */
  
} SecDetPrior_t;

void SecDetPrior_Init_Params(SecDetPrior_t * prior, const char * fname);

void SecDetPrior_UnInit(SecDetPrior_t * prior);

int SecDetPrior_Time_Possible(const SecDetPrior_t * prior, double sec_time,
                              double prim_time);

double SecDetPrior_Det_LogProb(const SecDetPrior_t * prior, int is_det,
                               double prim_amp);

double SecDetPrior_Time_LogProb(const SecDetPrior_t * prior, double sec_time,
                                double prim_time);

double SecDetPrior_Slow_LogProb(const SecDetPrior_t * prior, double sec_slow,
                                double prim_slow);

double SecDetPrior_Azimuth_LogProb(const SecDetPrior_t * prior, double sec_az,
                                   double prim_az);

double SecDetPrior_Phase_LogProb(const SecDetPrior_t * prior, int sec_phaseid);

double SecDetPrior_Amp_LogProb(const SecDetPrior_t * prior, double sec_amp,
                               double prim_amp);

double SecDetPrior_SNR_LogProb(const SecDetPrior_t * prior, double sec_snr,
                               double prim_snr);
