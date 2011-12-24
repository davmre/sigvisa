typedef struct ArrivalSNRPrior_t
{
  int step_size;
  int num_bins;
  int num_phases;
  
  double * true_prob;                        /* num_phases * num_bins */
  double * false_prob;                       /* num_bins */

} ArrivalSNRPrior_t;

void ArrivalSNRPrior_Init_Params(ArrivalSNRPrior_t * prior,
                                 const char * filename);

double ArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                               int siteid, int phaseid, double snr);

double FalseArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                                    int siteid, double snr);

void ArrivalSNRPrior_UnInit(ArrivalSNRPrior_t * prior);
