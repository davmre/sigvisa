typedef struct ArrivalSNRPrior_t
{
  int numsites;
  
  double * true_mean;
  double * true_sigma;
  double * false_mean;
  double * false_sigma;

} ArrivalSNRPrior_t;

void ArrivalSNRPrior_Init_Params(ArrivalSNRPrior_t * prior,
                                 const char * filename);

double ArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                               int siteid, double snr);

double FalseArrivalSNRPrior_LogProb(const ArrivalSNRPrior_t * prior,
                                    int siteid, double snr);

void ArrivalSNRPrior_UnInit(ArrivalSNRPrior_t * prior);
