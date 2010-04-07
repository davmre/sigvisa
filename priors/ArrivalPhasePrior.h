typedef struct ArrivalPhasePrior_t
{
  int numphases;
  
  double * phase2phase;              /* numphases x numphases array */
  double * logphase2phase;           /* numphases x numphases array */
  
  double * falsephase;
  double * logfalsephase;

} ArrivalPhasePrior_t;

void ArrivalPhasePrior_Init_Params(ArrivalPhasePrior_t * prior,
                                   const char * filename);

double ArrivalPhasePrior_LogProb(const ArrivalPhasePrior_t * prior,
                                 int arrphaseid, int true_phaseid);

double FalseArrivalPhasePrior_LogProb(const ArrivalPhasePrior_t * prior,
                                      int arr_phaseid);

void ArrivalPhasePrior_UnInit(ArrivalPhasePrior_t * prior);
