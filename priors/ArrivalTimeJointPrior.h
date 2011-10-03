typedef struct ArrivalTimeJointPrior_t
{
  ArrivalTimePrior single_prior;
} ArrivalTimeJointPrior_t;

void ArrivalTimeJointPrior_Init_Params(ArrivalTimePrior_t * prior,
                                  const char * filename);

double ArrivalTimeJointPrior_LogProb(const ArrivalTimePrior_t * prior,
                                double arrtime, double pred_arrtime,
                                double det_deltime, int siteid, int phaseid);

void ArrivalTimeJointPrior_UnInit(ArrivalTimePrior_t * prior);
