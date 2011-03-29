typedef struct ArrivalTimePrior_t
{
  int numsites;
  int numphases;

  double * locs;                             /* numsites x numphases */
  double * scales;                           /* numsites x numphases */
  double * minvals;                          /* numsites x numphases */
  double * maxvals;                          /* numsites x numphases */
} ArrivalTimePrior_t;

void ArrivalTimePrior_Init_Params(ArrivalTimePrior_t * prior,
                                  const char * filename);

double ArrivalTimePrior_LogProb(const ArrivalTimePrior_t * prior,
                                double arrtime, double pred_arrtime,
                                double det_deltime, int siteid, int phaseid);

double ArrivalTimePrior_MeanResidual(const ArrivalTimePrior_t * prior,
                                     int siteid, int phaseid);

void ArrivalTimePrior_UnInit(ArrivalTimePrior_t * prior);
