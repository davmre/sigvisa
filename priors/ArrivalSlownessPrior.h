typedef struct ArrivalSlownessPrior_t
{
  int numsites;
  int numphases;

  double * locs;                             /* numsites x numphases */
  double * scales;                           /* numsites x numphases */
} ArrivalSlownessPrior_t;

void ArrivalSlownessPrior_Init_Params(ArrivalSlownessPrior_t * prior,
                                      const char * filename);

double ArrivalSlownessPrior_LogProb(const ArrivalSlownessPrior_t * prior,
                                    double arrslo, double pred_arrslo,
                                    double det_delslo, int siteid, 
                                    int phaseid);

void ArrivalSlownessPrior_UnInit(ArrivalSlownessPrior_t * prior);
