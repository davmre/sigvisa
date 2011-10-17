typedef struct ArrivalAzimuthPrior_t
{
  int numsites;
  int numphases;

  double * locs;                             /* numsites x numphases */
  double * scales;                           /* numsites x numphases */
} ArrivalAzimuthPrior_t;

void ArrivalAzimuthPrior_Init_Params(ArrivalAzimuthPrior_t * prior,
                                  const char * filename);

double ArrivalAzimuthPrior_LogProb(const ArrivalAzimuthPrior_t * prior,
                                   double arraz, double pred_arraz,
                                   double det_delaz, int siteid, int phaseid);

void ArrivalAzimuthPrior_UnInit(ArrivalAzimuthPrior_t * prior);
