#include "ArrivalTimePrior.h"


typedef struct ArrivalTimeJointPrior_t
{
  ArrivalTimePrior_t single_prior;
} ArrivalTimeJointPrior_t;

void ArrivalTimeJointPrior_Init_Params(ArrivalTimeJointPrior_t * prior,
                                  const char * filename);

double ArrivalTimeJointPrior_LogProb(const ArrivalTimeJointPrior_t * prior,
				     EarthModel_t * p_earth,
				     int numevents,
				     Event_t * events,
				     PyArrayObject * arrtimes);

void ArrivalTimeJointPrior_UnInit(ArrivalTimeJointPrior_t * prior);
