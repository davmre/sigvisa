#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../netvisa.h"
#include "ArrivalTimePrior.h"
#include 

void ArrivalTimeJointPrior_Init_Params(ArrivalTimeJointPrior_t * prior,
                                  const char * filename)
{
    ArrivalTimePrior_Init_Params(&prior->single_prior, filename)    
}

double ArrivalTimeJointPrior_LogProb(const ArrivalTimeJointPrior_t * prior,
				     EarthModel_t * p_earth,
				     int numevents,
				     Event_t * events,
				     double *** arrtimes)
{
  
  double log_prob = 0;

  for (siteid=0; siteid < p_earth->numsites; ++siteid) {
    for (i=0; i<numevents; i++) {

      Event_t * event = events+i;

      for (phaseid=0; phaseid < p_earth_numphases; ++phaseid) {

	pred_arrtime = EarthModel_ArrivalTime(p_earth, event->evlon, event->evlat, event->evdepth, event->evtime, phaseid, siteid);
    
	log_prob += ArrivalTimePrior_LogProb(&prior->single_prior, arrtimes[i][siteid][phaseid], pred_arrtime, 0, siteid, phaseid);
      }
    }
  }

  return log_prob;

}

void ArrivalTimeJointPrior_UnInit(ArrivalTimePrior_t * prior)
{
  ArrivalTimePrior_UnInit(&prior->single_prior);
}

