#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../sigvisa.h"

void ArrivalTimeJointPrior_Init_Params(ArrivalTimeJointPrior_t * prior,
                                  const char * filename)
{
  ArrivalTimePrior_Init_Params(&prior->single_prior, filename);   
}

double ArrivalTimeJointPrior_LogProb(const ArrivalTimeJointPrior_t * prior,
				     EarthModel_t * p_earth,
				     int numevents,
				     Event_t * events,
				     PyArrayObject * arrtimes)
{
  
  double log_prob = 0;

  for (int siteid=0; siteid < p_earth->numsites; ++siteid) {
    for (int i=0; i<numevents; i++) {

      Event_t * event = events+i;

      int phaseid = 0;
      // TODO: loop over phases
      // for (int phaseid=0; phaseid < p_earth->numphases; ++phaseid) {

      double pred_arrtime = EarthModel_ArrivalTime(p_earth, event->evlon, event->evlat, event->evdepth, event->evtime, phaseid, siteid);

	double arrtime = ARRAY3(arrtimes, siteid, i, phaseid);

	if (arrtime == -1 || pred_arrtime == -1) {
	  continue;
	}

	double ll = ArrivalTimePrior_LogProb(&(prior->single_prior), 
					     arrtime, 
					     pred_arrtime, 
					     0, 
					     siteid, 
					     phaseid);
	log_prob += ll;

	// }
    }
  }

  return log_prob;

}

double ArrivalTimeJointPrior_Point(const ArrivalTimeJointPrior_t * prior,
				     EarthModel_t * p_earth,
				     int numevents,
				     Event_t * events,
				     PyArrayObject * arrtimes) {

  double pred_arrtime = EarthModel_ArrivalTime(p_earth, event->evlon, event->evlat, event->evdepth, event->evtime, phaseid, siteid);

}

void ArrivalTimeJointPrior_UnInit(ArrivalTimeJointPrior_t * prior)
{
  ArrivalTimePrior_UnInit(&prior->single_prior);
}

