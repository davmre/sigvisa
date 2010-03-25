#include "../netvisa.h"

void EarthModel_Init(EarthModel_t * p_earth, char * dirname,
                     int numphases, char * phasenames, int * timdefphase,
                     int numsites, Site_t * sites)
{
}

void EarthModel_UnInit(EarthModel_t * p_earth)
{
}

int EarthModel_InRange(EarthModel_t * p_earth, double * evloc, int phaseid,
                       int siteid)
{
  return 0;
}

double EarthModel_ArrivalTime(EarthModel_t * p_earth, double * evloc,
                              double evtime, int phaseid, int statin)
{
  return evtime;
}

double EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, double * evloc,
                                 int phaseid, int siteid)
{
  return 0;
}


double EarthModel_ArrivalSlowness(EarthModel_t * p_earth, double * evloc,
                               int phaseid, int siteid)
{
  return 0;
}


int EarthModel_TimeDefPhase(EarthModel_t * p_earth, int phaseid)
{
  return 0;
}
