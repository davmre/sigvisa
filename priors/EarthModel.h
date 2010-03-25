typedef struct Site_t
{
  double sitelon;
  double sitelat;
  double siteelev;
  int    siteisarr;

  double start_time;
  int * up_time;
} Site_t;

typedef struct EarthPhaseModel_t
{
  double mindist;
  double maxdist;
  
} EarthPhaseModel_t;

typedef struct EarthModel_t
{
  EarthPhaseModel_t * phasemodel;
} EarthModel_t;

void EarthModel_Init(EarthModel_t * p_earth, char * dirname,
                     int numphases, char * phasenames, int * timdefphase,
                     int numsites, Site_t * sites);

void EarthModel_UnInit(EarthModel_t * p_earth);

int EarthModel_InRange(EarthModel_t * p_earth, double * evloc, int phaseid,
                       int siteid);

double EarthModel_ArrivalTime(EarthModel_t * p_earth, double * evloc,
                              double evtime, int phaseid, int siteid);

double EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, double * evloc,
                                 int phaseid, int siteid);

double EarthModel_ArrivalSlowness(EarthModel_t * p_earth, double * evloc,
                                  int phaseid, int siteid);

int EarthModel_TimeDefPhase(EarthModel_t * p_earth, int phaseid);

