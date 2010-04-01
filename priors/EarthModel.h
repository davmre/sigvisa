typedef struct EarthPhaseModel_t
{
  int numdepth;
  int numdist;
  
  double * p_depths;
  double * p_dists;
  
  double * p_samples;                        /* numdepth x numdist */
  
  double mindist;
  double maxdist;

  double surf_vel;                           /* surface velocity */
  
} EarthPhaseModel_t;

#define EarthPhaseModel_Sample(p_phase, depthi, disti) \
((p_phase)->p_samples[(depthi) * (p_phase)->numdist + (disti)])

typedef struct EarthModel_t
{
  PyObject_HEAD

  int numsites;
  struct Site_t * p_sites;

  int numphases;

  char ** p_phasenames;
  int * p_phase_time_def;

  EarthPhaseModel_t * p_phases;
  
} EarthModel_t;

int py_EarthModel_Init(EarthModel_t * p_earth, PyObject * args);

void py_EarthModel_UnInit(EarthModel_t * p_earth);

PyObject * py_EarthModel_InRange(EarthModel_t * p_earth, PyObject * args);

int EarthModel_InRange(EarthModel_t * p_earth, double lon, double lat,
                       double depth, int phaseid, int siteid);

PyObject * py_EarthModel_Delta(EarthModel_t * p_earth, PyObject * args);

PyObject * py_EarthModel_ArrivalTime(EarthModel_t * p_earth, PyObject * args);

double EarthModel_ArrivalTime(EarthModel_t * p_earth, double lon, double lat,
                              double depth, double evtime, 
                              int phaseid, int siteid);

PyObject * py_EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, 
                                        PyObject * args);

double EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, double lon,
                                 double lat, int siteid, double *p_esaz,
                                 double *p_seaz);

double EarthModel_ArrivalSlowness(EarthModel_t * p_earth, double * evloc,
                                  int phaseid, int siteid);

int EarthModel_TimeDefPhase(EarthModel_t * p_earth, int phaseid);

