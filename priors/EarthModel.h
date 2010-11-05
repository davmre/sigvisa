#define EARTH_PHASE_P       0
#define EARTH_PHASE_Pn      1
#define EARTH_PHASE_PKP     2
#define EARTH_PHASE_Sn      3
#define EARTH_PHASE_S       4
#define EARTH_PHASE_PKPbc   5
#define EARTH_PHASE_PcP     6
#define EARTH_PHASE_pP      7
#define EARTH_PHASE_Lg      8
#define EARTH_PHASE_PKPab   9
#define EARTH_PHASE_ScP     10
#define EARTH_PHASE_PKKPbc  11
#define EARTH_PHASE_Pg      12
#define EARTH_PHASE_Rg      13
#define EARTH_PHASE_tx      14
#define EARTH_PHASE_Sx      15
#define EARTH_PHASE_Px      16
#define EARTH_PHASE_N       17


typedef struct EarthPhaseModel_t
{
  int numdepth;
  int numdist;
  
  double * p_depths;
  double * p_dists;
  
  double * p_samples;                        /* numdepth x numdist */
  
  double surf_vel;                           /* surface velocity */
  
} EarthPhaseModel_t;

#define EarthPhaseModel_GetSample(p_phase, depthi, disti) \
((p_phase)->p_samples[(depthi) * (p_phase)->numdist + (disti)])

typedef struct EarthModel_t
{
  PyObject_HEAD

  int numsites;
  struct Site_t * p_sites;

  int numtimedefphases;
  
  int numphases;

  char ** p_phasenames;
  int * p_phase_time_def;

  EarthPhaseModel_t * p_phases;
  
} EarthModel_t;

#define EarthModel_IsTimeDefPhase(p_earth, phaseid)\
 ((p_earth)->p_phase_time_def[(phaseid)])

#define EarthModel_NumPhases(p_earth) ((p_earth)->numphases)

#define EarthModel_NumTimeDefPhases(p_earth) ((p_earth)->numtimedefphases)

#define EarthModel_NumSites(p_earth) ((p_earth)->numsites)

int py_EarthModel_Init(EarthModel_t * p_earth, PyObject * args);

void py_EarthModel_UnInit(EarthModel_t * p_earth);

PyObject * py_EarthModel_InRange(EarthModel_t * p_earth, PyObject * args);

int EarthModel_InRange(EarthModel_t * p_earth, double lon, double lat,
                       double depth, int phaseid, int siteid);

double EarthModel_Delta(EarthModel_t * p_earth, double lon, double lat,
                        int siteid);

double simple_distance_deg(double lon1, double lat1, double lon2,
                           double lat2);

PyObject * py_EarthModel_Delta(EarthModel_t * p_earth, PyObject * args);

PyObject * py_EarthModel_ArrivalTime(EarthModel_t * p_earth, PyObject * args);

double EarthModel_ArrivalTime(EarthModel_t * p_earth, double lon, double lat,
                              double depth, double evtime, 
                              int phaseid, int siteid);

PyObject * py_EarthModel_ArrivalTime_Coord(EarthModel_t * p_earth,
                                           PyObject * args);

double EarthModel_ArrivalTime_Coord(EarthModel_t * p_earth, double lon, 
                                    double lat, double depth, double evtime, 
                                    int phaseid, double sitelon,
                                    double sitelat, double siteelev);

PyObject * py_EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, 
                                        PyObject * args);

double EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, double lon,
                                 double lat, int siteid);

PyObject * py_EarthModel_ArrivalSlowness(EarthModel_t * p_earth, 
                                         PyObject * args);

double EarthModel_ArrivalSlowness(EarthModel_t * p_earth, double lon,
                                  double lat, double depth,
                                  int phaseid, int siteid);

PyObject * py_EarthModel_IsTimeDefPhase(EarthModel_t * p_earth, 
                                        PyObject * args);

PyObject * py_EarthModel_NumTimeDefPhases(EarthModel_t * p_earth, 
                                          PyObject * args);

PyObject * py_EarthModel_NumPhases(EarthModel_t * p_earth, 
                                   PyObject * args);

PyObject * py_EarthModel_NumSites(EarthModel_t * p_earth, 
                                  PyObject * args);

PyObject * py_EarthModel_DiffAzimuth(EarthModel_t * p_earth, 
                                     PyObject * args);

double EarthModel_DiffAzimuth(double azi1, double azi2);

PyObject * py_EarthModel_PhaseName(EarthModel_t * p_earth, 
                                   PyObject * args);

PyObject * py_EarthModel_MaxTravelTime(EarthModel_t * p_earth, 
                                       PyObject * args);

int invert_detection(const EarthModel_t * p_earth, const Detection_t * p_det,
                     Event_t * p_event, int perturb);

PyObject * py_EarthModel_PhaseRange(EarthModel_t * p_earth, PyObject * args);
