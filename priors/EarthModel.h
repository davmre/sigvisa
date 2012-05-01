/*
 * Copyright (c) 2012, Bayesian Logic, Inc.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Bayesian Logic, Inc. nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 * Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
 */
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

#define MAX_PHASE_DDRANGES  5 /* maximum number of DDRanges per phase */

typedef struct DDRange
{
  char phase[PHASENAME_MAXLEN+1];
  double mindist;
  double maxdist;
  double mindepth;
  double maxdepth;
  double minmag;
} DDRange;

typedef struct EarthPhaseModel_t
{
  int numdepth;
  int numdist;
  
  double * p_depths;
  double * p_dists;
  
  double * p_samples;                        /* numdepth x numdist */
  
  double surf_vel;                           /* surface velocity */

  int     numddrange;                        /* number of ddranges for phase */
  DDRange p_ddranges[MAX_PHASE_DDRANGES];
  
} EarthPhaseModel_t;

#define EarthPhaseModel_GetSample(p_phase, depthi, disti) \
((p_phase)->p_samples[(depthi) * (p_phase)->numdist + (disti)])

typedef struct QFactorModel_t
{
  int numdepth;
  int numdist;
  
  double * p_depths;                         /* numdepth */
  double * p_dists;                          /* numdist */

  double * p_samples;                        /* numdepth x numdist */
  
} QFactorModel_t;

#define EarthQFactorModel_GetSample(p_qfvc, depthi, disti) \
((p_qfvc)->p_samples[(depthi) * (p_qfvc)->numdist + (disti)])

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

  QFactorModel_t qfvc;

  int enforce_ddrange;
  
  ArrivalAmplitudePrior_t arr_amp_prior;
  int arr_amp_prior_loaded;                  /* boolean */
  
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

PyObject * py_EarthModel_TravelTime(EarthModel_t * p_earth,
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

PyObject * py_EarthModel_QFVC(EarthModel_t * p_earth, PyObject * args);

double EarthModel_QFVC(EarthModel_t * p_earth, double depth, double dist);

PyObject * py_EarthModel_LoadAmpModel(EarthModel_t * p_earth, PyObject * args);

PyObject * py_EarthModel_LogArrivalAmp(EarthModel_t * p_earth, PyObject * args);
double dist_depth_range_error(EarthModel_t * p_earth, int phaseid,
                              double depth, double distance);
