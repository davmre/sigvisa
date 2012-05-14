#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../sigvisa.h"

/* CONSTANTS */

#define	SQRT3_OVER2	       ((double) 0.866025404)

#define MAX_LINE               1000      /* for reading ttime files */

#define EARTH_SURF_P_VEL    5.8              /* surface velocity of P waves */
#define EARTH_SURF_S_VEL    (EARTH_SURF_P_VEL / 1.73)

/* MACROS */

/* convert latitude x in radians to geocentric co-latitude in radians */
#define GEOCENTRIC_COLAT(x) \
  ((x) + (((0.192436*sin((x)+(x))) + (0.000323*sin(4.0*(x))))*DEG2RAD))
/* convert from geocentric co-latitude in radians back to geographic latitude
 * in degrees */
#define GEOGRAPHIC_LAT(x)                                               \
  (90.0 - (x*RAD2DEG - (0.192436*sin(x+x) - 0.000323*sin(4.0*x))))

#define	SIGN(a1, a2)	((a2) >= 0 ? -(a1) : (a1))

static char * read_line(FILE * fp)
{
  char * buf;
  buf = (char *)malloc((MAX_LINE+1) * sizeof(*buf));
  int pos;
  int ch;
  
  /* read till the end of the line or the file */
  pos = 0;
  while (((ch = fgetc(fp)) != '\n') && (ch != EOF) && (pos < MAX_LINE))
  {
    buf[pos++] = ch;
  }
  buf[pos] = '\0';

  if (ch != '\n')
  {
    fprintf(stderr, "error: bad file format at '%s'\n", buf);
    free(buf);
    exit(1);
  }

  return buf;
}

static int read_first_int(FILE * fp)
{
  char * buf;
  char * ptr;
  int first_int;
  
  while (1)
  {
    buf = read_line(fp);
    // 'n' is a special-case hard-coded to read TauP output files
    if (buf[0] != '#' && buf[0] != 'n')
      break;
    free(buf);
  }

  ptr = strtok(buf, " ");
  
  if (!ptr || (1 != sscanf(ptr, "%d", &first_int)))
  {
    fprintf(stderr, "error: expecting an integer at '%s'\n", buf);
    free(buf);
    exit(1);
  }
  free(buf);
  return first_int;
}

static double read_first_double(FILE * fp)
{
  char * buf;
  char * ptr;
  double first_double;
  
  while (1)
  {
    buf = read_line(fp);
    if (buf[0] != '#')
      break;
    free(buf);
  }

  ptr = strtok(buf, " ");
  
  if (!ptr || (1 != sscanf(ptr, "%lf", &first_double)))
  {
    fprintf(stderr, "error: expecting a double at '%s'\n", buf);
    free(buf);
    exit(1);
  }
  free(buf);
  return first_double;
}

static int read_doubles(FILE * fp, double * out)
{
  char * buf;
  char * ptr;
  int cnt;
  
  while (1)
  {
    buf = read_line(fp);
    if (buf[0] != '#')
      break;
    free(buf);
  }

  ptr = strtok(buf, " ");
  
  if (!ptr)
  {
    fprintf(stderr, "error: expecting a floating point at '%s'\n", buf);
    free(buf);
    exit(1);
  }

  cnt = 0;
  while(ptr)
  {
    if (1 != sscanf(ptr, "%lg", &out[cnt++]))
    {
      fprintf(stderr, "error: expecting a floating point at '%s'\n", ptr);
      free(buf);
      exit(1);
    }
    
    ptr = strtok(NULL, " ");
  }

  free(buf);
  
  return cnt;
}

static void read_n_doubles(FILE * fp, double * out, int n)
{
  int cnt;
  
  cnt = 0;
  while (cnt < n)
    cnt += read_doubles(fp, out + cnt);
}

static void read_n_lines_doubles(FILE * fp, double * out, int n)
{
  int cnt;
  
  for(cnt=0; cnt < n; ++cnt) {
    *(out+cnt) = read_first_double(fp);
  }
}

static void alloc_sites(PyArrayObject * sitesobj, int * p_nsites,
                        Site_t ** p_p_sites)
{
  int nsites;
  Site_t * p_sites;
  int i;
  
  nsites = sitesobj->dimensions[0];

  p_sites = (Site_t *)calloc(nsites, sizeof(*p_sites));

  for(i=0; i<nsites; i++)
  {
    p_sites[i].sitelon = ARRAY2(sitesobj, i, SITE_LON_COL);
    p_sites[i].sitelat = ARRAY2(sitesobj, i, SITE_LAT_COL);
    p_sites[i].siteelev = ARRAY2(sitesobj, i, SITE_ELEV_COL);
    p_sites[i].siteisarr = (int) ARRAY2(sitesobj, i, SITE_ISARR_COL);
  }
  
  *p_nsites = nsites;
  *p_p_sites = p_sites;
}

static void free_sites(int nsites, Site_t * p_sites)
{
  free(p_sites);
}

static void alloc_phases(PyArrayObject * phasenamesobj, 
                         PyArrayObject * phasetimedefobj,
                         int * p_nphases, char *** p_p_phasenames,
                         int ** p_p_phase_time_def)
{
  int nphases;
  
  char ** p_phasenames;
  int * p_phase_time_def;

  int i;

  nphases = phasenamesobj->dimensions[0];

  p_phasenames = (char **) calloc(nphases, sizeof(*p_phasenames));
  p_phase_time_def = (int *) calloc(nphases, sizeof(*p_phase_time_def));
  
  for (i=0; i<nphases; i++)
  {
    p_phasenames[i] = (char *)calloc(PHASENAME_MAXLEN+1, 
                                     sizeof(p_phasenames[i]));

    strncpy(p_phasenames[i], (char *)PyArray_GETPTR1(phasenamesobj, i),
            PHASENAME_MAXLEN);
    
    p_phase_time_def[i] = BOOLARRAY1(phasetimedefobj, i);
  }

  *p_nphases = nphases;
  *p_p_phasenames = p_phasenames;
  *p_p_phase_time_def = p_phase_time_def;
}

static void free_phases(int nphases, char ** p_phasenames,
                        int * p_phase_time_def)
{
  int i;
  
  for (i=0; i<nphases; i++)
  {
    free(p_phasenames[i]);
  }
  free(p_phasenames);
  free(p_phase_time_def);
}


void read_samples(EarthModel_t * p_earth, int phasenum, const char * table_prefix, double ** pp_samples) {

      FILE * fp;
      int len;
      EarthPhaseModel_t * p_phase = p_earth->p_phases + phasenum;
      char * phasename = p_earth->p_phasenames[phasenum];
      
      /* use the last leg of the phase to determine the surface velocity 
       * otherwise use the first leg and finaly just stick to S */
      if (phasename[strlen(phasename)-1] == 'P')
        p_phase->surf_vel = EARTH_SURF_P_VEL;
      else if (phasename[strlen(phasename)-1] == 'S')
        p_phase->surf_vel = EARTH_SURF_S_VEL;
      else if (phasename[0] == 'P')
        p_phase->surf_vel = EARTH_SURF_P_VEL;
      else if (phasename[0] == 'S')
        p_phase->surf_vel = EARTH_SURF_S_VEL;
      else
        p_phase->surf_vel = EARTH_SURF_S_VEL;        

      
      char * fname = (char *)calloc(1, strlen(table_prefix) + PHASENAME_MAXLEN + 10);
      len = sprintf(fname, "%s%s", table_prefix, phasename);
      fname[len] = '\0';

      fp = fopen(fname, "r");
      if (!fp)
      {
        LogTrace("EarthModel: Unable to open travel time file %s",
		 fname);
	
	// TODO: TauP doesn't seem to give Lg, Rg, or PKPab info; let's ignore that for the moment
	free(fname);
	return;
        //exit(1);
      }
      free(fname);
      
      int numdepth = read_first_int(fp);
      if (p_phase->numdepth != 0) assert(numdepth == p_phase->numdepth);
      p_phase->numdepth = numdepth;
            
      p_phase->p_depths = (double *)calloc(p_phase->numdepth,
                                           sizeof(*p_phase->p_depths));
      read_n_doubles(fp, p_phase->p_depths, p_phase->numdepth);
      
      int numdist = read_first_int(fp);
      if (p_phase->numdist != 0) assert(numdist == p_phase->numdist);
      p_phase->numdist = numdist;
      
      p_phase->p_dists = (double *)calloc(p_phase->numdist,
                                          sizeof(*p_phase->p_dists));
      read_n_doubles(fp, p_phase->p_dists, p_phase->numdist);
      
      *(pp_samples) = (double *)calloc(p_phase->numdist
				       * p_phase->numdepth,
				       sizeof(*pp_samples));
      read_n_lines_doubles(fp, *pp_samples, p_phase->numdist
			   * p_phase->numdepth);
      
      fclose(fp);
}

int py_EarthModel_Init(EarthModel_t * p_earth, PyObject * args)
{
  /* input arguments */
  PyArrayObject * sitesobj;
  PyArrayObject * phasenamesobj;
  PyArrayObject * phasetimedefobj;
  const char * tttable_prefix;
  const char * iatable_prefix;

  int i;
  
  if (!PyArg_ParseTuple(args, "O!O!O!ss", &PyArray_Type, &sitesobj,
                        &PyArray_Type, &phasenamesobj,
                        &PyArray_Type, &phasetimedefobj,
                        &tttable_prefix, &iatable_prefix) || !sitesobj || !phasenamesobj
      || !phasetimedefobj)
    return -1;

  if ((2 != sitesobj->nd) || (NPY_DOUBLE != sitesobj->descr->type_num)
      || (SITE_NUM_COLS != sitesobj->dimensions[1]))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: incorrect shape or type"
                    " of sites array");
    return -1;
  }

  if ((1 != phasenamesobj->nd) 
      || (NPY_STRING != phasenamesobj->descr->type_num))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: incorrect shape or type"
                    " of phasenames array");
    return -1;
  }

  if ((1 != phasetimedefobj->nd) 
      || (NPY_BOOL != phasetimedefobj->descr->type_num))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: incorrect shape or type"
                    " of phasetimedef array");
    return -1;
  }
  
  if (phasenamesobj->dimensions[0] != phasetimedefobj->dimensions[0])
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: phasenames and "
                    "phasetimedef have different number of phases");
    return -1;
  }
  
  alloc_sites(sitesobj, &p_earth->numsites, &p_earth->p_sites);

  alloc_phases(phasenamesobj, phasetimedefobj, &p_earth->numphases,
               &p_earth->p_phasenames, &p_earth->p_phase_time_def);

  
  
  p_earth->p_phases = (EarthPhaseModel_t *)calloc(p_earth->numphases, 
                                                  sizeof(*p_earth->p_phases));

  /* calculate the number of time-defining phases, assuming that the
   * time-defining phases precede the non-time-defining ones */
  for (i=0; i<p_earth->numphases; i++)
  {
    if(!p_earth->p_phase_time_def[i])
      break;
  }
  p_earth->numtimedefphases = i;
  
  for (i=0; i<p_earth->numphases; i++)
  {
    EarthPhaseModel_t * p_phase;
    p_phase = p_earth->p_phases + i;
    if (p_earth->p_phase_time_def[i])
    {
      read_samples(p_earth, i, tttable_prefix, &p_phase->p_ttsamples);
      read_samples(p_earth, i, iatable_prefix, &p_phase->p_iasamples);
    }
  }

  return 0;
}


void py_EarthModel_UnInit(EarthModel_t * p_earth)
{
  int i;

  for (i=0; i<p_earth->numphases; i++)
  {
    EarthPhaseModel_t * p_phase;
    p_phase = p_earth->p_phases + i;

    if (p_earth->p_phase_time_def[i])
    {
      free(p_phase->p_depths);
      free(p_phase->p_dists);
      free(p_phase->p_ttsamples);
      free(p_phase->p_iasamples);
    }
  }
  
  free(p_earth->p_phases);

  free_sites(p_earth->numsites, p_earth->p_sites);
  /* phase_time_def is used above so it has to be freed at the end */
  free_phases(p_earth->numphases, p_earth->p_phasenames,
              p_earth->p_phase_time_def);
}

PyObject * py_EarthModel_InRange(EarthModel_t * p_earth, PyObject * args)
{
  double lon, lat, depth;
  int phaseid, siteid;
  
  if (!PyArg_ParseTuple(args, "dddii", &lon, &lat, &depth, &phaseid, &siteid))
    return NULL;
  
  if (EarthModel_InRange(p_earth, lon, lat, depth, phaseid, siteid))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

double simple_distance_deg(double lon1, double lat1, double lon2,
                           double lat2)
{
  double tmp;
  double dist;
  
  tmp = sin(lat1 * DEG2RAD) * sin(lat2 * DEG2RAD)
    + cos(lat1 * DEG2RAD) * cos(lat2 * DEG2RAD)
    * cos((lon2 - lon1) * DEG2RAD);

  /* handle cases where we get overflow, for example -1.0000000000000002 */
  if (tmp > 1)
    dist = 0;
  else if (tmp < -1)
    dist = 180.0;
  else
    dist = acos(tmp) * RAD2DEG;

  return dist;
}

int EarthModel_InRange(EarthModel_t * p_earth, double lon, double lat,
                       double depth, int phaseid, int siteid)
{
  assert((phaseid < p_earth->numphases) && (siteid < p_earth->numsites));
  
  /* non-time-defining phases are never in-range */
  if (!p_earth->p_phase_time_def[phaseid] 
      || (EarthModel_ArrivalTime(p_earth, lon, lat, depth, 0, phaseid, siteid)
          < 0))
    return 0;

  return 1;
}

static void dist_azimuth(double alon1, double alat1, double alon2, 
                         double alat2, double *delta, double *azi, double *baz)
{
  double clat1, cdlon, cdel, geoc_co_lat, geoc_lat1, geoc_lat2;
  double geog_co_lat, rdlon, slat1, sdlon, xazi, xbaz, yazi, ybaz;
  double clat2, slat2;

  /*
   * Simple case when both sets of lat/lons are the same.
   */

  if ((alat1 == alat2) && (alon1 == alon2))
  {
    *delta = 0.0;
    *azi = 0.0;
    *baz = 180.0;
    return;
  }
  
  /*
   * Convert alat2 from geographic latitude to geocentric latitude 
   * (radians) in geoc_lat2
   */
  geog_co_lat = (90.0-(alat2))*DEG2RAD;
  geoc_co_lat = GEOCENTRIC_COLAT(geog_co_lat);
  geoc_lat2 = 90.0*DEG2RAD-geoc_co_lat;
  
  clat2 = cos(geoc_lat2);
  slat2 = sin(geoc_lat2);

  
  /*
   * Convert alat1 from geographic latitude to geocentric latitude 
   * (radians) in geoc_lat1
   */
  geog_co_lat = (90.0-(alat1))*DEG2RAD;
  geoc_co_lat = GEOCENTRIC_COLAT(geog_co_lat);
  geoc_lat1 = 90.0*DEG2RAD-geoc_co_lat;

  rdlon = DEG2RAD * (alon2 - alon1);

  clat1 = cos(geoc_lat1);
  slat1 = sin(geoc_lat1);
  cdlon = cos(rdlon);
  sdlon = sin(rdlon);

  cdel = slat1*slat2 + clat1*clat2*cdlon;
  cdel = (cdel <  1.0) ? cdel :  1.0;
  cdel = (cdel > -1.0) ? cdel : -1.0;
  yazi = sdlon * clat2;
  xazi = clat1*slat2 - slat1*clat2*cdlon;
  ybaz = -sdlon * clat1;
  xbaz = clat2*slat1 - slat2*clat1*cdlon;
  
  *delta = RAD2DEG * acos(cdel);
  *azi   = RAD2DEG * atan2(yazi, xazi);
  *baz   = RAD2DEG * atan2(ybaz, xbaz);
  
  if (*azi < 0.0)
    *azi += 360.0;
  if (*baz < 0.0)
    *baz += 360.0;
}

double EarthModel_Delta(EarthModel_t * p_earth, double lon, double lat,
                        int siteid)
{
  double delta, esaz, seaz;
  Site_t * p_site;
  p_site = p_earth->p_sites + siteid;
  
  dist_azimuth(lon, lat, p_site->sitelon, p_site->sitelat,
               &delta, &esaz, &seaz);
  return delta;
}

PyObject * py_EarthModel_Delta(EarthModel_t * p_earth, PyObject * args)
{
  double lon, lat;
  int siteid;
  
  if (!PyArg_ParseTuple(args, "ddi", &lon, &lat, &siteid))
    return NULL;

  if ((siteid < 0) || (siteid > p_earth->numsites))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid siteid");
    return NULL;
  }

  return Py_BuildValue("d", EarthModel_Delta(p_earth, lon, lat, siteid));
}


// convert an arriving slowness (seconds/degree) to incidence angle
int slowness_to_iangle(double slowness, int phase, double * iangle) {
 int success = 1;

  double v; // phase-specific velocity in deg/sec
  switch (phase) {
  case 0:
  case 1:
  case 2:
  case 5:
  case 6:
  case 7:
  case 8:
  case 10:
  case 12:
  case 13:
    /* these are essentially all of the P-wave phases, except for ScP
       which is backwards for some reason */
    v = EARTH_SURF_P_VEL * 360/AVG_EARTH_CIRCUMFERENCE_KM;
    break;
  case 11:
    /* ScP */
    v = -1 * EARTH_SURF_P_VEL * 360/AVG_EARTH_CIRCUMFERENCE_KM;
    break;
  case 3:
  case 4:
    /* S-wave phases */
    v = EARTH_SURF_S_VEL * 360/AVG_EARTH_CIRCUMFERENCE_KM;
    break;
  default:
    /* I don't currently have numbers for Love or Raleigh waves */
    success = 0;
  }
  *iangle = asin(v * slowness) * RAD2DEG;
  if (isnan(*iangle)) success = 0;
  return success;
}


// convert an arriving slowness (seconds/degree) to incidence angle
int iangle_to_slowness(double iangle, int phase, double * slowness) {
 int success = 1;

  double v; // phase-specific velocity in deg/sec
  switch (phase) {
  case 0:
  case 1:
  case 2:
  case 5:
  case 6:
  case 7:
  case 8:
  case 10:
  case 12:
  case 13:
    /* these are essentially all of the P-wave phases, except for ScP
       which is backwards for some reason */
    v = EARTH_SURF_P_VEL * 360/AVG_EARTH_CIRCUMFERENCE_KM;
    break;
  case 11:
    /* ScP */
    v = -1 * EARTH_SURF_P_VEL * 360/AVG_EARTH_CIRCUMFERENCE_KM;
    break;
  case 3:
  case 4:
    /* S-wave phases */
    v = EARTH_SURF_S_VEL * 360/AVG_EARTH_CIRCUMFERENCE_KM;
    break;
  default:
    /* I don't currently have numbers for Love or Raleigh waves */
    success = 0;
  }
  
  *slowness = sin(iangle*DEG2RAD)/v;
  if (isnan(*slowness)) success = 0;
  return success;
}

static void travel_time(EarthPhaseModel_t * p_phase, double depth, double
                        distance, double * p_trvtime, double * p_slow,
			double * p_iangle)
{
  int depthi, disti, depthi2, disti2;
  double val11, val12, val21, val22;
  double iaval11=0, iaval12=0, iaval21=0, iaval22=0;
  double mdist11, mdist12, mdist21, mdist22;
  double d_depth, d_dist;
  double slo_val1, slo_val2;
  double slo_mdist1, slo_mdist2;

  /* check that the depth and distance are within the bounds for this phase */

  if ((depth < p_phase->p_depths[0]) 
      || (depth > p_phase->p_depths[p_phase->numdepth-1])
      || (distance < p_phase->p_dists[0]) 
      || (distance > p_phase->p_dists[p_phase->numdist-1]))
  {
    *p_trvtime = *p_slow = *p_iangle = -1;
    return;
  }
  
  for (depthi = 0; (depthi < p_phase->numdepth) 
         && (depth >= p_phase->p_depths[depthi]); depthi++)
    ;
  depthi --;

  if (depthi < (p_phase->numdepth-1))
    depthi2 = depthi + 1;
  else
  {
    depthi = p_phase->numdepth-2;
    depthi2 = p_phase->numdepth-1;
  }
  
  for (disti = 0; (disti < p_phase->numdist) 
         && (distance >= p_phase->p_dists[disti]); disti++)
    ;
  disti --;

  if (disti < (p_phase->numdist-1))
    disti2 = disti + 1;
  else
  {
    disti = p_phase->numdist-2;
    disti2 = p_phase->numdist-1;
  }
  /* the four points are as follows :
   *    1,1   1,2
   *    2,1   2,2
   */
  val11 = EarthPhaseModel_GetTTSample(p_phase, depthi, disti);
  val12 = EarthPhaseModel_GetTTSample(p_phase, depthi, disti2);
  val21 = EarthPhaseModel_GetTTSample(p_phase, depthi2, disti);
  val22 = EarthPhaseModel_GetTTSample(p_phase, depthi2, disti2);

  if(p_phase->p_iasamples != NULL) {
    iaval11 = EarthPhaseModel_GetIASample(p_phase, depthi, disti);
    iaval12 = EarthPhaseModel_GetIASample(p_phase, depthi, disti2);
    iaval21 = EarthPhaseModel_GetIASample(p_phase, depthi2, disti);
    iaval22 = EarthPhaseModel_GetIASample(p_phase, depthi2, disti2);
  }

  if ((val11 < 0) || (val12 < 0) || (val21 < 0) || (val22 < 0))
  {
    *p_trvtime = *p_slow = *p_iangle = -1;
    return;
  }

  d_depth = p_phase->p_depths[depthi2] - p_phase->p_depths[depthi];
  d_dist = p_phase->p_dists[disti2] - p_phase->p_dists[disti];
  
  /* compute the scaled manhattan distance to the four corners */
  mdist11 = (depth - p_phase->p_depths[depthi]) / d_depth 
    + (distance - p_phase->p_dists[disti]) / d_dist;

  mdist12 = (depth - p_phase->p_depths[depthi]) / d_depth
    + (p_phase->p_dists[disti2] - distance) / d_dist;

  mdist21 = (p_phase->p_depths[depthi2] - depth) / d_depth
    + (distance - p_phase->p_dists[disti]) / d_dist;

  mdist22 = (p_phase->p_depths[depthi2] - depth) / d_depth
    + (p_phase->p_dists[disti2] - distance) / d_dist;

  /* compute the travel time and incident angle */
  if (!mdist11) {
    *p_trvtime = val11;
    *p_iangle = iaval11;
  }
  else if (!mdist12) {
    *p_trvtime = val12;
    *p_iangle = iaval12;
  }
  else if (!mdist21) {
    *p_trvtime = val21;
    *p_iangle = iaval21;
  }
  else if (!mdist22) {
    *p_trvtime = val22;
    *p_iangle = iaval22;
  }
  else
  {
    assert((mdist11 > 0) && (mdist12 > 0) && (mdist21 > 0) && (mdist22 > 0));

    /* for debugging*
    if (fabs(distance - 137.1301) < 1e-3)
    {
      printf("depth %lf distance %lf\n", depth, distance);
      printf("val11 %lf val12 %lf\nval21 %lf val22 %lf\n", val11, val12, val21,
             val22);
      printf("mdist11 %lf mdist12 %lf mdist21 %lf mdist22 %lf\n",
             mdist11, mdist12, mdist21, mdist22);
    }
    **/

    *p_trvtime = (val11 / mdist11 + val12 / mdist12 + val21 / mdist21 
                  + val22 / mdist22)
      / (1/mdist11 + 1/mdist12 + 1/mdist21 + 1/mdist22);
    *p_iangle = (iaval11 / mdist11 + iaval12 / mdist12 + iaval21 / mdist21 
                  + iaval22 / mdist22)
      / (1/mdist11 + 1/mdist12 + 1/mdist21 + 1/mdist22);
  }

  slo_val1 = (val12 - val11) / d_dist;
  slo_val2 = (val22 - val21) / d_dist;
  slo_mdist1 = depth - p_phase->p_depths[depthi];
  slo_mdist2 = p_phase->p_depths[depthi2] - depth;

  /* compute the slowness */
  if (!slo_mdist1)
    *p_slow = slo_val1;

  else if (!slo_mdist2)
    *p_slow = slo_val2;

  else
  {
    assert((slo_mdist1 > 0) && (slo_mdist2 > 0));

    /* for debugging*
    if (fabs(distance - 80.653904) < 1e-3)
    {
      printf("depth %lf distance %lf\n", depth, distance);
      printf("val11 %lf val12 %lf\nval21 %lf val22 %lf\n", val11, val12, val21,
             val22);
      printf("mdist11 %lf mdist12 %lf mdist21 %lf mdist22 %lf\n",
             mdist11, mdist12, mdist21, mdist22);
      printf("val1 %lf val2 %lf\nmdist1 %lf mdist2 %lf\n",
             slo_val1, slo_val2, slo_mdist1, slo_mdist2);
    }
    **/
    
    *p_slow = (slo_val1 / slo_mdist1 + slo_val2 / slo_mdist2) 
      / (1 / slo_mdist1 + 1 / slo_mdist2);
  }
}

static double ellipticity_corr (double delta, double esaz, double ecolat,
                                double depth, int phaseid)
{
  int     iphs;
  double  adepth, azim, edist, ellip_corr = 0.0;
  double  sc0, sc1, sc2, t0, t1, t2;

  static  double  t[][8][10] =
    {
      {
        /* t0 constants */
        { -0.01711, -1.7791,   0.0000,   0.0000,  0.0000, -0.9630,
          -13.2326,  13.7390,   0.0000,   0.0000 },
        { -0.08291, -2.1455,   2.4538,  -0.7907,  0.0000,  2.0258,
          -12.9357,   2.1287,   5.2668,  -0.9229 },
        { -1.5022,  -0.0943,   1.9655,  -1.1661,  0.1393,  3.4920,
          -9.9051,  -0.3875,   5.3581,  -0.0686 },
        {  2.9971,  -2.9549,   0.4082,   0.0000,  0.0000, 28.1650,
           9.2160, -17.9030,  -5.2995,   3.2029 },
        {  3.6775,  -2.2221,   0.0000,   0.0000,  0.0000, -1.3127,
           -6.2476,   1.6684,   0.0000,   0.0000 },
        { -10.6238,  15.4993,  -7.4840,   1.0673,  0.0000,  3.2763,
          -6.4596,  -0.4923,   0.0000,   0.0000 },
        { -0.01332, -3.2777,  -1.2243,   7.5246,  0.0000, -3.4856,
          -10.3187,  43.4834, -70.5341, -50.2287 },
        { -0.07859, -4.0924,   4.6116,  -1.4760,  0.0000,  2.9104,
          -17.8661,   4.6262,   7.1486,  -1.9154 }
      },
      {
        /* t1 constants */
        { 0.0040,  -0.7841,   6.0441, -17.5535,  0.0000, -0.2549,
          2.0519, -19.0605, -37.8235,  54.5110 },
        { -0.0048,   0.0839,  -2.2705,   2.4137, -0.5957, -2.4241,
          -4.2792,   1.9728,   3.5644,  -0.5285 },
        { 0.0033,  -1.3485,   0.1735,   1.1583, -0.4162, -0.1096,
          0.2576,  -0.5978,   0.1888,   0.1600 },
        { 2.6249,  -0.0025,  -0.2086,  -0.0184,  0.0000, -1.5077,
          0.9904,   0.3513,   0.0000,   0.0000 },
        { 3.4213,  -0.9359,   0.0000,   0.0000,  0.0000,  0.0000,
          0.0000,   0.0000,   0.0000,   0.0000 },
        { -8.0633,   8.0238,  -1.7407,   0.0000,  0.0000,  0.0000,
          0.0000,   0.0000,   0.0000,   0.0000 },
        { 0.0109,  -1.2300,   8.9145, -27.5847,  0.0000, -0.6951,
          5.6201, -33.0908, -83.8233, 102.4333 },
        { -0.0311,   0.1896,  -4.0694,   4.2599, -1.0387, -3.9368,
          -8.4379,   2.6814,   6.9535,  -0.6086 }
      },
      {
        /* t2 constants */
        { 0.0107,   0.0275,  -0.6912,   0.0347,  0.1157, -0.1836,
          0.0000,   0.0296,   0.0000,   0.0000 },
        { 0.0107,   0.0275,  -0.6912,   0.0347,  0.1157, -0.1836,
          0.0000,   0.0296,   0.0000,   0.0000 },
        { 0.0005,  -0.01231, -1.0156,   0.4396,  0.0000,  0.0000,
          0.0000,   0.0000,   0.0000,   0.0000 },
        { -3.5838,   2.1474,  -0.3548,   0.0000,  0.0000, -1.3369,
          -5.4889,   0.6809,   1.5096,  -0.0763 },
        { -2.9912,   1.0313,   0.0000,   0.0000,  0.0000,  0.0000,
          0.0000,   0.0000,   0.0000,   0.0000 },
        { 3.2814,  -7.1224,   3.5418,  -0.5115,  0.0000,  0.0000,
          0.0000,   0.0000,   0.0000,   0.0000 },
        { 0.00025,  0.1685,  -2.2435,   3.3433,  0.0000, -0.0503,
          0.5353,   1.5362, -14.3118,  -3.2938 },
        { 0.0843,  -0.2917,  -0.6767,  -0.2934,  0.2779, -0.4336,
          0.0306,  0.07113,   0.0000,   0.0000 }
      }
    };
 

  /*
   * First, determine phase-type index
   */
  if (EARTH_PHASE_P == phaseid)
  {
    if (delta < 15.0)
      iphs = 0;
    else if (delta < 110.0)
      iphs = 1;
    else
      iphs = 5;                             /* Use the PKPdf branch */
  }
  else if (EARTH_PHASE_PcP == phaseid)
  {
    iphs = 2;
    if (delta > 90.0)          /* Correction not valid at < 90 deg. */
      return (0.0);
  }
  else if (EARTH_PHASE_PKPab == phaseid)
  {
    iphs = 3;
    if (delta < 140.0)        /* Correction not valid at < 140 deg. */
      return (0.0);
  }
  else if (EARTH_PHASE_PKPbc == phaseid)
  {
    iphs = 4;
    if (delta < 140.0 || delta > 160.0) 
      /* Correction not valid except between 140 & 160 deg. */
      return (0.0);
  }
  else if (EARTH_PHASE_PKP == phaseid) /* TODO (! strcmp (phase, "PKIKP"))*/
  {
    iphs = 5;
    if (delta < 110.0)        /* Correction not valid at < 110 deg. */
      return (0.0);
  }
  else if (EARTH_PHASE_S == phaseid)
  {
    if (delta < 15.0)
      iphs = 6;
    else if (delta < 110.0)
      iphs = 7;
    else                      /* Correction not valid at > 110 deg. */
      return (0.0);
  }
  else
    return (0.0);     /* No ellipt. correction exists for this phase*/


  edist  = delta * DEG2RAD;            /* Event to station distance */
  azim   = esaz * DEG2RAD;              /* Event to station azimuth */

  /*
   * Set up reference constants
   */

  sc0 = 0.25*(1.0 + 3.0*cos(2.0*ecolat));
  sc1 = SQRT3_OVER2*sin(2.0*ecolat);
  sc2 = SQRT3_OVER2*sin(ecolat)*sin(ecolat);

  adepth = depth/AVG_EARTH_RADIUS_KM;

  /*
   * Compute tau coefficients of Dziewonski and Gilbert (1976).
   */

  t0 = t[0][iphs][0] + edist*(t[0][iphs][1] + edist*(t[0][iphs][2]
                + edist*(t[0][iphs][3] + edist*t[0][iphs][4])))
                + adepth*(t[0][iphs][5] + adepth*t[0][iphs][6])
                + adepth*edist*(t[0][iphs][7] + t[0][iphs][8]*adepth
                                + t[0][iphs][9]*edist);
  t1 = t[1][iphs][0] + edist*(t[1][iphs][1] + edist*(t[1][iphs][2]
                + edist*(t[1][iphs][3] + edist*t[1][iphs][4])))
                + adepth*(t[1][iphs][5] + adepth*t[1][iphs][6])
                + adepth*edist*(t[1][iphs][7] + t[1][iphs][8]*adepth
                                + t[1][iphs][9]*edist);
  t2 = t[2][iphs][0] + edist*(t[2][iphs][1] + edist*(t[2][iphs][2]
                + edist*(t[2][iphs][3] + edist*t[2][iphs][4])))
                + adepth*(t[2][iphs][5] + adepth*t[2][iphs][6])
                + adepth*edist*(t[2][iphs][7] + t[2][iphs][8]*adepth
                                + t[2][iphs][9]*edist);

  /*
   * Compute ellipticity correction via equations (22) and (26) 
   * of Dziewonski and Gilbert (1976).
   */

  ellip_corr = sc0*t0 + sc1*cos(azim)*t1 + sc2*cos(2.0*azim)*t2;

/*
        printf ("dist = %5.1f  t0 = %7.3f  t1 = %7.3f  t2 = %7.3f  ellip_corr = %7.3f\n", delta, t0, t1, t2, ellip_corr);
 */

  return (ellip_corr);
}

PyObject * py_EarthModel_ArrivalTime(EarthModel_t * p_earth, PyObject * args)
{
  double lon, lat, depth, evtime;
  int phaseid, siteid;
  
  if (!PyArg_ParseTuple(args, "ddddii", &lon, &lat, &depth, &evtime, 
                        &phaseid, &siteid))
    return NULL;

  if ((phaseid < 0) || (phaseid > p_earth->numphases) || (siteid < 0) ||
      (siteid > p_earth->numsites))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid phaseid or siteid"
      );
    return NULL;
  }

  if (!p_earth->p_phase_time_def[phaseid])
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: phaseid is not time-def");
    return NULL;
  }

  return Py_BuildValue("d", EarthModel_ArrivalTime(p_earth, lon, lat, depth,
                                                   evtime, phaseid, siteid));
}

PyObject * py_EarthModel_ArrivalTime_Coord(EarthModel_t * p_earth,
                                           PyObject * args)
{
  double lon, lat, depth, evtime;
  int phaseid;
  double sitelon, sitelat, siteelev;
  
  if (!PyArg_ParseTuple(args, "ddddiddd", &lon, &lat, &depth, &evtime, 
                        &phaseid, &sitelon, &sitelat, &siteelev))
    return NULL;

  if ((phaseid < 0) || (phaseid > p_earth->numphases))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid phaseid");
    return NULL;
  }

  if (!p_earth->p_phase_time_def[phaseid])
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: phaseid is not time-def");
    return NULL;
  }

  return Py_BuildValue("d",
                       EarthModel_ArrivalTime_Coord(p_earth, lon, lat, depth,
                                                    evtime, phaseid, sitelon,
                                                    sitelat, siteelev));
}

double EarthModel_ArrivalTime(EarthModel_t * p_earth, double lon,
                              double lat, double depth, double evtime, 
                              int phaseid, int siteid)
{
  Site_t * p_site;
  
  assert(siteid < p_earth->numsites);
  
  p_site = p_earth->p_sites + siteid;  

  return EarthModel_ArrivalTime_Coord(p_earth, lon, lat, depth,
                                      evtime, phaseid, p_site->sitelon,
                                      p_site->sitelat, p_site->siteelev);
}


double EarthModel_ArrivalTime_Coord(EarthModel_t * p_earth, double lon, 
                                    double lat, double depth, double evtime, 
                                    int phaseid, double sitelon,
                                    double sitelat, double siteelev)
{
  double trvtime, slow, iangle;
  EarthPhaseModel_t * p_phase;
  double delta, esaz, seaz;

  assert((phaseid < p_earth->numphases) && p_earth->p_phase_time_def[phaseid]);
  
  p_phase = p_earth->p_phases + phaseid;
  
  dist_azimuth(lon, lat, sitelon, sitelat, &delta, &esaz, &seaz);
  
  travel_time(p_phase, depth, delta, &trvtime, &slow, &iangle);
  
  if (trvtime < 0)
    return -1;
  
  trvtime += ellipticity_corr(delta, esaz,
                              GEOCENTRIC_COLAT((90-lat) * DEG2RAD), 
                              depth, phaseid);

  if (siteelev >= -998)
    trvtime += siteelev / p_phase->surf_vel;
  
  return evtime + trvtime;
}


PyObject * py_EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, 
                                        PyObject * args)
{
  double lon, lat;
  int siteid;
  
  if (!PyArg_ParseTuple(args, "ddi", &lon, &lat, &siteid))
    return NULL;

  if ((siteid < 0) || (siteid > p_earth->numsites))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid siteid");
    return NULL;
  }

  return Py_BuildValue("d", EarthModel_ArrivalAzimuth(p_earth, lon, lat,
                                                      siteid));
}

double EarthModel_ArrivalAzimuth(EarthModel_t * p_earth, double lon,
                                 double lat, int siteid)
{
  Site_t * p_site;
  double delta;
  double esaz;
  double seaz;
  
  assert((siteid >= 0) && (siteid < p_earth->numsites));

  p_site = p_earth->p_sites + siteid;

  dist_azimuth(lon, lat, p_site->sitelon, p_site->sitelat,
               &delta, &esaz, &seaz);
  
  return seaz;
}


double EarthModel_ArrivalIncidentAngle(EarthModel_t * p_earth, double lon,
				       double lat, double depth,
				       int phaseid, int siteid)
{
  double trvtime, slow, iangle;
  Site_t * p_site;
  EarthPhaseModel_t * p_phase;
  double delta, esaz, seaz;

  assert((siteid < p_earth->numsites) && (phaseid < p_earth->numphases)
         && p_earth->p_phase_time_def[phaseid]);
  
  p_site = p_earth->p_sites + siteid;
  p_phase = p_earth->p_phases + phaseid;
  
  dist_azimuth(lon, lat, p_site->sitelon, p_site->sitelat, &delta, &esaz,
               &seaz);
  
  travel_time(p_phase, depth, delta, &trvtime, &slow, &iangle);

  return iangle;
}

PyObject * py_EarthModel_ArrivalIncidentAngle(EarthModel_t * p_earth, 
					      PyObject * args)
{
  double lon, lat, depth;
  int phaseid, siteid;
  
  if (!PyArg_ParseTuple(args, "dddii", &lon, &lat, &depth, 
                        &phaseid, &siteid))
    return NULL;

  if ((phaseid < 0) || (phaseid > p_earth->numphases) || (siteid < 0) ||
      (siteid > p_earth->numsites))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid phaseid or siteid"
      );
    return NULL;
  }

  if (!p_earth->p_phase_time_def[phaseid])
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: phaseid is not time-def");
    return NULL;
  }

  return Py_BuildValue("d", EarthModel_ArrivalIncidentAngle(p_earth, lon, 
							    lat, depth, 
							    phaseid,siteid));
}


double EarthModel_ArrivalSlowness(EarthModel_t * p_earth, double lon,
                                  double lat, double depth,
                                  int phaseid, int siteid)
{
  double trvtime, slow, iangle;
  Site_t * p_site;
  EarthPhaseModel_t * p_phase;
  double delta, esaz, seaz;

  assert((siteid < p_earth->numsites) && (phaseid < p_earth->numphases)
         && p_earth->p_phase_time_def[phaseid]);
  
  p_site = p_earth->p_sites + siteid;
  p_phase = p_earth->p_phases + phaseid;
  
  dist_azimuth(lon, lat, p_site->sitelon, p_site->sitelat, &delta, &esaz,
               &seaz);
  
  travel_time(p_phase, depth, delta, &trvtime, &slow, &iangle);

  return slow;
}

PyObject * py_EarthModel_ArrivalSlowness(EarthModel_t * p_earth, 
                                         PyObject * args)
{
  double lon, lat, depth;
  int phaseid, siteid;
  
  if (!PyArg_ParseTuple(args, "dddii", &lon, &lat, &depth, 
                        &phaseid, &siteid))
    return NULL;

  if ((phaseid < 0) || (phaseid > p_earth->numphases) || (siteid < 0) ||
      (siteid > p_earth->numsites))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid phaseid or siteid"
      );
    return NULL;
  }

  if (!p_earth->p_phase_time_def[phaseid])
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: phaseid is not time-def");
    return NULL;
  }

  return Py_BuildValue("d", EarthModel_ArrivalSlowness(p_earth, lon, lat,
                                                       depth, phaseid,siteid));
}

PyObject * py_EarthModel_IsTimeDefPhase(EarthModel_t * p_earth, 
                                        PyObject * args)
{
  int phaseid;
  
  if (!PyArg_ParseTuple(args, "i", &phaseid))
    return NULL;

  if ((phaseid < 0) || (phaseid > p_earth->numphases))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid phaseid");
    return NULL;
  }

  if(EarthModel_IsTimeDefPhase(p_earth, phaseid))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject * py_EarthModel_NumPhases(EarthModel_t * p_earth, 
                                   PyObject * args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_BuildValue("i", EarthModel_NumPhases(p_earth));
}

PyObject * py_EarthModel_NumTimeDefPhases(EarthModel_t * p_earth, 
                                   PyObject * args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_BuildValue("i", EarthModel_NumTimeDefPhases(p_earth));
}

PyObject * py_EarthModel_NumSites(EarthModel_t * p_earth, 
                                  PyObject * args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_BuildValue("i", EarthModel_NumSites(p_earth));
}

PyObject * py_EarthModel_DiffAzimuth(EarthModel_t * p_earth, 
                                     PyObject * args)
{
  double azi1, azi2;
  
  if (!PyArg_ParseTuple(args, "dd", &azi1, &azi2))
    return NULL;

  return Py_BuildValue("d", EarthModel_DiffAzimuth(azi1, azi2));
}

double EarthModel_DiffAzimuth(double azi1, double azi2)
{
  double diff;
  
  diff = azi2 - azi1;
  
  if (diff > 180)
    return diff - 360;
  else if (diff <= -180)
    return diff + 360;
  else
    return diff;
}

PyObject * py_EarthModel_PhaseName(EarthModel_t * p_earth, 
                                   PyObject * args)
{
  int phaseid;
  
  if (!PyArg_ParseTuple(args, "i", &phaseid))
    return NULL;

  if ((phaseid < 0) || (phaseid >= p_earth->numphases))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: illegal phaseid");
    
    return NULL;
  }

  return Py_BuildValue("s", p_earth->p_phasenames[phaseid]);
}

PyObject * py_EarthModel_MaxTravelTime(EarthModel_t * p_earth, 
                                       PyObject * args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  return Py_BuildValue("d", MAX_TRAVEL_TIME);
}

/* returns distance or -1 if not invertible */
static double invert_slowness(const EarthPhaseModel_t * p_phase, double depth,
                              double slow)
{
  int depthi;
  int disti;
  
  /* check that the depth is within the bounds for this phase */
  if ((depth < p_phase->p_depths[0])
      || (depth > p_phase->p_depths[p_phase->numdepth-1]))
    return -1;

  for (depthi = 0; (depthi < p_phase->numdepth) 
         && (depth >= p_phase->p_depths[depthi]); depthi++)
    ;
  depthi --;
  
  for (disti = 0; (disti+1) < p_phase->numdist; disti++)
  {
    double curr_slow;
    
    curr_slow = (EarthPhaseModel_GetTTSample(p_phase, depthi, disti+1)
                 - EarthPhaseModel_GetTTSample(p_phase, depthi, disti))
      / (p_phase->p_dists[disti+1] - p_phase->p_dists[disti]);
    
    if (slow > curr_slow)
      break;
  }

  return p_phase->p_dists[disti];
}

/* delta,azi is geocentric; lon,lat are all geographic */
/* From idc ->libgeog -> lat_lon.c */
static void invert_dist_azimuth(double alon1, double alat1, double delta, 
                                double azi, double *alon2, double *alat2)
{
  double alat, alon, a, b, c, coslat, dlon;
  double geoc_co_lat, geog_co_lat;
  double  r123, r13, r13sq, sinlat, x1, x2, x3;
  
  /*
   * Convert a geographical location to geocentric cartesian 
   * coordinates, assuming a spherical Earth.
   */
  alat = 90.0 - delta;
  alon = 180.0 - azi;
  r13  = cos(DEG2RAD*alat);

  /*
   * x1:  Axis 1 intersects equator at  0 deg longitude  
   * x2:  Axis 2 intersects equator at 90 deg longitude  
   * x3:  Axis 3 intersects north pole
   */

  x1 = r13*sin(DEG2RAD*alon);
  x2 = sin(DEG2RAD*alat);
  x3 = r13*cos(DEG2RAD*alon);

  geog_co_lat = (90.0-alat1)*DEG2RAD;          /* radians */
  geoc_co_lat = GEOCENTRIC_COLAT(geog_co_lat);    /* radians */

  /*
   * Rotate in cartesian coordinates.  The cartesian coordinate system 
   * is most easily described in geocentric terms.  The origin is at 
   * the Earth's center.  Rotation by alat1 degrees southward, about 
   * the 1-axis.
   */

  sinlat = sin(geoc_co_lat);
  coslat = cos(geoc_co_lat);
  b      = x2;
  c      = x3;
  x2     = b*coslat - c*sinlat;
  x3     = b*sinlat + c*coslat;

  /*
   * Finally, convert geocentric cartesian coordinates back to 
   * a geographical location.
   */
 
  r13sq  = x3*x3 + x1*x1;
  r13    = sqrt(r13sq);
  r123   = sqrt(r13sq + x2*x2);
  dlon   = RAD2DEG*atan2(x1, x3);
  a      = 90.0*DEG2RAD - atan2(x2, r13);
  *alat2 = GEOGRAPHIC_LAT(a);
  *alon2 = alon1 + dlon;
  if (fabs(*alat2) > 90.0)
    *alat2 = SIGN((180.0-fabs(*alat2)), *alat2);
  if (fabs(*alon2) > 180.0)
    *alon2 = SIGN((360.0-fabs(*alon2)), *alon2);
}


/* if possible to invert, returns 0 and stores the lon, lat, depth, time
 * fields, else it returns -1 */
int invert_detection(const EarthModel_t * p_earth, const Detection_t * p_det,
                     Event_t * p_event, int perturb)
{
  double dist;
  int phaseid;
  const EarthPhaseModel_t * p_phase;
  const Site_t * p_site;
  double arrtime;  
  
  /* inverting the P phase leads to the most number of events */
  phaseid = EARTH_PHASE_P;

  p_phase = p_earth->p_phases + phaseid;
  
  /* we want to fix the depth at zero otherwise invert_slowness can produce
   * unexpected results */
  p_event->evdepth = 0;
  
  dist = invert_slowness(p_phase, p_event->evdepth, p_det->slo_det);
  
  /* we don't want to propose an event smack on top of a station! */
  if (dist < 1e-3) {
    printf("invert failed, too close to station\n");
    return -1;
  }
  p_site = p_earth->p_sites + p_det->site_det;
  
  invert_dist_azimuth(p_site->sitelon, p_site->sitelat, dist,
                      p_det->azi_det,
                      &p_event->evlon, &p_event->evlat);

  /* now, perturb the event location */
  if (perturb)
  {
    p_event->evlon += Gaussian_sample(0, 1);

    p_event->evlat += Gaussian_sample(0, 1);
  
    if (p_event->evlon < -180)
      p_event->evlon += 360;
    else if (p_event->evlon >= 180)
      p_event->evlon -= 360;
  
    if (p_event->evlat < -90)
      p_event->evlat = -180 - p_event->evlat;
    else if (p_event->evlat > 90)
      p_event->evlat = 180 - p_event->evlat;
  }
  
  arrtime = EarthModel_ArrivalTime((EarthModel_t *)p_earth, 
                                   p_event->evlon, p_event->evlat,
                                   p_event->evdepth, 0 /* evtime */,
                                   phaseid, p_det->site_det);
  
  if (arrtime < 0) {
    printf("invert failed, couldn't compute arrtime from lon %lf lat %lf dep %lf\n", p_event->evlon, p_event->evlat, p_event->evdepth);
    return -1;
  }
  
  p_event->evtime = p_det->time_det - arrtime;
  
  /* perturb the arrival time as well */
  if (perturb)
  {
    p_event->evtime += Gaussian_sample(0, 5);
  }
  
  /* low magnitude to ensure better odds of acceptance */
  p_event->evmag = MIN_MAGNITUDE;
  
  return 0;
}

PyObject * py_EarthModel_InvertDetection(const EarthModel_t * p_earth, PyObject * args) {
  int siteid;
  double azi, slo, time;
  
  if (!PyArg_ParseTuple(args, "iddd", &siteid, &azi, &slo, &time))
    return NULL;

  Detection_t d;
  d.site_det = siteid-1;
  d.time_det = time;
  d.azi_det = azi;
  d.slo_det = slo;

  Event_t ev;
  
  invert_detection(p_earth, &d, &ev, 0);

  return Py_BuildValue("dddd", ev.evlon, ev.evlat, ev.evdepth, ev.evtime);
}


PyObject * py_EarthModel_PhaseRange(EarthModel_t * p_earth, PyObject * args)
{
  int phaseid;
  EarthPhaseModel_t * p_phase;
  
  if (!PyArg_ParseTuple(args, "i", &phaseid))
    return NULL;

  if ((phaseid < 0) || (phaseid >= p_earth->numphases))
  {
    PyErr_SetString(PyExc_ValueError, "EarthModel: invalid phaseid");
    return NULL;
  }

  p_phase = p_earth->p_phases + phaseid;

  return Py_BuildValue("dd", p_phase->p_dists[0],
                       p_phase->p_dists[p_phase->numdist-1]);
}


