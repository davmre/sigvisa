#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../netvisa.h"

/* CONSTANTS */

#define PI                     ((double) 3.1415926535897931)
#define DEG2RAD                ((double) (PI / 180))
#define RAD2DEG                ((double) (180 / PI))
#define AVG_EARTH_RADIUS_KM    ((double) 6371) /* when modeled as a sphere */
#define	SQRT3_OVER2	       ((double) 0.866025404)

#define MAX_LINE               1000      /* for reading ttime files */

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
    if (buf[0] != '#')
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

int py_EarthModel_Init(EarthModel_t * p_earth, PyObject * args)
{
  /* input arguments */
  PyArrayObject * sitesobj;
  PyArrayObject * phasenamesobj;
  PyArrayObject * phasetimedefobj;
  const char * tttable_prefix;

  int i;
  char * fname;
  
  if (!PyArg_ParseTuple(args, "O!O!O!s", &PyArray_Type, &sitesobj,
                        &PyArray_Type, &phasenamesobj,
                        &PyArray_Type, &phasetimedefobj,
                        &tttable_prefix) || !sitesobj || !phasenamesobj
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

  fname = (char *)calloc(1, strlen(tttable_prefix) + PHASENAME_MAXLEN + 10);
  
  p_earth->p_phases = (EarthPhaseModel_t *)calloc(p_earth->numphases, 
                                                  sizeof(*p_earth->p_phases));
  
  for (i=0; i<p_earth->numphases; i++)
  {
    EarthPhaseModel_t * p_phase;
    p_phase = p_earth->p_phases + i;
    
    if (p_earth->p_phase_time_def[i])
    {
      FILE * fp;
      int len;

      len = sprintf(fname, "%s%s", tttable_prefix, p_earth->p_phasenames[i]);
      fname[len] = '\0';

      fp = fopen(fname, "r");
      if (!fp)
      {
        fprintf(stderr, "EarthModel: Unable to open travel time file %s",
                fname);
        exit(1);
      }
      
      p_phase->numdepth = read_first_int(fp);
      
      p_phase->p_depths = (double *)calloc(p_phase->numdepth,
                                           sizeof(*p_phase->p_depths));
      read_n_doubles(fp, p_phase->p_depths, p_phase->numdepth);
      
      p_phase->numdist = read_first_int(fp);
      
      p_phase->p_dists = (double *)calloc(p_phase->numdist,
                                          sizeof(*p_phase->p_dists));
      read_n_doubles(fp, p_phase->p_dists, p_phase->numdist);
      
      p_phase->p_samples = (double *)calloc(p_phase->numdist
                                            * p_phase->numdepth,
                                            sizeof(*p_phase->p_samples));
      read_n_doubles(fp, p_phase->p_samples, p_phase->numdist
                     * p_phase->numdepth);
      
      fclose(fp);
    }
  }
  free(fname);

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
      free(p_phase->p_samples);
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

static double simple_distance_deg(double lon1, double lat1, double lon2,
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
  EarthPhaseModel_t * p_phase;
  Site_t * p_site;
  double dist;
  
  assert((phaseid < p_earth->numphases) && (siteid < p_earth->numsites));
  
  /* non-time-defining phases are always feasible */
  /* TODO: are tx Px and Sx phases always feasible? */
  if (!p_earth->p_phase_time_def[phaseid])
    return 1;

  p_phase = p_earth->p_phases + phaseid;

  /* check the depth */
  if ((depth < p_phase->p_depths[0]) 
      || (depth > p_phase->p_depths[p_phase->numdepth-1]))
    return 0;

  /* compute distance to site */
  p_site = p_earth->p_sites + siteid;
  dist = simple_distance_deg(lon, lat, p_site->sitelon, p_site->sitelat);

  /* check the distance */
  if ((dist < p_phase->p_dists[0]) 
      || (dist > p_phase->p_dists[p_phase->numdist-1]))
    return 0;

  /* TODO: check for holes in the travel time tables */
  return 1;
}

double EarthModel_ArrivalTime(EarthModel_t * p_earth, double * evloc,
                              double evtime, int phaseid, int siteid)
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
