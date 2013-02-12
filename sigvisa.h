#ifndef SIGVISA_INCLUDE
#define SIGVISA_INCLUDE

#include <Python.h>
#include <math.h>
#include <float.h>

#if !defined SIGVISA_MAIN_MODULE
  #define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "numpy/arrayobject.h"


#undef NDEBUG
#include <assert.h>

#define TRUE 1
#define FALSE 0

#define Event2R3Vector(event, vector) do {\
(vector)[0] = (event)->evlon; (vector)[1] = (event)->evlat;\
(vector)[2] = (event)->evdepth;} while(0)

#define PI                     ((double) 3.1415926535897931)
#define DEG2RAD                ((double) (PI / 180))
#define RAD2DEG                ((double) (180 / PI))
#define AVG_EARTH_RADIUS_KM    ((double) 6371) /* when modeled as a sphere */
#define AVG_EARTH_CIRCUMFERENCE_KM ((double) 40008)

#define LAT2Z(lat) (sin((lat) * PI / 180.0))
#define Z2LAT(z) (asin(z) * 180.0 / PI)

#define ARRAY3(arr,i,j,k) (*((double *)PyArray_GETPTR3(arr,i,j,k)))
#define ARRAY2(arr,i,j) (*((double *)PyArray_GETPTR2(arr,i,j)))
#define ARRAY1(arr,i) (*((double *)PyArray_GETPTR1(arr,i)))

#define SETARRAY3(arr,i,j,k,v) (*((double *)PyArray_GETPTR3(arr,i,j,k)))
#define SETARRAY2(arr,i,j,v) (*((double *)PyArray_GETPTR2(arr,i,j)))
#define SETARRAY1(arr,i,v) (*((double *)PyArray_GETPTR1(arr,i)))

#define BOOLARRAY2(arr,i,j) (*((npy_bool *)PyArray_GETPTR2(arr,i,j)))
#define BOOLARRAY1(arr,i) (*((npy_bool *)PyArray_GETPTR1(arr,i)))

#define UPDATE_AND_VERIFY(a, b) if (*a == -1) *a = b; else assert(*a == b);

#define CHECK_ERROR if(PyErr_Occurred()) { PyErr_Print(); exit(1); }
#define CHECK_PTR(p) if (p == NULL) { LogFatal("memory allocation failed, or null pointer detected!"); exit(1);}
#define CHECK_FATAL(x) if(x < 0) { CHECK_ERROR; LogFatal("fatal error!"); exit(1);}

#define MIN(a,b) ((a) <= (b) ? (a) : (b))
#define MAX(a,b) ((a) >= (b) ? (a) : (b))
#define BOUND(x, low, high) MIN(high, MAX(x, low))

// given log(x) and log(y), returns log(x+y)
#define LOGSUM(logx,logy) (logx > logy) ? logx + log(1 + exp(logy-logx)) : logy + log(1 + exp(logx-logy))

/* RAND_DOUBLE -> random number between 0 and 1 (exclusive) */
#define RAND_DOUBLE ( ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0) )
/* RAND_UNIFORM(a,b) -> random value between a and b */
#define RAND_UNIFORM(a,b) (((double) (a)) + ((double) ((b)-(a))) * RAND_DOUBLE)

#define SPHERE2X(azi, incl) sin(DEG2RAD * incl)*cos(DEG2RAD * azi)
#define SPHERE2Y(azi, incl) sin(DEG2RAD * incl)*sin(DEG2RAD * azi)
#define SPHERE2Z(azi, incl) cos(DEG2RAD * incl)

/* DELTA_TIME and DELTA_DIST are used in evaluating the answer */
#define DELTA_TIME 50                        /* in seconds */
#define DELTA_DIST 5                         /* in degrees */

extern PyObject * traceClass_obj;

int pydict_get_double(PyObject * py_dict, char * key, double *value);
int pydict_get_int(PyObject * py_dict, char * key, long * value);
int pydict_get_string(PyObject * py_dict, char * key, char ** value);


#include "sigmodel.h"

#define MIN_AMP 0.001
#define LOG_MIN_AMP -6.90775527898
#define LOG10_MIN_AMP -3
#define MAX_AMP 100000.0
#define LOG_MAX_AMP 11.512925465
#define LOG10_MAX_AMP 5.0

#define MAX_ENVELOPE_LENGTH 300


/* information about phases */
#define NUM_TD_PHASES 14

void convert_tuple_int(PyObject * tuple,
			      int * p_len, int ** pp_ints);
#endif // SIGVISA_INCLUDE
