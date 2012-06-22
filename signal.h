#ifndef SIGVISA_SIGNAL_INCLUDE
#define SIGVISA_SIGNAL_INCLUDE

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

/* define signal channels */
#define NUM_CHANS   4
#define CHAN_BHE    0
#define CHAN_BHN    1
#define CHAN_BHZ    2
#define CHAN_HORIZ_AVG    3
#define CHAN_OTHER  4

#define NUM_BANDS   10

#define DEFAULT_BAND NARROW_20_30

#define BROADBAND       0
#define BB_ENVELOPE     1
#define NARROW_05_07    2
#define NARROW_07_10    3
#define NARROW_10_15    4
#define NARROW_15_20    5
#define NARROW_20_30    6
#define NARROW_30_40    7
#define NARROW_40_60    8
#define NARROW_60_80    9


/* parameters for specifying a signal envelope */
#define ARR_TIME_PARAM 0
#define PEAK_OFFSET_PARAM 1
#define PEAK_HEIGHT_PARAM 2
#define PEAK_DECAY_PARAM 3
#define CODA_HEIGHT_PARAM 4
#define CODA_DECAY_PARAM 5

#define MIN_LOGENV_CUTOFF -3
#define EXP_MIN_LOGENV_CUTOFF 0.0497870684
#define SAFE_LOG(x) (x > EXP_MIN_LOGENV_CUTOFF) ? log(x) : MIN_LOGENV_CUTOFF

#define DEFAULT_HZ 40

#define MIN_AMP 0.01
#define LOG_MIN_AMP -4.6052
#define LOG10_MIN_AMP -2
#define MAX_AMP 10000.0
#define LOG_MAX_AMP 9.2103404
#define LOG10_MAX_AMP 4.0

#define MAX_ENVELOPE_LENGTH 300


/* information about phases */
#define NUM_TD_PHASES 14



typedef struct Trace_t
{
  long len;
  double * p_data;
  PyArrayObject * py_array;   /*  we're forced to keep the Python
				 object around so that we can DECREF
				 it when finished */

  double start_time;
  double hz;
  int siteid;
  int chan;
  int band;

  double p_time;
  double s_time;
  int p_phaseid;
  int s_phaseid;
  double noise_floor;
} Trace_t;

typedef struct Channel_t
{

  long len;
  Trace_t * p_bands[NUM_BANDS];

  double start_time;
  double hz;

  int siteid;
  int chan;

} Channel_t;

typedef struct Segment_t {
  long len;

  double start_time;
  double hz;

  int siteid;

  Channel_t * p_channels[NUM_CHANS];
} Segment_t;


Trace_t * alloc_trace(void);
PyObject * build_trace(Trace_t * p_trace);
void free_trace(Trace_t * p_trace);

Channel_t * alloc_channel(Segment_t * p_segment);

void alloc_segment_inner(Segment_t * p_segment);
void free_segment_inner(Segment_t * p_segment);
double Segment_EndTime(Segment_t * b);

int canonical_channel_num(char* chan_str);

int canonical_band_num(char* band_str);
PyObject * canonical_band_name(int num);
PyObject* canonical_channel_name(int num);

PyObject * channel_bundle_to_trace_bundle(Segment_t * p_segment);
int signal_to_trace(Channel_t * p_signal, PyObject ** pp_trace);
int trace_to_signal(PyObject * py_trace, Trace_t ** pp_trace);

int py_segment_to_c_segment(PyObject * py_segment, Segment_t * p_segment);
PyObject * c_segment_to_py_segment(Segment_t * p_segment);

PyObject * py_canonical_channel_num(PyObject * self, PyObject * args);
PyObject * py_canonical_band_num(PyObject * self, PyObject * args);
#endif
