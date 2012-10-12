
#include <Python.h>

#include "../../sigvisa.h"


void Spectral_Envelope_Model_Init_Params(Spectral_Envelope_Model_t * p_params, int numsites);

void Spectral_Envelope_Model_Set_Params(Spectral_Envelope_Model_t * p_params, int siteid, PyObject * py_dict);

int Spectral_Envelope_Model_Has_Model(SigModel_t * p_sigmodel, int siteid, int chan);

double Spectral_Envelope_Model_Likelihood(SigModel_t * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals);

void Spectral_Envelope_Model_Sample_Trace(SigModel_t * p_sigmodel, Trace_t * p_trace, int num_arrivals, const Arrival_t ** pp_arrivals, int sample_noise, int sample_wiggles);
void Spectral_Envelope_Model_Sample(SigModel_t * p_sigmodel, Segment_t * p_segment, int num_arrivals, const Arrival_t ** pp_arrivals, int sample_noise, int sample_wiggles);

void Spectral_Envelope_Model_UnInit(Spectral_Envelope_Model_t * p_params);
