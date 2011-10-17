PyObject * py_propose(NetModel_t * p_netmodel, PyObject * args);

int propose_invert_step(NetModel_t * p_netmodel, SigModel_t * p_sigmodel,
			Event_t **pp_events,
                        double time_low, double time_high, int det_low,
                        int det_high, double degree_step, int num_step,
                        int numthreads);

void optimize_arrivals(SigModel_t * p_sigmodel,
		       Event_t * p_event,
		       int num_other_events,
		       const Event_t ** pp_other_events);
