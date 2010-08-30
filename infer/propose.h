PyObject * py_propose(NetModel_t * p_netmodel, PyObject * args);

int propose_hough(NetModel_t * p_netmodel, Event_t **pp_events,
                  double time_low, double time_high, int det_low,
                  int det_high, double degree_step, double time_step);


int propose_invert(NetModel_t * p_netmodel, Event_t **pp_events,
                   double time_low, double time_high, int det_low,
                   int det_high, double degree_step, int num_step);
