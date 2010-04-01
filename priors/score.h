
double score_world(NetModel_t * p_netmodel,
                   int numevents, Event_t * p_events,
                   int verbose);

void score_event(NetModel_t * p_netmodel,
                 Event_t * p_event, double * p_locsc, double * p_magsc,
                 double * p_detsc, int * p_detcnt);

