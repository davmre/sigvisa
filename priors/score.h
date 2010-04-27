
double score_world(NetModel_t * p_netmodel,
                   int numevents, Event_t * p_events,
                   int verbose);

double score_event(NetModel_t * p_netmodel, Event_t * p_event);

int score_event_site_phase(NetModel_t * p_netmodel,
                           const Event_t * p_event,
                           int siteid, int phaseid,
                           double distance, double pred_az,
                           double * p_score);

int score_event_site_phase_simple(NetModel_t * p_netmodel,
                                  const Event_t * p_event, int siteid,
                                  int phaseid, double * p_score);

double prob_event(NetModel_t * p_netmodel, Event_t * p_event);
