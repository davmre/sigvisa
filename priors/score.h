
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

double logprob_false(NetModel_t * p_netmodel, int numdets, int * p_detids,
                     int verbose);

double logprob_noise(NetModel_t * p_netmodel, Detection_t * p_det,
                     Detection_t * p_prev_det);
double logprob_coda_coda(NetModel_t * p_netmodel, Detection_t * p_secdet,
                         Detection_t * p_det);

/* log probability that p_det generates the coda arrival p_secdet 
 * minus
 * log probability that p_det doesn't generate a secondary arrival and
 * p_secdet is a noise arrival
 */
double score_coda(NetModel_t * p_netmodel, Detection_t * p_secdet,
                  Detection_t * p_det);
