double score_world_sig(SigModel_t * p_sigmodel,
		       int numevents, Event_t * p_events,
		       PyArrayObject * arrtimes,
		       int verbose);

double score_event_sig(SigModel_t * p_sigmodel, Event_t * p_event);

int score_event_site_phase_sig(SigModel_t * p_sigmodel,
                           const Event_t * p_event,
                           int siteid, int phaseid,
                           double distance, double pred_az,
                           double * p_score);

int score_event_site_phase_simple_sig(SigModel_t * p_sigmodel,
                                  const Event_t * p_event, int siteid,
                                  int phaseid, double * p_score);

double prob_event_sig(SigModel_t * p_sigmodel, Event_t * p_event);

double logprob_false_sig(SigModel_t * p_sigmodel, int numdets, int * p_detids,
                     int verbose);

