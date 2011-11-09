/*double score_world_sig(SigModel_t * p_sigmodel,
		       int numevents, Event_t * p_events,
		       PyArrayObject * arrtimes,
		       int verbose);*/
double score_event_sta_sig(SigModel_t * p_sigmodel, const Event_t * p_event, int siteid, int num_other_events, const Event_t ** pp_other_events);

double score_event_evprior(SigModel_t * p_sigmodel, Event_t * p_event);
double score_event_prior(SigModel_t * p_sigmodel, Event_t * p_event);

int score_event_sig(SigModel_t * p_sigmodel, Event_t * p_event, int num_other_events, const Event_t ** pp_other_events);


//double prob_event_sig(SigModel_t * p_sigmodel, Event_t * p_event);

double logprob_false_sig(SigModel_t * p_sigmodel, int numdets, int * p_detids,
                     int verbose);

