/*double score_world_sig(SigModel_t * p_sigmodel,
		       int numevents, Event_t * p_events,
		       PyArrayObject * arrtimes,
		       int verbose);*/

int score_event_sig(SigModel_t * p_sigmodel, Event_t * p_event, int num_other_events, const Event_t ** pp_other_events);


//double prob_event_sig(SigModel_t * p_sigmodel, Event_t * p_event);

double logprob_false_sig(SigModel_t * p_sigmodel, int numdets, int * p_detids,
                     int verbose);

