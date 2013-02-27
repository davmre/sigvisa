from sigvisa.models.graph import DirectedGraphModel
from sigviasa.models.ttime import tt_predict, tt_log_p


class SigvisaGraph(DirectedGraphModel):


    """
    Construct the Sigvisa graphical model.

    """

    def __init__(self, template_model_type, template_shape, wiggle_model_type, wiggle_model_basis):
        super(DirectedGraphModel, self).__init__()

        self.ev_prior_model = EventPriorModel()
        self.ttmodel = TravelTimeModel()

        self.template_model_type = template_model_type
        self.template_shape = template_shape
        self.wiggle_model_type = wiggle_model_type
        self.wiggle_model_basis = wiggle_model_basis



    def wave_captures_event(self, ev, wave, phase):
        """

        Check whether a particular waveform might be expected to contain a signal from a particular event.

        """
        stime = wave['stime']
        sta = wave['sta']
        etime = wave['etime']

        TT_PROB_THRESHOLD = 1e-5
        MAX_DECAY_LEN_S = 400

        captures = True
        predicted_atime = ev.time + tt_predict(event=ev, sta=sta, phase=phase)

        # if the predicted arrival time precedes the waveform start by
        # more than the maximum decay length, and the tail probability
        # is small, then we can confidently say no.
        if predicted_atime + MAX_DECAY_LEN_S < stime and tt_log_p(x = stime - MAX_DECAY_LEN_S - ev.time,
                                                          event=ev, sta=sta, phase=phase) < TT_PROB_THRESHOLD:
            captures = False

        # similarly if the predicted arrival time is after the
        # waveform end time, and the tail probability is small.
        elif predicted_atime > etime and tt_log_p(x = etime-ev.time,
                                                  event=ev, sta=sta, phase=phase) < TT_PROB_THRESHOLD:
            captures = False

        return captures

    def connect_ev_wave(self, event_node, wave_node, phases):
        ev = event_node.get_value()
        wave = wave_node.get_value()


        for phase in phases:

            tm_node = load_template_model(runid = self.runid, sta=wave['sta'], chan=wave['chan'], band=wave['band'], phase=phase, model_type = self.template_model_type, label="template_%s_%s" % (ev.id, phase), template_shape = self.template_shape)
            tm_node.addParent(event_node)
            event_node.addChild(tm_node)
            wave_node.addParent(tm_node)
            tm_node.addChild(wave_node)


            wpmids = get_wiggle_param_model_ids(runid = self.runid, sta=wave['sta'], chan=wave['chan'], band=wave['band'], phase=phase, model_type = self.wiggle_model_type, basisid = self.wiggle_model_basis)
            wm_node = WiggleModelNode(label="wiggle_%s_%s" % (ev.id, phase), wiggle_param_model_ids=wpmids)
            wm_node.addParent(event_node)
            event_node.addChild(wm_node)
            wave_node.addParent(wm_node)

    def add_event(self, ev, phases=None):
        if phases=None:
            phases = self.phases

        event_node = Node(model = self.ev_prior_model, fixed_value=True)
        event_node.phases = phases

        self.toplevel_nodes.append(event_node)

        for wave_node in self.leaf_nodes:
            wave = w.get_value()
            if not self.wave_captures_event(ev=ev, wave=wave):
                continue
            self.connect_ev_wave(ev_node=event_node, wave_node=wave_node, phases=phases)

        self.__topo_sort()

    def add_wave(self, wave):
        wave_node = Node(model = EnvelopeModel(model_waveform=wave), fixed_value=True)
        self.leaf_nodes.append(wave_node)

        for event_node in self.toplevel_nodes:
            wave = w.get_value()
            if not self.wave_captures_event(ev=ev, wave=wave):
                continue
            # TODO: need a smarter way of specifying per-event phases
            self.connect_ev_wave(ev_node=event_node, wave_node=wave_node, phases=self.phases)

        self.__topo_sort()

    def signal_lprob_point(self):
        """

        Joint probability of events, templates, wiggles, and noise,
        when templates and wiggles are set to their prior mean (or
        other predictive value) conditioned on its paretns.

        """

        self.prior_predict_all()
        return self.current_log_p()

    def signal_lprob_optimize(self, optim_params):
        self.optimize_bottom_up()
        return self.current_log_p()
