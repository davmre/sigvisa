import numpy as np

from sigvisa import Sigvisa

from sigvisa.database.dataset import read_event_detections, DET_PHASE_COL
from sigvisa.database.signal_data import get_fitting_runid

from sigvisa.models.ev_prior import EventPriorModel
from sigvisa.models.ttime import tt_predict, tt_log_p
from sigvisa.models.graph import Node, DirectedGraphModel
from sigvisa.models.envelope_model import EnvelopeNode
from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.models.wiggles.wiggle_models import get_wiggle_param_model_ids

class SigvisaGraph(DirectedGraphModel):


    """
    Construct the Sigvisa graphical model.

    """

    def __init__(self, template_model_type, template_shape,
                 wiggle_model_type, wiggle_model_basis,
                 phases, nm_type, run_name, iteration):
        """

        phases: controls which phases are modeled for each event/sta pair
                "auto": model all phases for which we have travel-time predictions from an event to station.
                "leb": model all phases whose arrival is recorded in the LEB. (works only on LEB training data)
                [list of phase names]: model a fixed set of phases
        """

        super(DirectedGraphModel, self).__init__()

        self.ev_prior_model = EventPriorModel()

        self.template_model_type = template_model_type
        self.template_shape = template_shape

        self.wiggle_model_type = wiggle_model_type
        self.wiggle_model_basis = wiggle_model_basis

        self.nm_type = nm_type
        self.phases = phases

        cursor = Sigvisa().dbconn.cursor()
        self.runid = get_fitting_runid(cursor, run_name, iteration, create_if_new = True)
        cursor.close()

        self.template_nodes = []
        self.wiggle_nodes = []

    def predict_phases(self, ev, sta):
        s = Sigvisa()
        if self.phases == "leb":
            cursor = s.dbconn.cursor()
            phases = [s.phasenames[id_minus1] for id_minus1 in read_event_detections(cursor=cursor, evid=ev.evid, stations=[sta, ], evtype="leb")[:,DET_PHASE_COL]]
            cursor.close()
        elif self.phases == "auto":
            phases = s.arriving_phases(event=ev, sta=sta)
        else:
            phases = self.phases
        return phases

    def wave_captures_event(self, ev, sta, stime, etime):
        for phase in self.predict_phases(ev, sta):
            if self.wave_captures_event_phase(ev, sta, stime, etime, phase):
                return True
        return False

    def wave_captures_event_phase(self, ev, sta, stime, etime, phase):
        """

        Check whether a particular waveform might be expected to contain a signal from a particular event.

        """

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

    def connect_ev_wave(self, event_node, wave_node):
        ev = event_node.get_value()
        wave = wave_node.mw

        s = Sigvisa()

        phases =  self.predict_phases(ev, wave['sta'])

        for phase in phases:

            tm_node = load_template_model(runid = self.runid, sta=wave['sta'], chan=wave['chan'], band=wave['band'], phase=phase, model_type = self.template_model_type, label="template_%s_%s" % (ev.id, phase), template_shape = self.template_shape)
            tm_node.addParent(event_node)
            event_node.addChild(tm_node)
            wave_node.addParent(tm_node)
            tm_node.addChild(wave_node)
            self.template_nodes.append(tm_node)

            wpmids = get_wiggle_param_model_ids(runid = self.runid, sta=wave['sta'], chan=wave['chan'], band=wave['band'], phase=phase, model_type = self.wiggle_model_type, basisid = self.wiggle_model_basis)
            wm_node = WiggleModelNode(label="wiggle_%s_%s" % (ev.id, phase), wiggle_param_model_ids=wpmids)
            wm_node.addParent(event_node)
            event_node.addChild(wm_node)
            wave_node.addParent(wm_node)
            self.wiggle_nodes.append(wm_node)

    def add_event(self, ev):
        event_node = Node(model = self.ev_prior_model, fixed_value=True, initial_value=ev)
        self.toplevel_nodes.append(event_node)

        for wave_node in self.leaf_nodes:
            wave = wave_node.mw
            if not self.wave_captures_event(ev=ev, sta=wave['sta'], stime=wave['stime'], etime=wave['etime']):
                continue

            self.connect_ev_wave(event_node=event_node, wave_node=wave_node)

        self._topo_sort()

    def add_wave(self, wave):
        wave_node = EnvelopeNode(model_waveform=wave, nm_type=self.nm_type, observed=True)
        self.leaf_nodes.append(wave_node)

        for event_node in self.toplevel_nodes:
            wave = wave_node.mw
            if not self.wave_captures_event(ev=event_node.get_value(), sta=wave['sta'],
                                            stime=wave['stime'], etime=wave['etime']):
                continue
            # TODO: need a smarter way of specifying per-event phases
            self.connect_ev_wave(event_node=event_node, wave_node=wave_node)

        self._topo_sort()


    def save_template_params(self, optim_param_str, hz, run_name, iteration, elapsed):
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        runid = get_fitting_runid(cursor, run_name, iteration)
        fitids = []

        for wave_node in self.leaf_nodes:
            wave = wave_node.get_value()

            sta = wave['sta']
            siteid = wave['siteid']
            chan = wave['chan']
            band = wave['band']
            st = wave['stime']
            et = wave['etime']
            event = get_event(evid=wave['evid'])

            distance = geog.dist_km((event.lon, event.lat), (s.sites[siteid - 1][0], s.sites[siteid - 1][1]))
            azimuth = geog.azimuth((s.sites[siteid - 1][0], s.sites[siteid - 1][1]), (event.lon, event.lat))

            optim_param_str = optim_param_str.replace("'", "''")

            sql_query = "INSERT INTO sigvisa_coda_fit (runid, evid, sta, chan, band, optim_method, iid, stime, etime, hz, acost, dist, azi, timestamp, elapsed, nmid) values (%d, %d, '%s', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f, %d)" % (runid, event.evid, sta, chan, band, optim_param_str, 1 if wave_node.nm_type != 'ar' else 0, st, et, hz, wave_node.log_p(), distance, azimuth, time.time(), elapsed, wave_node.nmid)

            fitid = execute_and_return_id(s.dbconn, sql_query, "fitid")

            parent_templates = [tm for tm in wave_node.parents.values() if tm.label.startswith("template_")]

            for tm in parent_templates:

                fit_params = tm.get_value()

                transfer = fit_params[2] - event.source_logamp(band, tm.phase)

                phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, param1, param2, param3, param4, amp_transfer) values (%d, '%s', 'paired_exp', %f, %f, %f, %f, %f)" % (
                    fitid, phase, fit_params[0], fit_params[1], fit_params[2], fit_params[3], transfer)
                cursor.execute(phase_insert_query)


            cursor.close()
            fitids.append(fitid)

        return fitids
