import time
import numpy as np

from sigvisa import Sigvisa

from sigvisa.database.dataset import read_event_detections, DET_PHASE_COL
from sigvisa.database.signal_data import get_fitting_runid

from sigvisa.source.event import get_event
import sigvisa.utils.geog as geog
from sigvisa.models.ev_prior import EventPriorModel
from sigvisa.models.ttime import tt_predict, tt_log_p
from sigvisa.models.graph import Node, DirectedGraphModel
from sigvisa.models.envelope_model import EnvelopeNode
from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.models.wiggles.wiggle_models import WiggleModelNode
from sigvisa.database.signal_data import execute_and_return_id

class SigvisaGraph(DirectedGraphModel):


    """
    Construct the Sigvisa graphical model.

    """

    def __init__(self, template_model_type="dummy", template_shape="paired_exp",
                 wiggle_model_type="dummy", wiggle_family="fourier_0.01",
                 nm_type="ar", run_name=None, iteration=None,
                 runid = None, phases="auto"):
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
        self.wiggle_family = wiggle_family

        self.nm_type = nm_type
        self.phases = phases

        self.runid = runid
        if run_name is not None and iteration is not None:
            cursor = Sigvisa().dbconn.cursor()
            self.runid = get_fitting_runid(cursor, run_name, iteration, create_if_new = True)
            cursor.close()

        self.all_nodes = {}
        self.template_nodes = set()
        self.wiggle_nodes = set()

    def topo_sorted_nodes(self):
        assert(len(self._topo_sorted_list) == len(self.all_nodes))
        return self._topo_sorted_list


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

    def _get_interior_node_label(self, ev, phase, wave=None, sta=None, chan=None, band=None):
        if wave is not None:
            sta = wave['sta']
            chan = wave['chan']
            band = wave['band']
        return "%d_%s_%s_%s_%s" % (ev.id, phase, sta, chan, band)

    def _get_wave_label(self, wave):
        return 'wave_%s_%s_%s_%.1f' % (wave['sta'], wave['chan'], wave['band'], wave['stime'])

    def connect_ev_wave(self, event_node, wave_node):
        ev = event_node.get_value()
        wave = wave_node.mw

        s = Sigvisa()

        phases =  self.predict_phases(ev, wave['sta'])

        for phase in phases:

            lbl = self._get_interior_node_label(ev=ev, phase=phase, wave=wave)

            tm_node = load_template_model(runid = self.runid, sta=wave['sta'], chan=wave['chan'], band=wave['band'], phase=phase, model_type = self.template_model_type, label="template_%s" % (lbl,), template_shape = self.template_shape)
            tm_node.addParent(event_node)
            tm_node.addChild(wave_node)
            self.all_nodes[tm_node.label] = tm_node
            self.template_nodes.add(tm_node)
            tm_node.prior_predict()

            wm_node = WiggleModelNode(label="wiggle_%s" % (lbl,), basis_family=self.wiggle_family, wiggle_model_type = self.wiggle_model_type, model_waveform=wave, phase=phase, runid=self.runid)
            wm_node.addParent(event_node)
            wm_node.addChild(wave_node)
            self.all_nodes[wm_node.label] = wm_node
            self.wiggle_nodes.add(wm_node)
            wm_node.prior_predict()

    def add_event(self, ev):
        event_node = Node(model = self.ev_prior_model, fixed_value=True, initial_value=ev, label='ev_%d' % ev.id)
        self.toplevel_nodes.append(event_node)
        self.all_nodes[event_node.label] = event_node

        for wave_node in self.leaf_nodes:
            wave = wave_node.mw
            if not self.wave_captures_event(ev=ev, sta=wave['sta'], stime=wave['stime'], etime=wave['etime']):
                continue

            self.connect_ev_wave(event_node=event_node, wave_node=wave_node)

        self._topo_sort()
        return event_node

    def add_wave(self, wave):
        wave_node = EnvelopeNode(model_waveform=wave, nm_type=self.nm_type, observed=True, label=self._get_wave_label(wave=wave))
        self.leaf_nodes.append(wave_node)
        self.all_nodes[wave_node.label] = wave_node

        for event_node in self.toplevel_nodes:
            wave = wave_node.mw
            if not self.wave_captures_event(ev=event_node.get_value(), sta=wave['sta'],
                                            stime=wave['stime'], etime=wave['etime']):
                continue
            # TODO: need a smarter way of specifying per-event phases
            self.connect_ev_wave(event_node=event_node, wave_node=wave_node)

        self._topo_sort()
        return wave_node

    def get_template_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["template_%s" % lbl]

    def get_wiggle_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["wiggle_%s" % lbl]

    def get_wave_node(self, wave):
        return self.all_nodes[self._get_wave_label(wave=wave)]

    def fix_arrival_times(self, fixed=True):
        for tm_node in self.template_nodes:
            tm_node.fix_arrival_time(fixed=fixed)

    def save_template_params(self, optim_param_str, hz, run_name, iteration, elapsed):
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        runid = get_fitting_runid(cursor, run_name, iteration)
        fitids = []

        for wave_node in self.leaf_nodes:
            wave = wave_node.mw

            sta = wave['sta']
            siteid = wave['siteid']
            chan = wave['chan']
            band = wave['band']
            st = wave['stime']
            et = wave['etime']
            event = get_event(evid=wave['evid'])

            parent_templates = [tm for tm in wave_node.parents.values() if tm.label.startswith("template_")]

            # make sure we can calculate source amplitude for all
            # relevant phases, so we don't get empty DB entries from
            # failing later on.
            for tm in parent_templates:
                try:
                    event.source_logamp(band, tm.phase)
                except:
                    raise



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
                    fitid, tm.phase, fit_params[0], fit_params[1], fit_params[2], fit_params[3], transfer)
                cursor.execute(phase_insert_query)


            cursor.close()
            fitids.append(fitid)

        return fitids
