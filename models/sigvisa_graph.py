import time
import numpy as np
import os
from sigvisa import Sigvisa

from sigvisa.database.dataset import read_event_detections, DET_PHASE_COL
from sigvisa.database.signal_data import get_fitting_runid, insert_wiggle, ensure_dir_exists

from sigvisa.source.event import get_event
from sigvisa.signals.io import load_event_station_chan
import sigvisa.utils.geog as geog
from sigvisa.models.ev_prior import EventPriorModel
from sigvisa.models.ttime import tt_predict, tt_log_p
from sigvisa.models.graph import Node, DirectedGraphModel
from sigvisa.models.envelope_model import EnvelopeNode
from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.models.wiggles import load_wiggle_node, load_wiggle_node_by_family
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle import extract_phase_wiggle
from sigvisa.signals.common import Waveform

class SigvisaGraph(DirectedGraphModel):


    """
    Construct the Sigvisa graphical model.

    """

    def __init__(self, template_model_type="dummy", template_shape="paired_exp",
                 wiggle_model_type="dummy", wiggle_family="fourier_0.8",
                 wiggle_len_s = 30.0,
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
        self.wiggle_len_s = wiggle_len_s

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

        self.optim_log = ""

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


    def connect_ev_wave(self, event_node, wave_node, basisids=None, tmshapes=None):
        ev = event_node.get_value()
        wave = wave_node.mw

        s = Sigvisa()

        phases =  self.predict_phases(ev, wave['sta'])

        for phase in phases:

            lbl = self._get_interior_node_label(ev=ev, phase=phase, wave=wave)

            tm_shape = tmshapes[phase] if tmshapes is not None else self.template_shape
            tm_node = load_template_model(runid = self.runid, sta=wave['sta'], chan=wave['chan'], band=wave['band'], phase=phase, model_type = self.template_model_type, label="template_%s" % (lbl,), template_shape = tm_shape)
            tm_node.addParent(event_node)
            tm_node.addChild(wave_node)
            self.all_nodes[tm_node.label] = tm_node
            self.template_nodes.add(tm_node)
            tm_node.prior_predict()

            if basisids is not None:
                wm_basisid = basisids[phase]
                wm_node = load_wiggle_node(basisid = wm_basisid,
                                           wiggle_model_type = self.wiggle_model_type,
                                           model_waveform=wave,
                                           phase=phase,
                                           runid=self.runid,
                                           label="wiggle_%s" % (lbl,))
            else:
                wm_node = load_wiggle_node_by_family(family_name = self.wiggle_family,
                                                     wiggle_model_type = self.wiggle_model_type,
                                                     len_s = self.wiggle_len_s,
                                                     model_waveform = wave,
                                                     phase=phase,
                                                     runid=self.runid,
                                                     label="wiggle_%s" % (lbl,))

            wm_node.addParent(event_node)
            wm_node.addChild(wave_node)
            self.all_nodes[wm_node.label] = wm_node
            self.wiggle_nodes.add(wm_node)
            wm_node.prior_predict()

    def add_event(self, ev, basisids=None, tmshapes=None):
        """

        Add an event node to the graph and connect it to all waves
        during which its signals might arrive.

        basisids: optional dictionary of wiggle basis ids (integers), keyed on phase name.
        tmshapes: optional dictionary of template shapes (strings), keyed on phase name.

        """

        event_node = Node(model = self.ev_prior_model, fixed_value=True, initial_value=ev, label='ev_%d' % ev.id)
        self.toplevel_nodes.append(event_node)
        self.all_nodes[event_node.label] = event_node

        for wave_node in self.leaf_nodes:
            wave = wave_node.mw
            if not self.wave_captures_event(ev=ev, sta=wave['sta'], stime=wave['stime'], etime=wave['etime']):
                continue

            self.connect_ev_wave(event_node=event_node, wave_node=wave_node, basisids=basisids, tmshapes=tmshapes)

        self._topo_sort()
        return event_node

    def add_wave(self, wave, basisids=None, tmshapes=None):
        """

        Add a wave node to the graph and connect it to all events
        whose signals might arrive during that time period.

        basisids: optional dictionary of wiggle basis ids (integers), keyed on phase name.
        tmshapes: optional dictionary of template shapes (strings), keyed on phase name.

        """

        wave_node = EnvelopeNode(model_waveform=wave, nm_type=self.nm_type, observed=True, label=self._get_wave_label(wave=wave))
        self.leaf_nodes.append(wave_node)
        self.all_nodes[wave_node.label] = wave_node

        for event_node in self.toplevel_nodes:
            wave = wave_node.mw
            if not self.wave_captures_event(ev=event_node.get_value(), sta=wave['sta'],
                                            stime=wave['stime'], etime=wave['etime']):
                continue
            # TODO: need a smarter way of specifying per-event phases
            self.connect_ev_wave(event_node=event_node, wave_node=wave_node, basisids=basisids, tmshapes=tmshapes)

        self._topo_sort()
        return wave_node

    def _get_interior_node_label(self, ev, phase, wave=None, sta=None, chan=None, band=None):
        if wave is not None:
            sta = wave['sta']
            chan = wave['chan']
            band = wave['band']
        return "%d_%s_%s_%s_%s" % (ev.id, phase, sta, chan, band)

    def _get_wave_label(self, wave):
        return 'wave_%s_%s_%s_%.1f' % (wave['sta'], wave['chan'], wave['band'], wave['stime'])

    def get_template_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["template_%s" % lbl]

    def get_wiggle_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["wiggle_%s" % lbl]

    def get_wave_node(self, wave):
        return self.all_nodes[self._get_wave_label(wave=wave)]

    def get_partner_node(self, n):
        if n.label.startswith("template_"):
            lbl = n.label[9:]
            return self.all_nodes["wiggle_%s" % lbl]
        elif n.label.startswith("wiggle_"):
            lbl = n.label[7:]
            return self.all_nodes["template_%s" % lbl]
        else:
            raise ValueError("node %s has no partner!" % n.label)

    def fix_arrival_times(self, fixed=True):
        for tm_node in self.template_nodes:
            tm_node.fix_arrival_time(fixed=fixed)


    def init_wiggles_from_template(self):
        wave_node = self.leaf_nodes[0]
        for wm_node in self.wiggle_nodes:
            tm_node = self.get_partner_node(wm_node)
            atime = tm_node.get_value()[0]

            wiggle, st, et = extract_phase_wiggle(tm_node=tm_node, wave_node=wave_node)
            if st is None or len(wiggle) < wm_node.npts:
                continue

            wiggle = wiggle[:wm_node.npts].filled(1)
            wm_node.set_params_from_wiggle(wiggle)

        ll = self.current_log_p()
        self.optim_log += ("init_wiggles: ll=%.1f\n" % ll)

    def optimize_wiggles(self, optim_params):
        st = time.time()
        self.joint_optimize_nodes(node_list=self.wiggle_nodes, optim_params=optim_params)
        et = time.time()
        ll = self.current_log_p()
        self.optim_log += ("optimize_wiggles: t=%.1fs ll=%.1f\n" % (et-st, ll))
        return ll

    def optimize_templates(self, optim_params):
        st = time.time()
        self.joint_optimize_nodes(node_list=self.template_nodes, optim_params=optim_params)
        et = time.time()
        ll = self.current_log_p()
        self.optim_log += ("optimize_templates: t=%.1fs ll=%.1f\n" % (et-st, ll))
        return ll

    def save_wiggle_params(self):
        """
        Saves the current values of the wiggle model parameters for
        all waves in the graph. Assumes template parameters have
        already been saved, so tm.fpid is available at each template
        node.
        """

        s = Sigvisa()
        for wave_node in self.leaf_nodes:
            wave = wave_node.mw
            event = get_event(evid=wave['evid'])
            parent_wiggles = [wm for wm in wave_node.parents.values() if wm.label.startswith("wiggle_")]
            for wm in parent_wiggles:
                try:
                    lbl = wm.label[7:]
                    tm = wave_node.parents['template_%s' % lbl]
                    fpid = tm.fpid
                except KeyError:
                    continue

                param_blob = wm.get_encoded_params()
                p = {"fpid": fpid, 'timestamp': time.time(),  "params": param_blob, 'basisid': wm.basisid}
                wiggleid = insert_wiggle(s.dbconn, p)
                wm.wiggleid = wiggleid

            s.dbconn.commit()

    def save_template_params(self, tmpl_optim_param_str,
                             wiggle_optim_param_str,
                             hz, elapsed):
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        runid = self.runid
        fitids = []

        wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
        run_wiggle_dir = os.path.join(wiggle_dir, "runid_" + str(self.runid))
        ensure_dir_exists(run_wiggle_dir)


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

            tmpl_optim_param_str = tmpl_optim_param_str.replace("'", "''")
            wiggle_optim_param_str = wiggle_optim_param_str.replace("'", "''")
            optim_log = wiggle_optim_param_str.replace("\n", "\\\\n")

            sql_query = "INSERT INTO sigvisa_coda_fit (runid, evid, sta, chan, band, tmpl_optim_method, wiggle_optim_method, optim_log, iid, stime, etime, hz, acost, dist, azi, timestamp, elapsed, nmid) values (%d, %d, '%s', '%s', '%s', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f, %d)" % (runid, event.evid, sta, chan, band, tmpl_optim_param_str, wiggle_optim_param_str, self.optim_log, 1 if wave_node.nm_type != 'ar' else 0, st, et, hz, wave_node.log_p(), distance, azimuth, time.time(), elapsed, wave_node.nmid)

            fitid = execute_and_return_id(s.dbconn, sql_query, "fitid")

            parent_templates = [tm for tm in wave_node.parents.values() if tm.label.startswith("template_")]

            for tm in parent_templates:
                fit_params = tm.get_value()
                transfer = fit_params[2] - event.source_logamp(band, tm.phase)

                phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, param1, param2, param3, param4, amp_transfer) values (%d, '%s', 'paired_exp', %f, %f, %f, %f, %f)" % (
                    fitid, tm.phase, fit_params[0], fit_params[1], fit_params[2], fit_params[3], transfer)
                fpid = execute_and_return_id(s.dbconn, phase_insert_query, "fpid")
                tm.fpid = fpid

                # extract the empirical wiggle for this template fit
                wiggle, wiggle_st, wiggle_et = extract_phase_wiggle(tm_node=tm, wave_node=wave_node)
                if len(wiggle) < wave['srate']:
                    print "evid %d phase %s at %s (%s, %s) is not prominent enough to extract a wiggle..." % (event.evid, tm.phase, sta, chan, band)
                    wiggle_fname = "NONE"
                    wiggle_st = -1
                else:
                    wiggle_wave = Waveform(data=wiggle, srate=wave['srate'], stime=wiggle_st, sta=sta, chan=chan, evid=event.evid)
                    wiggle_fname = os.path.join("runid_" + str(self.runid), "%d.wave" % (fpid,))
                    wiggle_wave.dump_to_file(os.path.join(wiggle_dir, wiggle_fname))
                sql_query = "update sigvisa_coda_fit_phase set wiggle_fname='%s', wiggle_stime=%f where fpid=%d" % (wiggle_fname, wiggle_st, fpid,)
                cursor.execute(sql_query)


            fitids.append(fitid)
            s.dbconn.commit()
        cursor.close()

        return fitids


def load_sg_from_db_fit(fitid, load_wiggles=True):

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    fit_sql_query = "select f.runid, f.evid, f.sta, f.chan, f.band, f.hz, f.stime, f.etime, nm.model_type from sigvisa_coda_fit f, sigvisa_noise_model nm where f.fitid=%d and f.nmid=nm.nmid" % (fitid)
    cursor.execute(fit_sql_query)
    fit = cursor.fetchone()
    ev = get_event(evid=fit[1])
    wave = load_event_station_chan(fit[1], fit[2], fit[3], cursor=cursor).filter('%s;env;hz_%.2f' % (fit[4], fit[5]))
    nm_type = fit[8]
    runid = fit[0]

    phase_sql_query = "select fpid, phase, template_model, param1, param2, param3, param4 from sigvisa_coda_fit_phase where fitid=%d" % fitid
    cursor.execute(phase_sql_query)
    phase_details = cursor.fetchall()
    phases = [p[1] for p in phase_details]
    templates = {}
    tmshapes = {}
    for (phase, p) in zip(phases, phase_details):
        templates[phase] = p[3:7]
        tmshapes[phase] = p[2]


    if load_wiggles:
        wiggle_family = None
        wiggles = {}
        basisids = {}
        wiggle_family = None
        for (phase, phase_detail) in zip(phases, phase_details):
            wiggle_sql_query = "select w.wiggleid, w.params, w.basisid from sigvisa_wiggle w where w.fpid=%d " % (phase_detail[0])
            cursor.execute(wiggle_sql_query)
            w = cursor.fetchall()
            assert(len(w) == 1) # if there's more than one wiggle
                                # parameterization of a phase, we'd need
                                # some way to disambiguate.

            basisids[phase] = w[0][2]
            wiggles[phase] = w[0][1]
    else:
        wiggle_family = "fourier_0.1"
        basisids = None

    sg = SigvisaGraph(template_model_type="dummy", wiggle_model_type="dummy",
                      template_shape=None, wiggle_family=wiggle_family,
                      nm_type = nm_type, runid=runid, phases=phases)
    sg.add_event(ev)
    wave_node = sg.add_wave(wave, basisids=basisids, tmshapes=tmshapes)

    for phase in phases:
        tm_node = sg.get_template_node(ev=ev, phase=phase, wave=wave)
        wm_node = sg.get_wiggle_node(ev=ev, phase=phase, wave=wave)

        tm_node.set_value(templates[phase])
        if load_wiggles:
            wm_node.set_encoded_params(wiggles[phase])

    return sg
