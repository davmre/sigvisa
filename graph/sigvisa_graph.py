import time
import numpy as np
import os
import re
from sigvisa import Sigvisa

from sigvisa.database.signal_data import get_fitting_runid, insert_wiggle, ensure_dir_exists

from sigvisa.source.event import get_event
from sigvisa.learn.train_param_common import load_model
import sigvisa.utils.geog as geog
from sigvisa.models import DummyModel
from sigvisa.models.ev_prior import EventNode
from sigvisa.models.ttime import tt_predict, tt_log_p, ArrivalTimeNode
from sigvisa.graph.nodes import Node
from sigvisa.graph.dag import DirectedGraphModel
from sigvisa.graph.graph_utils import extract_sta_node, predict_phases, create_key, get_parent_value
from sigvisa.models.signal_model import ObservedSignalNode
from sigvisa.models.templates.load_by_name import load_template_generator
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle import extract_phase_wiggle
from sigvisa.models.wiggles import load_wiggle_generator_by_family
from sigvisa.signals.common import Waveform

class ModelNotFoundError(Exception):
    pass



def get_param_model_id(runid, sta, phase, model_type, param,
                       template_shape, basisid=None, chan=None, band=None):
    # get a DB modelid for a previously-trained parameter model
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    chan_cond = "and chan='%s'" % chan if chan else ""
    band_cond = "and band='%s'" % band if band else ""
    basisid_cond = "and wiggle_basisid='%d'" if basisid else ""

    sql_query = "select modelid from sigvisa_param_model where model_type = '%s' and site='%s' %s %s and phase='%s' and fitting_runid=%d and template_shape='%s' and param='%s' %s" % (model_type, sta, chan_cond, band_cond, phase, runid, template_shape, param, basisid_cond)
    try:
        cursor.execute(sql_query)
        modelid = cursor.fetchone()[0]
    except:
        raise ModelNotFoundError("no model found matching model_type = '%s' and site='%s' %s %s and phase='%s' and fitting_runid=%d and template_shape='%s' and param='%s'" % (model_type, sta, chan_cond, band_cond, phase, runid, template_shape, param))
    finally:
        cursor.close()
    return modelid


class SigvisaGraph(DirectedGraphModel):


    """
    Construct the Sigvisa graphical model.

    """

    def _tm_type(self, param):
        try:
            return self.template_model_type[param]
        except TypeError:
            return self.template_model_type

    def __init__(self, template_model_type="dummy", template_shape="paired_exp",
                 wiggle_model_type="dummy", wiggle_family="fourier_0.8",
                 wiggle_len_s = 30.0, dummy_fallback=False,
                 nm_type="ar", run_name=None, iteration=None,
                 runid = None, phases="auto", base_srate=40.0):
        """

        phases: controls which phases are modeled for each event/sta pair
                "auto": model all phases for which we have travel-time predictions from an event to station.
                "leb": model all phases whose arrival is recorded in the LEB. (works only on LEB training data)
                [list of phase names]: model a fixed set of phases
        """

        super(SigvisaGraph, self).__init__()

        self.template_model_type = template_model_type
        self.template_shape = template_shape
        self.tg = load_template_generator(self.template_shape)

        self.wiggle_model_type = wiggle_model_type
        self.wiggle_family = wiggle_family
        self.wiggle_len_s = wiggle_len_s
        self.wgs = dict()
        self.base_srate = base_srate

        self.dummy_fallback = dummy_fallback

        self.nm_type = nm_type
        self.phases = phases

        self.runid = runid
        if run_name is not None and iteration is not None:
            cursor = Sigvisa().dbconn.cursor()
            self.runid = get_fitting_runid(cursor, run_name, iteration, create_if_new = True)
            cursor.close()

        self.template_nodes = []

        self.station_waves = dict()
        self.site_elements = dict()
        self.site_bands = dict()
        self.site_chans = dict()

        self.optim_log = ""

    def template_generator(self, phase):
        return self.tg

    def wiggle_generator(self, phase, srate):
        if srate not in self.wgs:
            self.wgs[srate] =  load_wiggle_generator_by_family(family_name=self.wiggle_family, len_s=self.wiggle_len_s, srate=srate)
        return self.wgs[srate]

    def set_template(self, eid, sta, phase, band, chan, values):
        for (param, value) in values.items():
            if param == "arrival_time":
                b = None
                c = None
            else:
                b = band
                c = chan
            self.set_value(key=create_key(param=param, eid=eid,
                                          sta=sta, phase=phase,
                                          band=b, chan=c),
                           value=value)

    def wave_captures_event(self, ev, sta, stime, etime):
        for phase in predict_phases(ev=ev, sta=sta, phases=self.phases):
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

    def add_node(self, node):
        if type(node) != EventNode and type(node) != ObservedSignalNode:
            self.template_nodes.append(node)

        super(SigvisaGraph, self).add_node(node)

    def add_event(self, ev, basisids=None, tmshapes=None):
        """

        Add an event node to the graph and connect it to all waves
        during which its signals might arrive.

        basisids: optional dictionary of wiggle basis ids (integers), keyed on phase name.
        tmshapes: optional dictionary of template shapes (strings), keyed on phase name.

        """

        event_node = EventNode(event=ev, label='ev_%d' % ev.eid, fixed=True)
        self.toplevel_nodes.append(event_node)
        self.add_node(event_node)

        for (site, element_list) in self.site_elements.iteritems():
            for phase in predict_phases(ev=ev, sta=site, phases=self.phases):
                tg = self.template_generator(phase)
                wg = self.wiggle_generator(phase, self.base_srate)
                self.add_event_site_phase(tg, wg, site, phase, event_node)

        self._topo_sort()
        return event_node

    def load_node_from_modelid(self, modelid, label):
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        sql_query = "select param, model_type, model_fname from sigvisa_param_model where modelid=%d" % (modelid)
        cursor.execute(sql_query)
        param, db_model_type, fname = cursor.fetchone()
        basedir = os.getenv("SIGVISA_HOME")
        model = load_model(os.path.join(basedir, fname), db_model_type)
        node = Node(model=model, label=label)
        node.modelid = modelid
        cursor.close()
        return node

    def create_unassociated_template(self, tg, wg, wave_node, atime, phase="XX", eid=-1):
        unassociated_templateid = self.next_uatemplateid
        self.next_uatemplateid += 1

        nodes = dict()
        at_label = create_key(param="arrival_time", sta=wave_node.sta,
                           phase=phase, eid=eid,
                           chan=wave_node.chan, band=wave_node.band)

        nodes['arrival_time'] = Node(label=at_label, model=DummyModel(),
                                     initial_value=atime, children=(wave_node,))

        for param in tg.params():
            label = create_key(param=param, sta=wave_node.sta,
                               phase=phase, eid=eid,
                               chan=wave_node.chan, band=wave_node.band)
            model = tg.unassociated_model(param)
            nodes[param] = Node(label=label, model=model, children=(wave_node,))
        for param in wg.params():
            label = create_key(param=param, sta=wave_node.sta,
                               phase=phase, eid=eid,
                               chan=wave_node.chan, band=wave_node.band)
            model = wg.unassociated_model(param)
            nodes[param] = Node(label=label, model=model, children=(wave_node,))

        for node in nodes.values():
            nodes.unassociated_templateid = unassociated_templateid
            self.add_node(node)

        return nodes

    def setup_site_param_node(self, param, site, phase, parent, chan=None, band=None, basisid=None, model_type=None, modelid=None):

        if model_type is None:
            model_type = self._tm_type(param)

        # for each station at this site, create a node with the
        # appropriate parameter model.
        nodes = dict()
        for sta in self.site_elements[site]:
            if model_type != "dummy" and modelid is None:
                try:
                    modelid = get_param_model_id(runid=self.runid, sta=site,
                                                 phase=phase, model_type=model_type,
                                                 param=param, template_shape=self.template_shape,
                                                 chan=chan, band=band, basisid=basisid)
                except ModelNotFoundError:
                    if self.dummy_fallback:
                        print "warning: falling back to dummy model for %s, %s, %s phase %s param %s" % (site, chan, band, phase, param)
                        model_type = "dummy"
                    else:
                        raise
            label = create_key(param=param, sta=site,
                               phase=phase, eid=parent.eid,
                               chan=chan, band=band)
            if model_type=="dummy":
                node = Node(label=label, model=DummyModel(), parents=[parent,])
            else:
                node = self.load_node_from_modelid(modelid, label)
                node.addParent(parent)
            nodes[sta] = node
            self.add_node(node)
        return nodes

    """

    okay, so to set up an arrival time node, I need to load the GP residual model. and that's great.
    but I also need to, somehow, add in the IASPEI predictions, and the event time. That's what the TravelTimeModel node already does.

    for the coda_height node, I'll need to do something similar. it depends on the amp_transfer node -- which is itself just a GP model -- but has to add in the event source stuff, and maybe projections onto the three components.

    also of course the wiggle node is a special case at the moment.

    any new template shape model / templategenerator would need to keep a notion of arrival time. it might add some other params, and it might define multiple height params (say a coda height and a peak height). I think maybe amp_transfer should be something more fundamental (though obviously the values we learn for it will be mediated by the way in which it gets used), and then the actual height params used by the shape model can be something else.
    so my generic code should set up an arrivalTime node and an amp_transfer node.
    then it loops through the params of the templateGenerator.
    for each one, it calls a templateGenerator.create_node() function.
    in the generic case, the templateGenerator just calls right back to the SG node creator.
    but for stuff like a coda_height node, it can do something custom.

    how do we know which params should be shared across channels/bands/arrays?
    some params ("coda_decay") should be joint across arrays, and exist above chan/band
    some params ("coda_height") should be specific to a specific sta/chan/band

    """

    def setup_tt(self, site, phase, event_node, tt_residual_node):
        nodes = dict()
        for sta in self.site_elements[site]:
            ttrn = extract_sta_node(tt_residual_node, sta)
            label = create_key(param="arrival_time", sta=sta, phase=phase, eid=event_node.eid)
            arrtimenode = ArrivalTimeNode(eid=event_node.eid, sta=sta, phase=phase, parents=[event_node, ttrn], label=label)
            self.add_node(arrtimenode)
            nodes[sta] = arrtimenode
        return nodes

    def add_event_site_phase(self, tg, wg, site, phase, event_node):
        # the "nodes" we create here can either be
        # actual nodes (if we are modeling these quantities
        # jointly across an array) or sta:node dictionaries (if we
        # are modeling them independently).
        def extract_sta_node_list(n):
            try:
                return n.values()
            except AttributeError:
                return [n,]

        tt_residual_node = self.setup_site_param_node(param="tt_residual", site=site,
                                                      phase=phase, parent=event_node)
        tt_node = self.setup_tt(site, phase, event_node=event_node, tt_residual_node=tt_residual_node)
        amp_transfer_node = tg.create_param_node(self, site, phase, band=None, chan=None, param="amp_transfer", event_node=event_node)

        for n in extract_sta_node_list(tt_residual_node):
            n.parent_predict()
        for n in extract_sta_node_list(tt_node):
            n.parent_predict()
        for n in extract_sta_node_list(amp_transfer_node):
            n.parent_predict()

        nodes = dict()
        nodes["arrival_time"] = tt_node

        for band in self.site_bands[site]:
            for chan in self.site_chans[site]:
                for param in tg.params():
                    # here the "create param node" creates, potentially, a single node or a dict of nodes
                    nodes[(band, chan, param)] = tg.create_param_node(self, site, phase, band, chan, param=param, event_node=event_node, tt_node=tt_node, amp_transfer_node=amp_transfer_node)
                for param in wg.params():
                    nodes[(band, chan, param)] = self.setup_site_param_node(param=param, site=site,
                                                                  phase=phase, parent=event_node,
                                                                  band=band, chan=chan, basisid=wg.basisid)

        child_wave_nodes = set()
        for sta in self.site_elements[site]:
            for wave_node in self.station_waves[sta]:
                if self.wave_captures_event_phase(ev=event_node.get_event(),
                                                  sta=sta, stime=wave_node.st,
                                                  etime=wave_node.et, phase=phase):
                    child_wave_nodes.add(wave_node)

        for (band_chan_param_key, ni) in nodes.items():
            for n in extract_sta_node_list(ni):
                n.parent_predict()
                for wn in child_wave_nodes:
                    n.addChild(wn)


    def add_wave(self, wave):
        """
        Add a wave node to the graph. Assume that all waves are added before all events.
        """

        wave_node = ObservedSignalNode(model_waveform=wave, nm_type=self.nm_type, observed=True, label=self._get_wave_label(wave=wave), graph=self)

        s = Sigvisa()
        sta = wave['sta']
        _, _, _, isarr, _, _, ref_site_id = s.earthmodel.site_info(sta, wave['stime'])
        ref_site_name = s.siteid_minus1_to_name[ref_site_id-1]
        if ref_site_name not in self.site_elements:
            self.site_elements[ref_site_name] = set()
            self.site_bands[ref_site_name] = set()
            self.site_chans[ref_site_name] = set()
        self.site_elements[ref_site_name].add(sta)
        self.site_bands[ref_site_name].add(wave['band'])
        self.site_chans[ref_site_name].add(wave['chan'])

        if sta not in self.station_waves:
            self.station_waves[sta] = []
        self.station_waves[sta].append(wave_node)

        self.leaf_nodes.append(wave_node)
        self.add_node(wave_node)
        return wave_node

    def _get_wave_label(self, wave):
        return 'wave_%s_%s_%s_%.1f' % (wave['sta'], wave['chan'], wave['band'], wave['stime'])

    """
    def get_template_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["template_%s" % lbl]

    def get_wiggle_node(self, **kwargs):
        lbl = self._get_interior_node_label(**kwargs)
        return self.all_nodes["wiggle_%s" % lbl]
   """

    def get_wave_node(self, wave):
        return self.all_nodes[self._get_wave_label(wave=wave)]

    def get_wave_node_log_p(self, wave_node):
        log_p = 0
        parents = wave_node.parents.values()
        for p in parents:
            log_p += p.log_p()
        log_p += wave_node.log_p()
        return log_p

    """
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
            if fixed:
                tm_node.fix_value(key='arrival_time')
            else:
                tm_node.unfix_value(key='arrival_time')

    def init_wiggles_from_template(self):
        wave_node = self.leaf_nodes[0]
        for wm_node in self.wiggle_nodes:
            tm_node = self.get_partner_node(wm_node)
            atime = tm_node.get_value()['arrival_time']

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

    def optimize_with_seed_time_and_depth(lon, lat, t, depth):
        assert(len(self.toplevel_nodes) == 1)
        ev_node = self.toplevel_nodes[0]
        ev_node.set_index(key="lon", value=lon)
        ev_node.set_index(key="lat", value=lat)
        ev_node.set_index(key="time", value=t)
        ev_node.set_index(key="depth", value=depth)
        ev_node.fix_value(key="lon")
        ev_node.fix_value(key="lat")
        self.fix_arrival_times(fixed=False)

        # initialize
        for n in self.template_nodes + self.wiggle_nodes:
            n.prior_predict()

        # TODO: optimize
    """

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

            slon, slat, _, _, _, _, _ = s.earthmodel.site_info(sta, st)
            distance = geog.dist_km((event.lon, event.lat), (slon, slat))
            azimuth = geog.azimuth((slon, slat), (event.lon, event.lat))

            tmpl_optim_param_str = tmpl_optim_param_str.replace("'", "''")
            wiggle_optim_param_str = wiggle_optim_param_str.replace("'", "''")
            optim_log = wiggle_optim_param_str.replace("\n", "\\\\n")

            sql_query = "INSERT INTO sigvisa_coda_fit (runid, evid, sta, chan, band, tmpl_optim_method, wiggle_optim_method, optim_log, iid, stime, etime, hz, acost, dist, azi, timestamp, elapsed, nmid) values (%d, %d, '%s', '%s', '%s', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f, %d)" % (runid, event.evid, sta, chan, band, tmpl_optim_param_str, wiggle_optim_param_str, self.optim_log, 1 if wave_node.nm_type != 'ar' else 0, st, et, hz, wave_node.log_p(), distance, azimuth, time.time(), elapsed, wave_node.nmid)

            fitid = execute_and_return_id(s.dbconn, sql_query, "fitid")

            parent_templates = [tm for tm in wave_node.parents.values() if tm.label.startswith("template_")]

            for tm in parent_templates:
                fit_params = tm.get_value()
                transfer = fit_params['coda_height'] - event.source_logamp(band, tm.phase)

                phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay, amp_transfer) values (%d, '%s', 'paired_exp', %f, %f, %f, %f, %f)" % (
                    fitid, tm.phase, fit_params['arrival_time'], fit_params['peak_offset'], fit_params['coda_height'], fit_params['coda_decay'], transfer)
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
