import time
import numpy as np
import os
from sigvisa import Sigvisa

from sigvisa.database.dataset import read_timerange, read_events, EV_MB_COL, EV_EVID_COL
from sigvisa.database.signal_data import read_fitting_run_iterations
from sigvisa.graph.sigvisa_graph import SigvisaGraph, get_param_model_id, ModelNotFoundError
from sigvisa.graph.graph_utils import create_key
from sigvisa.source.event import get_event
from sigvisa.signals.io import load_event_station_chan, load_segments

def load_sg_from_db_fit(fitid, load_wiggles=True):

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    fit_sql_query = "select f.runid, f.evid, f.sta, f.chan, f.band, f.hz, f.smooth, f.stime, f.etime, nm.model_type from sigvisa_coda_fit f, sigvisa_noise_model nm where f.fitid=%d and f.nmid=nm.nmid" % (fitid)
    cursor.execute(fit_sql_query)
    fit = cursor.fetchone()
    ev = get_event(evid=fit[1])
    wave = load_event_station_chan(fit[1], fit[2], fit[3], cursor=cursor).filter('%s;env;smooth_%d;hz_%.2f' % (fit[4], fit[6], fit[5]))
    nm_type = fit[9]
    runid = fit[0]

    phase_sql_query = "select fpid, phase, template_model, arrival_time, peak_offset, coda_height, peak_decay, coda_decay, wiggle_family from sigvisa_coda_fit_phase where fitid=%d" % fitid
    cursor.execute(phase_sql_query)
    phase_details = cursor.fetchall()
    phases = [p[1] for p in phase_details]
    templates = {}
    tmshapes = {}
    uatemplates = []
    wiggle_family="dummy"
    for (phase, p) in zip(phases, phase_details):
        shape = p[2]
        tparams = {'arrival_time': p[3], 'peak_offset': p[4], 'coda_height': p[5], 'coda_decay': p[7]}
        if p[2]=="lin_polyexp":
            tparams['peak_decay'] = p[6]
        wiggle_family=p[-1]

        tmshapes[phase] = shape
        if phase=="UA":
            uatemplates.append(tparams)
        else:
            templates[phase] = tparams

    sg = SigvisaGraph(template_model_type="dummy", wiggle_model_type="dummy",
                      template_shape=tmshapes, wiggle_family=wiggle_family,
                      nm_type = nm_type, runid=runid, phases=phases,
                      base_srate=wave['srate'])
    wave_node = sg.add_wave(wave)
    sg.add_event(ev)

    for uaparams in uatemplates:
        sg.create_unassociated_template(wave_node, atime=uaparams['arrival_time'], initial_vals=uaparams)

    for phase in templates.keys():
        sg.set_template(eid=ev.eid, sta=wave['sta'], band=wave['band'],
                        chan=wave['chan'], phase=phase,
                        values = templates[phase])
        print "setting template", ev.eid, phase, "to", templates[phase]

    return sg



def register_svgraph_cmdline(parser):
    parser.add_option("-s", "--sites", dest="sites", default=None, type="str",
                      help="comma-separated list of stations with which to locate the event")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str",
                      help="name of training run specifying the set of models to use")
    parser.add_option("--runid", dest="runid", default=None, type="int",
                      help="runid of training run specifying the set of models to use")
    parser.add_option(
        "--template_shape", dest="template_shape", default="lin_polyexp", type="str", help="template model type (lin_polyexp)")
    parser.add_option(
        "--phases", dest="phases", default="auto", help="comma-separated list of phases to include in predicted templates (auto)")
    parser.add_option(
        "--template_model_types", dest="tm_types", default="param",
        help="comma-separated list of param:model_type mappings (peak_offset:constant_gaussian,coda_height:constant_gaussian,coda_decay:constant_gaussian)")
    parser.add_option("--wiggle_model_type", dest="wm_type", default="dummy", help = "")
    parser.add_option("--wiggle_family", dest="wiggle_family", default="dummy", help = "")
    parser.add_option("--dummy_fallback", dest="dummy_fallback", default=False, action="store_true",
                      help="fall back to a dummy model instead of throwing an error if no model for the parameter exists in the database (False)")
    parser.add_option("--arrays_joint", dest="arrays_joint", default=False, action="store_true",
                      help="model array stations with joint nodes (False)")
    parser.add_option("--absorb_n_phases", dest="absorb_n_phases", default=False, action="store_true",
                      help="model Pn arrivals as P (false)")
    parser.add_option("--nm_type", dest="nm_type", default="ar", type="str",
                      help="type of noise model to use (ar)")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=1e-6, type=float, help="Poisson rate (per-second) for unassociated template prior (1e-6)")

def register_svgraph_signal_cmdline(parser):
    parser.add_option("--hz", dest="hz", default=5, type=float, help="downsample signals to a given sampling rate, in hz (5)")
    parser.add_option("--smooth", dest="smooth", default=None, type=int, help="perform the given level of smoothing")
    parser.add_option("--chans", dest="chans", default="auto", type="str",
                      help="comma-separated list of channel names to use for inference (auto)")
    parser.add_option("--bands", dest="bands", default="freq_2.0_3.0", type="str",
                      help="comma-separated list of band names to use for inference (freq_2.0_3.0)")
    parser.add_option("--array_refsta_only", dest="refsta_only", default=True, action="store_false",
                      help="load only the reference station for each array site (True)")
    parser.add_option("--start_time", dest="start_time", default=None, type="float",
                      help="load signals beginning at this UNIX time (None)")
    parser.add_option("--end_time", dest="end_time", default=None, type="float",
                      help="load signals end at this UNIX time (None)")
    parser.add_option("--dataset", dest="dataset", default="training", type="str",
                      help="if start_time and end_time not specified, load signals from the time period of the specified dataset (training)")
    parser.add_option("--hour", dest="hour", default=0, type="float",
                      help="start at a particular hour of the given dataset (0)")
    parser.add_option("--len_hours", dest="len_hours", default=1, type="float",
                      help="load this many hours from the given dateset")
    parser.add_option("--initialize_leb", dest="initialize_leb", default="no", type="str",
                      help="use LEB events to set the intial state. options are 'no', 'yes', 'perturb' to initialize with locations randomly perturbed by ~5 degrees, or 'count' to initialize with a set of completely random events, having the same count as the LEB events ")
    parser.add_option("--synth", dest="synth", default=False, action="store_true")

def register_svgraph_event_based_signal_cmdline(parser):
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID to locate")
    parser.add_option("--hz", dest="hz", default=5, type=float, help="downsample signals to a given sampling rate, in hz (5)")
    parser.add_option("--smooth", dest="smooth", default=None, type=int, help="perform the given level of smoothing")
    parser.add_option("--chans", dest="chans", default="auto", type="str",
                      help="comma-separated list of channel names to use for inference (auto)")
    parser.add_option("--bands", dest="bands", default="freq_2.0_3.0", type="str",
                      help="comma-separated list of band names to use for inference (freq_2.0_3.0)")
    parser.add_option("--array_refsta_only", dest="refsta_only", default=True, action="store_false",
                      help="load only the reference station for each array site (True)")

def setup_svgraph_from_cmdline(options, args):

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    if options.runid is None:
        run_name = options.run_name
        iters = np.array(sorted(list(read_fitting_run_iterations(cursor, run_name))))
        run_iter, runid = iters[-1, :]
    else:
        runid = options.runid


    tm_type_str = options.tm_types
    if tm_type_str == "param":
        tm_type_str = "tt_residual:constant_laplacian,peak_offset:param_linear_mb,amp_transfer:param_sin1,coda_decay:param_linear_distmb,peak_decay:param_linear_distmb"

    tm_types = {}
    if ',' in tm_type_str:
        for p in tm_type_str.split(','):
            (param, model_type) = p.strip().split(':')
            tm_types[param] = model_type
    else:
        tm_types = tm_type_str

    if options.phases in ("auto", "leb"):
        phases = options.phases
    else:
        phases = options.phases.split(',')

    cursor.close()

    sg = SigvisaGraph(template_shape = options.template_shape, template_model_type = tm_types,
                      wiggle_family = options.wiggle_family, wiggle_model_type = options.wm_type,
                      dummy_fallback = options.dummy_fallback, nm_type = options.nm_type,
                      runid=runid, phases=phases, gpmodel_build_trees=False, arrays_joint=options.arrays_joint,
                      absorb_n_phases=options.absorb_n_phases, uatemplate_rate=options.uatemplate_rate)


    return sg


def load_signals_from_cmdline(sg, options, args):

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sites = options.sites.split(',')
    stas = s.sites_to_stas(sites, refsta_only=options.refsta_only)

    if options.start_time is not None and options.end_time is not None:
        stime = options.start_time
        etime = options.end_time
    else:
        print "loading signals from dataset %s" % options.dataset
        (stime, etime) = read_timerange(cursor, options.dataset, hours=None, skip=0)
        stime += options.hour * 3600
        etime = stime + options.len_hours*3600.0

    print "loading signals from stime %.1f through etime %.1f" % (stime, etime)

    if options.bands == "all":
        bands = s.bands
    else:
        bands = options.bands.split(',')

    if options.chans == "all":
        chans = s.chans
    else:
        chans = options.chans.split(',')


    segments = load_segments(cursor, stas, stime, etime, chans = chans)
    segments = [seg.with_filter('env;hz_%.3f' % options.hz) for seg in segments]

    for seg in segments:
        for band in bands:
            filtered_seg = seg.with_filter(band)
            if options.smooth is not None:
                filtered_seg = filtered_seg.with_filter("smooth_%d" % options.smooth)

            for chan in filtered_seg.get_chans():
                try:
                    modelid = get_param_model_id(sg.runid, seg['sta'], 'P', sg._tm_type('amp_transfer', site=seg['sta'], wiggle_param=False), 'amp_transfer', options.template_shape, chan=chan, band=band)
                except ModelNotFoundError as e:
                    print "couldn't find amp_transfer model for %s,%s,%s, so not adding to graph." % (seg['sta'], chan, band), e
                    continue

                wave = filtered_seg[chan]
                wn = sg.add_wave(wave)


    evs = get_leb_events(sg, cursor)
    if options.initialize_leb != "no" or options.synth:
        if options.initialize_leb == "yes" or options.synth:
            for ev in evs:
                sg.add_event(ev, fixed=options.synth)
        elif options.initialize_leb=="perturb":
            raise NotImplementedError("not implemented!")
        elif options.initialize_leb=="count":
            evs = sg.prior_sample_events(stime=st, etime=et, n_events=len(events))
        else:
            raise Exception("unrecognized argument initialize_leb=%s" % options.initialize_leb)

    if options.synth:
        for (sta, wns) in sg.station_waves.items():
            for wn in wns:
                wn.unfix_value()

        sg.parent_sample_all()

        for (sta, wns) in sg.station_waves.items():
            for wn in wns:
                wn.fix_value()

        eids = sg.evnodes.keys()
        for eid in eids:
            if options.initialize_leb=="no":
                sg.remove_event(eid)
            else:
                for evnode in sg.evnodes[eid].values():
                    evnode.unfix_value()

    cursor.close()

    return evs

def get_leb_events(sg, cursor):
    st = sg.start_time
    et = sg.end_time
    events, orid2num = read_events(cursor, st, et, 'leb')
    events = [evarr for evarr in events if evarr[EV_MB_COL] > 2]

    evs = []
    eid = 1
    for evarr in events:
        ev = get_event(evid=evarr[EV_EVID_COL])
        ev.eid = eid
        eid += 1
        evs.append(ev)
    return evs

def load_event_based_signals_from_cmdline(sg, options, args):

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    evid = options.evid
    ev_true = get_event(evid=evid)

    sites = options.sites.split(',')
    stas = s.sites_to_stas(sites, options.refsta_only)

    if options.bands == "all":
        bands = s.bands
    else:
        bands = options.bands.split(',')

    if options.chans == "all":
        chans = s.chans
    else:
        chans = options.chans.split(',')

    # inference is based on segments from all specified stations,
    # starting at the min predicted arrival time (for the true event)
    # minus 60s, and ending at the max predicted arrival time plus
    # 240s
    statimes = [ev_true.time + tt_predict(event=ev_true, sta=sta, phase=phase) for (sta, phase) in itertools.product(sites, s.phases)]
    stime = np.min(statimes) - 60
    etime = np.max(statimes) + 240
    segments = load_segments(cursor, stas, stime, etime, chans = chans)
    segments = [seg.with_filter('env;hz_%.3f' % options.hz) for seg in segments]

    for seg in segments:
        for band in bands:
            filtered_seg = seg.with_filter(band)
            if options.smooth is not None:
                filtered_seg = filtered_seg.with_filter("smooth_%d" % options.smooth)

            for chan in filtered_seg.get_chans():
                wave = filtered_seg[chan]
                sg.add_wave(wave)

    evnodes = sg.add_event(ev_true)

    cursor.close()

    return evnodes
