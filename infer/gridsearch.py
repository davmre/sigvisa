import numpy as np
import scipy.stats as stats
import pdb
import itertools
import copy
import hashlib
import pickle
from optparse import OptionParser

from sigvisa import *
from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.source.event import *
from sigvisa.models.templates.load_by_name import load_template_model
from sigvisa.signals.io import load_segments
from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa.models.ttime import tt_predict
from sigvisa.models.ev_prior import EV_LON, EV_LAT, EV_DEPTH, EV_TIME, EV_MB
from sigvisa.graph.sigvisa_graph import SigvisaGraph, predict_phases

def event_at(ev, lon=None, lat=None, t=None, depth=None):
    ev2 = copy.copy(ev)
    ev2.lon = lon if lon is not None else ev.lon
    ev2.lat = lat if lat is not None else ev.lat
    ev2.depth = depth if depth is not None else ev.depth
    ev2.time = t if t is not None else ev.time
    return ev2

def propose_origin_times(ev, segments, phases, max_proposals=5):
    s = Sigvisa()

    # propose origin times based on phase arrival times
    event_time_proposals = []
    for segment in segments:
        backprojections = []
        arr = segment['arrivals']
        if len(arr) == 0:
            continue

        for phase in predict_phases(ev=ev, sta=segment['sta'], phases=phases):
            for arrtime in arr[:, DET_TIME_COL]:
                projection = arrtime - tt_predict(ev, segment['sta'], phase)
                event_time_proposals.append(projection)

    # if we have too many event time proposals, use a kernel density
    # estimate to get a representative sample.
    if len(event_time_proposals) > max_proposals:
        np.random.seed(0)
        kernel = stats.gaussian_kde(event_time_proposals, bw_method = lambda x : 0.01)
        event_time_proposals = list(kernel.resample(np.array((max_proposals,))).flatten())

    return event_time_proposals

def save_gsrun_to_db(d, ev, sg):
    from sigvisa.models.noise.noise_util import get_noise_model
    s = Sigvisa()
    sql_query = "insert into sigvisa_gridsearch_run (evid, timestamp, elapsed, lon_nw, lat_nw, lon_se, lat_se, pts_per_side, likelihood_method, phases, wiggle_model_type, heatmap_fname, max_evtime_proposals, true_depth) values (:evid, :timestamp, :elapsed, :lon_nw, :lat_nw, :lon_se, :lat_se, :pts_per_side, :likelihood_method, :phases, :wiggle_model_type, :heatmap_fname, :max_evtime_proposals, :true_depth)"
    gsid = execute_and_return_id(s.dbconn, sql_query, "gsid", **d)

    for wave_node in sg.leaf_nodes:

        sta = wave_node.sta
        filter_str = wave_node.filter_str
        srate = wave_node.srate
        stime = wave_node.st
        etime = wave_node.et
        nmid = wave_node.nmid
        band = wave_node.band
        chan = wave_node.chan

        gswid = execute_and_return_id(s.dbconn,
                                      "insert into sigvisa_gsrun_wave (gsid, sta, chan, band, stime, etime, hz, nmid) values (%d, '%s', '%s', '%s', %f, %f, %f, %d)"
                                      % (gsid, sta, chan, band, stime, etime, srate, nmid), 'gswid')


        for tm_node in [tmn for (lbl, tmn) in wave_node.parents.items() if lbl.startswith("template_")]:
            for modelid in tm_node.get_modelids():
                gsmid = execute_and_return_id(s.dbconn, "insert into sigvisa_gsrun_tmodel (gswid, modelid) values (%d, %d)" % (gswid, modelid), 'gsmid')

    s.dbconn.commit()



def main():

    parser = OptionParser()

    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID to locate")
    parser.add_option("-s", "--sites", dest="sites", default=None, type="str",
                      help="comma-separated list of stations with which to locate the event")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str",
                      help="name of training run specifying the set of models to use")
    parser.add_option(
        "--template_shape", dest="template_shape", default="paired_exp", type="str", help="template model type (paired_exp)")
    parser.add_option(
        "-m", "--model", dest="model", default=None, type="str", help="name of training run specifying the set of models to use")
    parser.add_option(
        "-w", "--map_width", dest="map_width", default=2, type="float", help="width in degrees of the plotted heat map (2)")
    parser.add_option("-n", dest="n", default=20, type="int", help="detail level of the heatmap, in number of points per side")
    parser.add_option(
        "--method", dest="method", default="monte_carlo", help="method for signal likelihood computation (monte_carlo)")
    parser.add_option(
        "--phases", dest="phases", default="auto", help="comma-separated list of phases to include in predicted templates (auto)")
    parser.add_option(
        "--template_model_types", dest="tm_types", default="peak_offset:constant_gaussian,amp_transfer:constant_gaussian,coda_decay:constant_gaussian",
        help="comma-separated list of param:model_type mappings (peak_offset:constant_gaussian,coda_height:constant_gaussian,coda_decay:constant_gaussian)")
    parser.add_option("--wiggle_model_type", dest="wm_type", default="dummy", help = "")
    parser.add_option("--wiggle_family", dest="wiggle_family", default="fourier_0.8", help = "")
    parser.add_option("--hz", dest="hz", default=5, type=float, help="downsample signals to a given sampling rate, in hz (5)")
    parser.add_option("--max_evtime_proposals", dest="max_evtime_proposals", default=5, type="int",
                      help="maximum number of event times to consider per gridsearch location")
    parser.add_option("--use_true_depth", dest="use_true_depth", default=False, action="store_true",
                      help="use the true depth of the event being searched for (default is to integrate over depth)")
    parser.add_option("--chans", dest="chans", default="BHZ", type="str",
                      help="comma-separated list of channel names to use for inference (BHZ)")
    parser.add_option("--bands", dest="bands", default="freq_2.0_3.0", type="str",
                      help="comma-separated list of band names to use for inference (freq_2.0_3.0)")
    parser.add_option("--nm_type", dest="nm_type", default="ar", type="str",
                      help="type of noise model to use (ar)")


    (options, args) = parser.parse_args()

    evid = options.evid
    sites = options.sites.split(',')

    map_width = options.map_width

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    # train / load coda models
    run_name = options.run_name
    iters = np.array(sorted(list(read_fitting_run_iterations(cursor, run_name))))
    run_iter, runid = iters[-1, :]

    tm_types = {}
    if ',' in options.tm_types:
        for p in options.tm_types.split(','):
            (param, model_type) = p.strip().split(':')
            tm_types[param] = model_type
    else:
        tm_types = options.tm_types

    if options.phases in ("auto", "leb"):
        phases = options.phases
    else:
        phases = options.phases.split(',')


    if options.bands == "all":
        bands = s.bands
    else:
        bands = options.bands.split(',')

    if options.chans == "all":
        chans = s.chans
    else:
        chans = options.chans.split(',')

    ev_true = get_event(evid=evid)

    # inference is based on segments from all specified stations,
    # starting at the min predicted arrival time (for the true event)
    # minus 60s, and ending at the max predicted arrival time plus
    # 240s
    statimes = [ev_true.time + tt_predict(event=ev_true, sta=sta, phase=phase) for (sta, phase) in itertools.product(sites, s.phases)]
    stime = np.min(statimes) - 60
    etime = np.max(statimes) + 240
    segments = load_segments(cursor, sites, stime, etime, chans = chans)
    segments = [seg.with_filter('env;hz_%.3f' % options.hz) for seg in segments]

    def build_gridsearch_graph():
        sg = SigvisaGraph(template_shape = options.template_shape, template_model_type = tm_types,
                          wiggle_family = options.wiggle_family, wiggle_model_type = options.wm_type,
                          nm_type = options.nm_type, runid=runid, phases=phases)
        ev_node = sg.add_event(ev_true)
        for seg in segments:
            for band in bands:
                filtered_seg = seg.with_filter(band)
                for chan in filtered_seg.get_chans():
                    wave = filtered_seg[chan]
                    sg.add_wave(wave)
        return sg

    def create_graph_predict(lon, lat, f_propose, ev_depths=[0, 100]):
        sg = build_gridsearch_graph()

        event_node = sg.toplevel_nodes[0]
        event_node.unfix_value()
        event_node.set_index(i=EV_LON, value=lon)
        event_node.set_index(i=EV_LAT, value=lat)
        event_node.fix_value(i=EV_LON)
        event_node.fix_value(i=EV_LAT)

        best_ll = np.float("-inf")
        best_time = 0
        best_depth = 0
        for depth in ev_depths:
            ev_times = f_propose(ev=event_at(ev=event_node.get_event(),lon=lon,lat=lat,depth=depth))
            for t in ev_times:

                event_node.set_index(i=EV_TIME, value=t)
                event_node.set_index(i=EV_DEPTH, value=depth)
                sg.prior_predict_all()
                ll = sg.current_log_p()
                if ll > best_ll:
                    best_ll = ll
                    best_time = t
                    best_depth = depth

        event_node.set_index(i=EV_TIME, value=best_time)
        event_node.set_index(i=EV_DEPTH, value=best_depth)
        sg.prior_predict_all()

        return sg

    if options.method == "mode":
        if options.wm_type != "dummy":
            raise Exception("WARNING: do you really want to do mode inference with a non-dummy wiggle model?")

        f_propose = lambda ev: propose_origin_times(ev=ev,
                                                    segments=segments,
                                                    phases=options.phases,
                                                    max_proposals = options.max_evtime_proposals)
        if options.use_true_depth:
            f_sg = lambda lon, lat: create_graph_predict(lon, lat, f_propose,
                                                         (ev_true.depth,))
        else:
            f_sg = lambda lon, lat: create_graph_predict(lon, lat, f_propose, (0,100))
    else:
        raise ValueError("unrecognized inference method '%s'" % options.method)

    sta_string = ":".join(sites)
    run_label = hashlib.sha1(repr(options.__dict__)).hexdigest()
    heatmap_dir = None if run_label is None else "logs/heatmaps/%s/" % run_label
    ensure_dir_exists(heatmap_dir)
    print "writing heat map files to", heatmap_dir
    stime = time.time()

    hm = EventHeatmap(f=None, n=options.n,
                      center=(ev_true.lon, ev_true.lat),
                      width=map_width, fname="%s/overall.txt" % heatmap_dir, calc=False)
    coord_list = hm.coord_list()
    coord_graphs = [f_sg(lon=lon, lat=lat) for (lon, lat) in coord_list]
    overall_lls = [sg.current_log_p() for sg in coord_graphs]
    hm.set_coord_fvals(overall_lls)
    hm.save()
    etime = time.time()
    print "finished heatmap; saving metadata to database..."

    sg0 = coord_graphs[0]
    for wn in sg.leaf_nodes:
        wave = wn.mw
        lls = [sg.get_wave_node_log_p(sg.get_wave_node(wave)) for sg in coord_graphs]
        wave_label = "%s_%s_%s" % (wave['sta'], wave['chan'], wave['band'])
        hm = EventHeatmap(f=None, n=options.n,
                          center=(ev_true.lon, ev_true.lat),
                          width=map_width, fname=os.path.join(heatmap_dir,
                                                              "wave_%s.txt" % ( wave_label)), calc=False)
        hm.set_coord_fvals(lls)
        hm.save()

    for (sg, (lon, lat)) in zip(coord_graphs, coord_list):
        f = open(os.path.join(heatmap_dir, "graph_%.3f_%.3f.pickle" % (lon, lat)), 'wb')
        pickle.dump(sg, f)
        f.close()

    d = {'evid': ev_true.evid,
         'timestamp': stime,
         'elapsed': etime - stime,
         'lon_nw': hm.min_lon,
         'lat_nw': hm.max_lat,
         'lon_se': hm.max_lon,
         'lat_se': hm.min_lat,
         'pts_per_side': hm.n,
         'phases': ','.join(phases),
         'likelihood_method': options.method,
         'wiggle_model_type': options.wm_type,
         'heatmap_fname': heatmap_dir,
         'max_evtime_proposals': options.max_evtime_proposals,
         'true_depth': 't' if options.use_true_depth else 'f',
         }

    save_gsrun_to_db(d=d, ev=ev_true, sg=sg)

        # hm.add_stations([s['sta'] for s in segments])
        # hm.set_true_event(evlon, evlat)
        # hm.savefig(themap_fname, title="location of event %d" % evid)




if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
