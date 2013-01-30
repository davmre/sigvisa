import numpy as np
import pdb
from sigvisa import *
from database.dataset import *
from database.signal_data import *
from source.event import *
import itertools, copy

from optparse import OptionParser
from signals.template_models.load_by_name import load_template_model
from signals.envelope_model import EnvelopeModel
from signals.io import load_segments
from plotting.event_heatmap import EventHeatmap
from wiggles.wiggle_models import PlainWiggleModel, StupidL1WiggleModel

def event_at(ev, lon=None, lat=None, t=None):
    ev2 = copy.copy(ev)
    ev2.lon = lon if lon is not None else ev.lon
    ev2.lat = lat if lat is not None else ev.lat
    ev2.time = t if t is not None else ev.time
    return ev2




# get the likelihood of an event location, if we don't know the event time.
# "likelihood" is a function of a segment and an event object (e.g. envelope_model.log_likelihood_optimize wrapped in a lambda)
def ev_loc_ll_at_optimal_time(ev, segments, log_likelihood, template_model):

    s = Sigvisa()

    # propose origin times based on phase arrival times
    event_time_proposals = []
    for segment in segments:
        backprojections = []
        arr = segment['arrivals']
        arr_phases = [s.phasenames[a] for a in arr[:,DET_PHASE_COL]]
        for phase, arrtime in zip(arr_phases, arr[:, DET_TIME_COL]):
            projection = arrtime - template_model.travel_time(ev, segment['sta'], phase)
            event_time_proposals.append(projection)

    # find the origin time that maximizes the likelihood
    maxll = np.float("-inf")
    maxt = 0
    f = lambda t: np.sum([log_likelihood(s, event_at(ev, t=t))[0] for s in segments])
    for proposed_t in event_time_proposals:
        ll = f(proposed_t)
        if ll > maxll:
            maxll = ll
            maxt = proposed_t
    return maxll


def main():

    parser = OptionParser()

    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID to locate")
    parser.add_option("-s", "--sites", dest="sites", default=None, type="str", help="comma-separated list of stations with which to locate the event")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="name of training run specifying the set of models to use")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type (paired_exp)")
    parser.add_option("-m", "--model", dest="model", default=None, type="str", help="name of training run specifying the set of models to use")
    parser.add_option("-w", "--map_width", dest="map_width", default=2, type="float", help="width in degrees of the plotted heat map (2)")
    parser.add_option("-n", dest="n", default=20, type="int", help="detail level of the heatmap, in number of points per side")
    parser.add_option("--method", dest="method", default="monte_carlo", help="method for signal likelihood computation (monte_carlo)")
    parser.add_option("--phases", dest="phases", default="all", help="comma-separated list of phases to include in predicted templates (all)")
    parser.add_option("--model_types", dest="model_types", default="peak_offset:constant_gaussian,amp_transfer:linear_distance,coda_decay:linear_distance", help="comma-separated list of param:model_type mappings (peak_offset:constant_gaussian,coda_height:linear_distance,coda_decay:linear_distance)")
    parser.add_option("--hz", dest="hz", default=5, type=float, help="downsample signals to a given sampling rate, in hz (5)")

    (options, args) = parser.parse_args()

    evid = options.evid
    sites = options.sites.split(',')

    map_width = options.map_width

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    # train / load coda models
    run_name = options.run_name
    iters = read_fitting_run_iterations(cursor, run_name)
    run_iter = np.max(iters[:, 0])

    model_types = {}
    for p in options.model_types.split(','):
        (param, model_type) = p.strip().split(':')
        model_types[param] = model_type

    tm = load_template_model(template_shape = options.template_shape, model_type=model_types, run_name=run_name, run_iter=run_iter)


    if options.phases == "all":
        phases = s.phases
    else:
        phases = options.phases.split(',')

#    wm = PlainWiggleModel(tm)
    wm = StupidL1WiggleModel(tm)
    em = EnvelopeModel(template_model=tm, wiggle_model=wm, phases=phases)
    ev_true = get_event(evid=evid)

    # inference is based on segments from all specified stations,
    # starting at the min predicted arrival time (for the true event)
    # minus 60s, and ending at the max predicted arrival time plus
    # 240s
    statimes = [ev_true.time + tm.travel_time(ev_true, sta, phase) for (sta,phase) in itertools.product(sites,s.phases)]
    stime = np.min(statimes) - 60
    etime = np.max(statimes) + 240
    segments = load_segments(cursor, sites, stime, etime)
    segments = [seg.with_filter('hz_%.3f' % options.hz) for seg in segments]

    if options.method == "mode":
        f_ll = em.log_likelihood_mode
    elif options.method == "monte_carlo":
        f_ll = em.log_likelihood_montecarlo
    elif options.method == "optimize":
        f_ll = em.log_likelihood_optimize
    else:
        raise Exception("unrecognized marginalization method %s" % options.method)

    f = lambda lon, lat: ev_loc_ll_at_optimal_time(event_at(ev_true, lon=lon, lat=lat), segments, log_likelihood=f_ll, template_model=tm)


    sta_string = ":".join(sites)
    run_label = hash(tuple(options.__dict__.items()))
    fname = None if run_label is None else "logs/heatmap_%s_values.txt" % run_label
    print "writing heat map to", fname
    stime = time.time()
    hm = EventHeatmap(f=f, n=options.n, center=(ev_true.lon, ev_true.lat), width=map_width, fname=fname)
    hm.save(fname)
    etime = time.time()

    d = {'evid':  ev_true.evid, \
         'timestamp': stime, \
         'elapsed': etime-stime, \
         'lon_nw': hm.min_lon, \
         'lat_nw': hm.max_lat, \
         'lon_se': hm.max_lon, \
         'lat_se': hm.min_lat, \
         'pts_per_side': hm.n, \
         'phases': ','.join(phases), \
         'likelihood_method': options.method, \
         'wiggle_model_type': wm.summary_str(), \
         'heatmap_fname': fname, \
    }
    save_gsrun_to_db(d, segments, em, tm)

        #hm.add_stations([s['sta'] for s in segments])
        #hm.set_true_event(evlon, evlat)
        #hm.savefig(themap_fname, title="location of event %d" % evid)




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
