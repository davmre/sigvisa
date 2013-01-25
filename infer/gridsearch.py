import numpy as np
import pdb
from sigvisa import *
from database.dataset import *
from database.signal_data import *
from source.event import *

from optparse import OptionParser
from signals.template_models.load_by_name import load_template_model


def event_at(ev, lon=None, lat=None, t=None):
    ev2 = ev.copy()
    ev2.lon = lon if lon is not None else ev.lon
    ev2.lat = lat if lat is not None else ev.lat
    ev2.t = t if t is not None else ev.t
    return ev2



#        f = lambda lon, lat: self.event_location_likelihood(event_at(base_event, lon=lon, lat=lat), segments, pp=pp, marginalize_method=marginalize_method, iid=iid)

# get the likelihood of an event location, if we don't know the event time.
# "likelihood" is a function of a segment and an event object (e.g. envelope_model.log_likelihood_optimize wrapped in a lambda)
def event_location_likelihood(ev, segments, pp, log_likelihood, template_model, iid=False):

    s = Sigvisa()

    event_time_proposals = []
    for segment in segments:
        is_new = lambda l, x : (len(l) == 0 or np.min([np.abs(lx - x) for lx in l]) > 1.5)
        backprojections = []
        arr = segment['arrivals']
        arr_phases = [s.phasenames[a] for a in arr[:,DET_PHASE_COL]]
        for phase, arrtime in zip(arr_phases, arrivals[:, DET_PHASE_COL]):
            try:
                projection = arrtime - template_model.travel_time(ev, segment['sta'], phase)
                if is_new(event_time_proposals, projection):
                    event_time_proposals.append(projection)
            except:
                pass
#        print "siteid %d, lat %f lon %f, distance from true %f, backprojecting p_error %f s_error %f" % (siteid, evlon, evlat, dist, p_projection-ev[EV_TIME_COL], s_projection-ev[EV_TIME_COL])


    # find the event time that maximizes the likelihood
    maxll = np.float("-inf")
    maxt = 0

    f = lambda t: np.sum([log_likelihood(s, self.event_at(ev, t=t), iid=iid)[0] for s in segments])
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
    parser.add_option("--iid", dest="iid", default=False, action="store_true", help="use a uniform iid noise model (instead of AR)")
    parser.add_option("--method", dest="method", default="monte_carlo", help="method for signal likelihood computation (monte_carlo)")

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

    template_model = load_template_model(template_shape = options.template_shape, model_type="gp_dad_log", run_name=run_name, run_iter=run_iter)

    sta_string = ":".join([siteid_to_sta(sid,cursor) for sid in siteids])
    run_label = "%d_%d_%s_%s_%s" % (evid, map_width, sta_string, options.method, "iid" if options.iid else "arwiggle")
    out_fname = os.path.join("logs", "heatmap_run_%s.pdf" % run_label)
    themap_fname = os.path.join("logs", "heatmap_%s.png" % run_label)
    pp = PdfPages(out_fname)
    print "saving plots to", out_fname

    load_wiggle_models(cursor, sigmodel, "parameters/signal_wiggles.txt")
    em = EnvelopeModel(template_model)

    ev_true = get_event(evid=evid)

    # inference is based on segments from all specified stations,
    # starting at the min predicted arrival time (for the true event)
    # minus 60s, and ending at the max predicted arrival time plus
    # 240s
    statimes = [ev.time + tm.travel_time(ev_true, sta, phase) for (sta,phase) in itertools.product(sites,s.phases)]
    stime = np.min(statimes) - 60
    etime = np.max(statimes) + 240
    segments = load_segments(cursor, sites, stime, etime)

    """
    for sta in sites:

        plot_band = 'narrow_envelope_2.00_3.00'
        plot_chan = 'BHZ'

        s = load_event_station(cursor, evid, sta)
        signals.append(s)

        tr = s[0][plot_chan][plot_band]
        fig = plot.plot_trace(tr)
        em.plot_predicted_signal(s[0], ev, pp, iid=True, chan=plot_chan, band=plot_band)

        pp.savefig()
        plt.close(fig)
     """
#    print "computing sensitivity..."
#    tm.sensitivity(pp, signals, ev)

    print "computing heat map..."
    try:

        f = lambda lon, lat: self.event_location_likelihood(event_at(base_event, lon=lon, lat=lat), segments, pp=pp, marginalize_method=marginalize_method, iid=iid)

        fname = None if run_label is None else "logs/heatmap_%s_values.txt" % run_label
        hm = EventHeatmap(f=f, n=n, center=(evlon, evlat), width=map_width, fname=fname)
        hm.add_stations([s['sta'] for s in segments])
        hm.set_true_event(evlon, evlat)
        hm.savefig(themap_fname, title="location of event %d" % evid)

    finally:
        pp.close()


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


