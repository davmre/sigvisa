import os
import numpy as np
from optparse import OptionParser

from sigvisa.results.compare import f1_and_error, find_matched, find_unmatched, find_matching
from sigvisa.graph.serialization import load_serialized_from_file
from sigvisa import Sigvisa
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.region import Region

from sigvisa.utils.geog import dist_km
from sigvisa.source.event import get_event

def get_bulletin(time_blocks, origin_type="isc"):

    region_cond = "and lon between -126 and -100 and lat between 33 and 49"
    extra_vars = ""
    if origin_type=="visa":
        extra_vars = ", evscore, orid"
    else:
        extra_vars = ", 0.0, orid"

    s = Sigvisa()
    bulletins = []
    for (stime, etime) in time_blocks:
        query = "select lon, lat, depth, time, mb%s from %s_origin where time between %f and %f %s order by time" % (extra_vars, origin_type, stime, etime, region_cond)

        block_bulletin = np.array(s.sql(query))
        if len(block_bulletin) > 0:
            bulletins.append(block_bulletin)
        
    return np.vstack(bulletins)


def filter_de_novo(bulletin):
    train_evids = [int(evid) for evid in np.loadtxt("/home/dmoore/python/sigvisa/notebooks/thesis/train_evids.txt")]
    train_evs = [get_event(evid) for evid in train_evids]

    def is_denovo(row, dist_threshold_km=50):
        lon, lat, depth, time, mb = row[:5]
        dists = [dist_km((lon, lat), (tev.lon, tev.lat)) for tev in train_evs]
        return np.min(dists) > dist_threshold_km

    filtered = np.array([row for row in bulletin if is_denovo(row)])
    return filtered

def fmt_ev(evarr):
    lon, lat, depth, time, mb = evarr[:5]
    return "%6.3f %6.3f %4.1f %.1f %3.1f " % (lon, lat, depth, time, mb)

def print_bulletin(bulletin):
    sb = sorted(bulletin, key = lambda b : b[3])
    for evinfo in sb:
        (lon, lat, depth, time, mb) = evinfo[:5]
        try:
            score = evinfo[5]
        except:
            score = 0.0

        print "%6.3f %6.3f %4.1f %.1f %3.1f %.1f" % (lon, lat, depth, time, mb, score)


def print_bulletin_matching(bulletin, ground_bulletin, indices):
    matching_map = dict([(j, i) for (i,j) in indices])
    #sb = sorted(bulletin, key = lambda b : b[3])
    for j, evinfo in enumerate(bulletin):
        (lon, lat, depth, time, mb) = evinfo[:5]
        print "%6.3f %6.3f %4.1f %.1f %3.1f " % (lon, lat, depth, time, mb),
        try:
            score = evinfo[5]
            print "%.3f " % score,
        except:
            pass

        if j in matching_map:
            print " matches %s" % fmt_ev(ground_bulletin[matching_map[j]])
        else:
            print

def simulate_prior(nevs, stime, etime):
    region = Region(lons=(-126.0, 100.0), lats=(33.0, 49.0), times=(stime, etime))
    sg = SigvisaGraph(inference_region=region)
    prior_evs = sg.prior_sample_events(n_events=nevs, stime=stime, etime=etime)
    prior_bulletin = np.array([(ev.lon, ev.lat, ev.depth, ev.time, ev.mb) for ev in prior_evs])
    return prior_bulletin


def map_bulletin(bulletin, true_bulletin, fname, stas=None):

    from sigvisa.plotting.event_heatmap import EventHeatmap
    from sigvisa.plotting.plot import savefig
    from matplotlib.figure import Figure

    s = Sigvisa()
    hm = EventHeatmap(f=None, calc=False, 
                      left_lon=-126, right_lon=-100, 
                      bottom_lat=33, top_lat=49)


    fig = Figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    hm.init_bmap(axes=ax, nofillcontinents=True, projection="cyl")
    hm.plot_earth()

    if stas is not None:
        sta_locations = [s.earthmodel.site_info(n, 0)[0:2] for n in stas]
        hm.plot_locations(sta_locations, labels=stas,
                          marker="^", ms=20, mec="none", mew=0,
                          alpha=1.0, mfc="blue")
    
    
    hm.bmap.scatter(bulletin[:, 0], bulletin[:, 1], marker="d", s=25, alpha=0.3, c="blue", lw=0)

    true_bulletin=np.asarray(true_bulletin)
    hm.bmap.scatter(true_bulletin[:, 0], true_bulletin[:, 1], marker=".", s=25, alpha=0.3, c="red", lw=0)


    if fname is not None:
        savefig(fname, fig, dpi=300, bbox_inches='tight')

def plot_mbs(bulletin, fname):
    from sigvisa.plotting.plot import savefig
    from matplotlib.figure import Figure
    import seaborn as sns


    fig = Figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    mbs = [ mb for mb in bulletin[:, 4] if mb > 0 ]
    sns.distplot(mbs, ax=ax)
    if fname is not None:
        ax.set_title(fname)
        savefig(fname, fig, dpi=300, bbox_inches='tight')

    



def main():
    parser = OptionParser()

    parser.add_option("--serialized", dest="serialized", default=None, type="str",
                      help="tgz file with serialized events")
    parser.add_option("--label", dest="label", default=None, type="str",
                      help="name to tag the output plots/files")    
    parser.add_option("--stime", dest="stime", default=None, type="float",
                      help="")
    parser.add_option("--etime", dest="etime", default=None, type="float",
                      help="")
    parser.add_option("--timefile", dest="timefile", default=None, type="str",
                      help="")
    parser.add_option("--max_delta_deg", dest="max_delta_deg", default=2.0, type="float",
                      help="")
    parser.add_option("--max_delta_time", dest="max_delta_time", default=50.0, type="float",
                      help="")
    parser.add_option("--plot_maps", dest="plot_maps", default=False, action="store_true",
                      help="")
    parser.add_option("--threshold", dest="threshold", default=None, type="float",
                      help="")
    parser.add_option("--denovo", dest="denovo", default=False, action="store_true",
                      help="")

    parser.add_option("--bulletin", dest="bulletin", default="sigvisa",
                      help="sigvisa, visa, leb, sel3")

    (options, args) = parser.parse_args()

    if options.timefile is not None:
        time_blocks = np.loadtxt(options.timefile).reshape((-1, 2))
    elif options.stime is None:
        timefile = os.path.join(os.path.dirname(options.serialized), 'times.txt')
        time_blocks = np.loadtxt(timefile).reshape((-1, 2))
    else:
        time_blocks = np.array(((stime, etime),)).reshape((1, 2))
        
    if options.label is None:
        if options.serialized is not None:
            label = os.path.basename(os.path.dirname(options.serialized))
        else:
            label = options.bulletin
    else:
        label = options.label

    plot_maps = options.plot_maps

    stas = "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(",")

    isc_bulletin = get_bulletin(time_blocks, origin_type="isc")

    np.savetxt("isc_bulletin_%s.txt" % (label), isc_bulletin)

    print "true (ISC) bulletin: %d events" % (len(isc_bulletin))
    print_bulletin(isc_bulletin)
    print

    if plot_maps:
        #map_bulletin(isc_bulletin, "isc_map.png", stas)
        plot_mbs(isc_bulletin, "isc_mbs.png")

    bulletin_name = options.bulletin
    if bulletin_name == "sigvisa":
        evdicts, uadicts_by_sta = load_serialized_from_file(options.serialized)
        bulletin = np.array(sorted([(d["lon"], d["lat"], d["depth"], d["time"], d["mb"], d['score'] if 'score' in d else 0.0) for d in evdicts], key = lambda x: x[3]))
    else:
        bulletin = get_bulletin(time_blocks, origin_type=options.bulletin)

    if options.threshold is not None:
        bulletin = np.asarray([row for row in bulletin if row[5] > options.threshold])

    f, p, r, err, errs = f1_and_error(isc_bulletin, bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    indices = find_matching(isc_bulletin, bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    print_bulletin_matching(bulletin, isc_bulletin, indices)

    if plot_maps:
        map_bulletin(bulletin, isc_bulletin, "%s_map.png" % (bulletin_name), stas)
        plot_mbs(bulletin, "%s_mbs.png" % (bulletin_name))

    print "f1", f
    print "precision", p
    print "recall", r
    print "mean location err %.1fkm" % err[0]
    print
    np.savetxt("%s_errors_%s.txt" % (bulletin_name, label), errs)
    np.savetxt("%s_bulletin_%s.txt" % (bulletin_name, label), bulletin)

    if options.denovo:
        print "DENOVO EVENTS"
        isc_denovo = filter_de_novo(isc_bulletin)
        new_denovo = filter_de_novo(bulletin)
        f, p, r, err, errs = f1_and_error(isc_denovo, bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
        indices = find_matching(isc_denovo, bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
        print_bulletin_matching(bulletin, isc_denovo, indices)
        print "precision", p
        print "recall", r
        print "mean location err %.1fkm" % err[0]



    precs, recalls = precision_recall_curve(bulletin, isc_bulletin, max_delta_deg=options.max_delta_deg, max_delta_time=options.max_delta_time)
    np.savetxt("%s_precisions_%s.txt" % (bulletin_name, label), precs)
    np.savetxt("%s_recalls_%s.txt" % (bulletin_name, label), recalls)



    

def plot_precision_recall(fname, precs, recalls, label=None, *args):

    from matplotlib.figure import Figure
    from sigvisa.plotting.plot import savefig

    fig = Figure(figsize=(8, 5), dpi=300)
    axes = fig.add_subplot(111)

    axes.plot(precs, recalls, label=label)

    for i in range(0, len(args), 3):
        precs = args[i]
        recalls = args[i+1]
        label = args[i+2]
        axes.plot(precs, recalls, label=label)

    axes.set_xlabel("Precision")
    axes.set_ylabel("Recall")
    axes.set_xlim((0, 100))
    axes.set_ylim((0, 100))

    if len(args) > 0:
        axes.legend(loc="upper right")

    savefig(fname, fig, bbox_inches="tight", dpi=300)
    print "saved plot to", fname

def precision_recall_curve(guess, gold, freq=15, cheap=False,  **kwargs):
    guess_by_score = np.asarray(sorted(guess, key = lambda x : -x[5]))
    precs = []
    recalls = []

    indices = find_matching(gold, guess_by_score, **kwargs)


    # for each prefix of the high scoring events
    for i in range(len(guess)):
        if i % freq != 0: continue

        if cheap:
            # to ssve computation, use one matching for all thresholds
            indices_truncated = [(gold_i, guess_i) for (gold_i, guess_i) in indices if guess_i <= i]
            p = 100. * float(len(indices_truncated)) / (i+1)
            r = 100. * float(len(indices_truncated)) / (len(gold))
        else:
            # compute the optimal matching for each thresholded bulletin
            guess_thresholded = guess_by_score[:i]
            f, p, r, err, errs = f1_and_error(gold, guess_thresholded, **kwargs)

        precs.append(p)
        recalls.append(r)

        thresh = guess_by_score[i][5]
        
        #if i % 50 == 0:
        print "%d events threshold %.1f precision %.1f recall %.1f" % (i, thresh, p, r)

    return precs, recalls

if __name__ == "__main__":
    main()
