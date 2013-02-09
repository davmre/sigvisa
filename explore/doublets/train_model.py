import database.db
from database.dataset import *
import utils.geog
import sys
import itertools
import time, calendar


import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from plotting.plot import subplot_waveform

from database.signal_data import ensure_dir_exists

from optparse import OptionParser

from sigvisa import *
from signals.io import fetch_waveform
from models.wiggles.fourier_features import FourierFeatures

from source.event import get_event
from explore.doublets.closest_event_pairs_at_sta import get_first_arrivals
from explore.doublets.xcorr_pairs import extract_phase_window

from plotting.event_heatmap import get_eventHeatmap


#from gpr import munge, kernels, evaluate, learn, distributions, plot
from gpr.gp import GaussianProcess
from models.spatial_regression.SpatialGP import SpatialGP


def train_and_save_models(training_events, sta, chan, window_len, filter_str, ff, model_folder, amp_params = [.05, .05, 1.5], phase_params = [.05, .05, 4]):
    # now load signals for each and compute features
    arriving_events, arrival_dict  = get_first_arrivals(training_events, sta)

    signals = []
    features = []
    loaded_events = []
    for ev in arriving_events:
        (atime, phase) = arrival_dict[ev.evid]

        try:
            wave = extract_phase_window(sta, chan, phase, atime, window_len, filter_str, ev.evid)
            feature = ff.basis_decomposition(wave)
            signals.append(wave)
            features.append(feature)
            loaded_events.append(ev)
        except Exception as e:
            print e
            continue

#    print "saving models to", model_folder

    # for each feature, train a GP model on all events
    ensure_dir_exists(model_folder)

    feature_shape = features[0].shape
    n_features = feature_shape[0]*feature_shape[1]
    X = np.array([(ev.lon, ev.lat, ev.depth) for ev in loaded_events])
    for i in range(feature_shape[0]):
        for j in range(feature_shape[1]):
            feature_name = str(i*ff.fundamental + ff.min_freq)
            feature_name += "_amp" if j==0 else "_phase"

            y = [f[i][j] for f in features]
            gp = SpatialGP(X, y, distfn_str="lld", kernel_params = amp_params if j==0 else phase_params)

            fname = os.path.join(model_folder, feature_name + ".gpmodel")
            gp.save_trained_model(fname)
#            print "saved model", fname, "trained on", len(y), "events", "mean", gp.mu


def read_training_events(sta, st, et, min_mb, max_mb, center, width):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    evids = read_evids_detected_at_station(cursor, sta, st, et, min_mb = min_mb, max_mb = max_mb)
    events = [get_event(evid) for evid in evids]


    def ev_distkm(ev1, ev2):
        return utils.geog.dist_km((ev1.lon, ev1.lat), (ev2.lon, ev2.lat))

    training_events = []
    for event in events:
        if event.evid != center.evid and ev_distkm(center, event) < width:
#        if ev_distkm(center, event) < width:
            training_events.append(event)

    return training_events


def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station")
    parser.add_option("-c", "--chan", dest="chan", default="BHZ", type="str", help="channel to correlate")
    parser.add_option("--center", dest="center_evid", default=None, type="int", help="evid to center on")

    parser.add_option("--days_before", dest="days_before", default=7, type="float", help="set time window reletive to center event")
    parser.add_option("--days_after", dest="days_after", default=7, type="float", help="set time window reletive to center event")
    parser.add_option("--width", dest="width", default=100, type="float", help="only load events within a distance of width km from the center event.")
    parser.add_option("-f", "--filter_str", dest="filter_str", default="freq_0.8_3.5", type="str", help="filter string to process waveforms")
    parser.add_option("--window_len", dest="window_len", default=30.0, type=float, help="length of window to extract / cross-correlate")
    parser.add_option("-o", "--model_folder", dest="model_folder", default=None, type="str", help="folder to save learned model")
    parser.add_option("--min_mb", dest="min_mb", default=1, type="float", help="exclude all events with mb less than this value (0)")
    parser.add_option("--max_mb", dest="max_mb", default=99, type="float", help="exclude all events with mb greater than this value (10)")

    (options, args) = parser.parse_args()

    s = Sigvisa()

    sta = options.sta
    chan = options.chan
    filter_str = options.filter_str
    window_len = options.window_len
    center_evid = options.center_evid
    width = options.width
    days_before = options.days_before
    days_after = options.days_after
    model_folder = os.path.realpath(options.model_folder)

    center = Event(center_evid)
    st = center.time - days_before*24*60*60
    et = center.time + days_after*24*60*60


    # first find the events we'll train with: everything near the center, but *not* the center itself
    training_events = read_training_events(sta, st, et, options.min_mb, options.max_mb, center, width)

    # plot the training events
    pp = PdfPages("train_set.pdf")
    hm = EventHeatmap(f=lambda x,y: 0, center = (center.lon, center.lat), width = width/70.0)
    hm.add_events([(ev.lon, ev.lat) for ev in training_events])
    hm.add_stations([sta,])
    hm.set_true_event(center.lon, center.lat)
    hm.plot()
    pp.savefig()
    pp.close()

    fundamental = 0.1
    min_freq=0.8
    max_freq=3.5
    ff = FourierFeatures(fundamental=fundamental, min_freq=min_freq, max_freq=max_freq)

    print "ts"
    train_and_save_models(training_events, sta, chan, window_len, filter_str, ff, model_folder)
    print "done"

if __name__ == "__main__":
    main()
