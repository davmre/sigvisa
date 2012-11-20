import database.db
from database.dataset import *
import utils.geog
import sys, hashlib
import itertools
import time, calendar


import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from plotting.plot import subplot_waveform
    
from optparse import OptionParser

from sigvisa import *
from signals.io import fetch_waveform
from explore.doublets.xcorr_pairs import extracted_wave_fname, xcorr
from source.event import Event
from plotting.event_heatmap import EventHeatmap
from signals.waveform_matching.fourier_features import FourierFeatures
from explore.doublets.closest_event_pairs_at_sta import get_first_arrivals
from explore.doublets.xcorr_pairs import extract_phase_window

from gpr.gp import GaussianProcess
from learn.SpatialGP import SpatialGP

from predict_doublet import predict_signal_at_point, load_models_from_dir
from train_model import train_and_save_models, read_training_events
from visualize_best_correlations import normalize

def main():
    parser = OptionParser()
    
    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station")
    parser.add_option("-c", "--chan", dest="chan", default="BHZ", type="str", help="channel to correlate")
    parser.add_option("--center", dest="center_evid", default=None, type="int", help="evid to center on")
    
    parser.add_option("-f", "--filter_str", dest="filter_str", default="freq_0.8_3.5", type="str", help="filter string to process waveforms")
    parser.add_option("--window_len", dest="window_len", default=30.0, type=float, help="length of window to extract / cross-correlate")
    parser.add_option("-o", "--model_folder", dest="model_folder", default=None, type="str", help="folder to save learned model")
    parser.add_option("--days_before", dest="days_before", default=7, type="float", help="set time window reletive to center event")
    parser.add_option("--days_after", dest="days_after", default=7, type="float", help="set time window reletive to center event")
    parser.add_option("--width", dest="width", default=100, type="float", help="only load events within a distance of width km from the center event.")
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


    fundamental = 0.1
    min_freq=0.8
    max_freq=3.5
    ff = FourierFeatures(fundamental=fundamental, min_freq=min_freq, max_freq=max_freq)


    # laod true waveform at the center point, for evaluation
    center = Event(center_evid)
    arriving_events, arrival_dict  = get_first_arrivals([center,], sta)
    (atime, phase) = arrival_dict[center.evid]
    true_center_wave = normalize(extract_phase_window(sta, chan, phase, atime, window_len, filter_str, center.evid))
    true_center_features = ff.basis_decomposition(true_center_wave)
    projected_center_wave = ff.signal_from_features(true_center_features)

    # first find the events we'll train with: everything near the center, but *not* the center itself
    training_events = read_training_events(sta, st, et, options.min_mb, options.max_mb, center, width)


    W =  [1.5, 4.0, 10.0]
    SN = [.05, 1]
    SF = [.05, 1]

    pp = PdfPages("logs/center_%d_models.pdf" % center.evid)

    # plot the training events
    hm = EventHeatmap(f=lambda x,y: 0, center = (center.lon, center.lat), width = width/70.0)
    hm.add_events([(ev.lon, ev.lat) for ev in training_events])
    hm.add_stations([sta,])
    hm.set_true_event(center.lon, center.lat)
    hm.plot()
    pp.savefig()


    amp_params = itertools.product(SN, SF, W)
    phase_params = itertools.product(SN, SF, W)
    for (ap, phase_p) in itertools.product(amp_params, phase_params):

        idstr = str(ap) + str(phase_p) + str(center.evid)
        hash = "_" + hashlib.sha1(idstr).hexdigest()[0:6]

        try:
            train_and_save_models(training_events, sta, chan, window_len, filter_str, ff, model_folder+hash, amp_params=ap, phase_params=phase_p)

            models = load_models_from_dir(model_folder+hash)

            predicted_center_wave, predicted_features = predict_signal_at_point((center.lon, center.lat, center.depth), models, ff, len(true_center_wave))

            feature_error = np.linalg.norm((predicted_features - true_center_features).flatten())
            signal_error = np.linalg.norm((predicted_center_wave - true_center_wave).flatten())

            print "evid", center.evid, "params", ap, phase_p, "feature error", feature_error, "signal error", signal_error

            fig = plt.figure()
            gs = gridspec.GridSpec(3,1)
            gs.update(left=0.1, right=0.95, hspace=1)
            x = np.linspace(-1, window_len, len(true_center_wave))

            axes = None
            axes = plt.subplot(gs[0,0], sharey=axes)
            axes.plot(x, predicted_center_wave, color="red")

            axes = plt.subplot(gs[1,0], sharey=axes)
            axes.plot(x, true_center_wave, color="blue")

            axes = plt.subplot(gs[2,0], sharey=axes)
            axes.plot(x, predicted_center_wave, color="red")
            axes.plot(x, true_center_wave, color="blue")

            title_str = "amp params ($\sigma^2_n = %.2f, \sigma^2_f = %.2f, w = %.2f$)\n" % ap
            title_str += "phase params ($\sigma^2_n = %.2f, \sigma^2_f = %.2f, w = %.2f$)\n" % phase_p
            title_str += "feature error %.3f, signal error %.3f" % (feature_error, signal_error)
            plt.suptitle(title_str)
            pp.savefig()

        except Exception as e:
            print e
            continue
    
    pp.close()

if __name__ == "__main__":
    main()
