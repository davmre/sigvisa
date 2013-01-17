"""
Reads the correlation files output by xcorrs.py, and plots all event pairs with correlation above some threshold.
"""
import database.db
from database.dataset import *
import utils.geog
import sys
import itertools
import time, calendar


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from plotting.plot import subplot_waveform
    
from optparse import OptionParser

from sigvisa import *
from signals.io import fetch_waveform
from explore.doublets.xcorr_pairs import extracted_wave_fname, xcorr
from source.event import get_event
from plotting.event_heatmap import get_eventHeatmap
from signals.waveform_matching.fourier_features import FourierFeatures
from explore.doublets.closest_event_pairs_at_sta import get_first_arrivals
from explore.doublets.xcorr_pairs import extract_phase_window
from train_model import train_and_save_models, read_training_events

from gpr.gp import GaussianProcess
from learn.SpatialGP import SpatialGP


def normalize(x):
    return x/np.std(x) - np.mean(x)


def plot_events(sta, center, training_events, pred_points, highlighted, width=1):
        # plot the training events
    hm = EventHeatmap(f=lambda x,y: 0, center = (center.lon, center.lat), width = width)

    hm.plot_earth()
    hm.plot_locations([(ev.lon, ev.lat) for ev in training_events], marker="x", ms=12, mfc="none", mec="red", mew=2, alpha=.6)

    hm.plot_locations([pp[:2] for pp in pred_points], marker=".", ms=12, mfc="none", mec="white", mew=2, alpha=0.5)
    hm.plot_locations([highlighted[0:2],], marker=".", ms=15, mfc="none", mec="white", mew=2, alpha=1)

    hm.plot_locations(((center.lon, center.lat),), labels=None,
                        marker="*", ms=12, mfc="none", mec="#44FF44", mew=2, alpha=1)


def predict_signal_at_point(pt, models, ff, n):
    freqs = sorted(models.keys())

    features = np.zeros((len(freqs), 2))
    for (i, freq) in enumerate(freqs):
        for (j, component) in enumerate(["amp", "phase"]):
            features[i][j] = models[freq][component].predict(pt)
#    print "got features", features

    signal = ff.signal_from_features(features, len_seconds = n / ff.srate)
    return signal, features

def load_models_from_dir(model_folder):
    models = NestedDict()
    
    for m in os.listdir(model_folder):
        model = SpatialGP(fname=os.path.join(model_folder, m))
        hz, f = m[:-12].split('_')
#        print "loaded", hz, f
        models[hz][f] = model
    return models

def main():
    parser = OptionParser()

    parser.add_option("-m", "--model_folder", dest="model_folder", default=None, type="str", help="folder to save learned model")
    parser.add_option("--center", dest="center_evid", default=None, type="int", help="evid to center on")
    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station")
    parser.add_option("-c", "--chan", dest="chan", default="BHZ", type="str", help="channel to correlate")
    parser.add_option("-f", "--filter_str", dest="filter_str", default="freq_0.8_3.5", type="str", help="filter string to process waveforms")
    parser.add_option("--window_len", dest="window_len", default=30.0, type=float, help="length of window to extract / cross-correlate")
    parser.add_option("-o", "--outfile", dest="outfile", default=None, type="str", help="save pdf plots to this file")

    (options, args) = parser.parse_args()
        

    sta = options.sta
    chan = options.chan
    filter_str = options.filter_str
    window_len = options.window_len

    center_evid = options.center_evid

    model_folder = os.path.realpath(options.model_folder)




    s = Sigvisa()

    # input format: evid1, evid2, dist, atime1, atime2, xc

    fundamental = 0.1
    min_freq=0.8
    max_freq=3.5
    ff = FourierFeatures(fundamental=fundamental, min_freq=min_freq, max_freq=max_freq)



    # load true waveform for the query point
    center = Event(center_evid)
    arriving_events, arrival_dict  = get_first_arrivals([center,], sta)
    (atime, phase) = arrival_dict[center.evid]
    true_center_wave = normalize(extract_phase_window(sta, chan, phase, atime, window_len, filter_str, center.evid))


    training_events = read_training_events(sta, center.time-7*24*3600, center.time+7*24*3600, 3.5, 10, center, 100)
    
    prediction_points = []
    for i in np.linspace(-.2, .2, 41):
        pt = (center.lon+i, center.lat, center.depth)
        prediction_points.append(pt)


    for event in training_events:
        print event
    print "center:", center



#    pp = PdfPages(options.outfile + "map.pdf")
    for (i, pt) in enumerate( prediction_points):
        plot_events(sta, center, training_events, prediction_points, pt)
        plt.savefig("%02d_map.png" % (i))

    print "all points", prediction_points

    return

    models = load_models_from_dir(model_folder)
    x = np.linspace(-1, window_len, len(true_center_wave))
    for (i, pt) in enumerate(prediction_points):
        predicted_wave, predicted_features = predict_signal_at_point(pt, models, ff, len(true_center_wave))

        fig = plt.figure()
        gs = gridspec.GridSpec(2,1)
        gs.update(left=0.1, right=0.95, hspace=1)

        axes = None
        axes = plt.subplot(gs[0:2,0], sharey=axes)
        axes.plot(x, predicted_wave, color="red")
        axes.plot(x, true_center_wave, color="blue")

        plt.ylim([-5,5])
        
#        axes = plt.subplot(gs[1,0], sharey=axes)


#        plt.suptitle(str(pt))

        plt.savefig("%02d_signal.png" % (i))


#    pp.close()

if __name__ == "__main__":
    main()
