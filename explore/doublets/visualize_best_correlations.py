"""
Reads the correlation files output by xcorrs.py, and plots all event pairs with correlation above some threshold.
"""
import sigvisa.database.db
from sigvisa.database.dataset import *
import sigvisa.utils.geog
import sys
import itertools
import time
import calendar


import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from sigvisa.plotting.plot import subplot_waveform

from optparse import OptionParser

from sigvisa import *
from sigvisa.signals.io import fetch_waveform
from sigvisa.explore.doublets.xcorr_pairs import extracted_wave_fname, xcorr
from sigvisa.source.event import get_event
from sigvisa.plotting.event_heatmap import get_eventHeatmap
from sigvisa.models.wiggles.fourier_features import FourierFeatures


def normalize(x):
    return x / np.std(x) - np.mean(x)


def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station")
    parser.add_option("-c", "--chan", dest="chan", default="BHZ", type="str", help="channel to correlate")
    parser.add_option(
        "-f", "--filter_str", dest="filter_str", default="freq_0.8_3.5", type="str", help="filter string to process waveforms")
    parser.add_option(
        "--window_len", dest="window_len", default=30.0, type=float, help="length of window to extract / cross-correlate")
    parser.add_option("-t", "--xc_threshold", dest="xc_threshold", default=0.75, type="float",
                      help="plot all events with cross-correlation peak above this threshold")
    parser.add_option(
        "-i", "--xcorrsfile", dest="xcorrsfile", default=None, type="str", help="load potential doublets from this file")
    parser.add_option("-o", "--outfile", dest="outfile", default=None, type="str", help="save pdf plots to this file")

    (options, args) = parser.parse_args()

    sta = options.sta
    chan = options.chan
    filter_str = options.filter_str
    window_len = options.window_len
    xc_threshold = options.xc_threshold

    pp = PdfPages(options.outfile)

    s = Sigvisa()

    # input format: evid1, evid2, dist, atime1, atime2, xc
    f = open(options.xcorrsfile)

    pairs = [[z.strip() for z in line.split(',')] for line in f]
    threshold_pairs = [p for p in pairs if float(p[5]) >= xc_threshold]

    PAD = 10

    evids = set()

    for pair in threshold_pairs:
        evid1 = int(pair[0])
        evid2 = int(pair[1])
        dist = float(pair[2])
        atime1 = float(pair[3])
        atime2 = float(pair[4])
        xc = float(pair[5])

        evids.add(evid1)
        evids.add(evid2)

        wave1_loaded = fetch_waveform(sta, chan, atime1 - 1, atime1 + window_len, pad_seconds=PAD)
        filtered1 = wave1_loaded.filter(filter_str)
        pad_samples = filtered1['srate'] * PAD

        wave2_loaded = fetch_waveform(sta, chan, atime2 - 1, atime2 + window_len, pad_seconds=PAD)
        filtered2 = wave2_loaded.filter(filter_str)

        # plot the doublet
        fig = plt.figure()

        gs = gridspec.GridSpec(2, 2)
        gs.update(left=0.1, right=0.95, hspace=1)
        axes = None

        axes = plt.subplot(gs[0, 0], sharey=axes)
        subplot_waveform(filtered1, axes)

        axes = plt.subplot(gs[0, 1], sharey=axes)
        subplot_waveform(filtered2, axes)

        axes = plt.subplot(gs[1, 0:2], sharey=None)

        p1 = normalize(filtered1.data)
        p2 = normalize(filtered2.data)
        p1 = p1[pad_samples:-pad_samples]
        p2 = p2[pad_samples:-pad_samples]

#        print p1
#        import pdb
#        pdb.set_trace()

        xc, offset = xcorr(p1, p2)
        offset = offset / filtered1['srate']

        ev1 = get_event(evid1)
        ev2 = get_event(evid2)
        plt.suptitle("evids %d, %d: xc %.3f offset %.3f\n(%.2f, %.2f) vs (%.2f, %.2f): %.2fkm" % (evid1, evid2, xc,
                     offset, ev1.lon, ev1.lat, ev2.lon, ev2.lat, dist))

        x = np.linspace(-1, window_len, len(p1))
        axes.plot(x, p1, color="red")
        axes.plot(x + offset, p2, color="blue")

        print "plotting", pair

        pp.savefig()

        # plot the projected-onto-fourier-features version
        ff = FourierFeatures(fundamental=0.2, min_freq=0.8, max_freq=3.5)

        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1)
        gs.update(left=0.1, right=0.95, hspace=1)
        axes = None

        axes = plt.subplot(gs[0, 0], sharey=axes)
        axes.plot(x, ff.project_down(p1), color="green")
        axes.plot(x + offset, ff.project_down(p2), color="purple")

        axes = plt.subplot(gs[1, 0], sharey=axes)
        axes.plot(x + offset, p1, color="red")
        axes.plot(x + offset, ff.project_down(p1), color="green")

        axes = plt.subplot(gs[2, 0], sharey=axes)
        axes.plot(x + offset, p2, color="blue")
        axes.plot(x + offset, ff.project_down(p2), color="purple")

        plt.suptitle("same as last but comparing fourier projections")

        pp.savefig()

    pp.close()

if __name__ == "__main__":
    main()
