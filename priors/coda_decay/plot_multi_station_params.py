import os, sys, traceback
import numpy as np, scipy


from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import priors.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util

(EVID_COL, SITEID_COL, PHASEID_COL, DISTANCE_COL, AZI_COL, DET_AZI_COL, LON_COL, LAT_COL, B_COL, PEAK_COL, PEAKFIT_COL, MB_COL, SNR_COL, CODA_LEN_COL, AVG_COST_COL, MAX_COST_COL, MIN_AZI_COL, NOISE_FLOOR_COL, NUM_COLS) = range(18+1)


def label_station(plot_idx, fname):
    ax = plt.subplot(len(sys.argv)-1, 9, plot_idx)
    plt.text(0.5, 0.5,fname,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax.transAxes,
             fontsize = 18)

def write_plot(plot_idx, band, l):

    if len(l.shape) < 2 or l.shape[1] != NUM_COLS or l.shape[0] < 2:
        return
    
    plt.subplot(len(sys.argv)-1, 9, plot_idx+1)
    plt.xlabel("distance (km)")
    plt.ylabel("b")
    plt.ylim([-0.1, 0])
    plt.xlim([0, 3000])
    plt.plot(l[:, DISTANCE_COL], l[:, B_COL], 'ro')

    plt.subplot(len(sys.argv)-1, 9, plot_idx+5)
    plt.xlabel("azimuth (deg)")
    plt.ylabel("b")
    plt.ylim([-0.1, 0])
    plt.xlim([0, 360])
    plt.plot(l[:, AZI_COL], l[:, B_COL], 'ro')


def main():
    pp = PdfPages('logs/shape_params_all_stations.pdf')
    plt.figure(figsize = (55,45))
    
    sta_idx = 0
    for fname in sys.argv:

        if not fname.endswith("data"):
            continue

        print "opening fname", fname
        f = open(fname, 'r')
        l = None
        plot_idx = sta_idx * 9 + 1

        for line in f:
            if line[0] == 'n':
            
                if l is not None:
                    l = clean_points(l)
                    write_plot(plot_idx, band, l)
                    plot_idx = plot_idx + 1

                band = line[:-1]
                l = None
                idx = 0
            else:
                new_row = np.array(  map( lambda x : float(x), (line[:-1]).split()))

                if l is None:
                    l = new_row
                else:
                    try:
                        l = np.vstack([l, new_row])
                    except:
                        break
            #(l[idx, 0], l[idx, 1], l[idx, 2]) = (line[:-1]).split()
                idx = idx + 1
 

        if l is not None:
            l = clean_points(l)

            write_plot(plot_idx, band, l)
            plot_idx = plot_idx + 1

            label_station(sta_idx*9+1, fname[12:23])
            sta_idx = sta_idx + 1
        else:
            print "weirdness reading station", fname, sta_idx
        
    pp.savefig()
    pp.close()

if __name__ == "__main__":
    main()








