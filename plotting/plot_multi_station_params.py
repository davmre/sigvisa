import os, sys, traceback
import numpy as np, scipy


from sigvisa.database.dataset import *
from sigvisa.database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import signals.SignalPrior
from sigvisa.utils.waveform import *
import sigvisa.utils.geog
import obspy.signal.util

from sigvisa.signals.coda_decay_common import *




def label_station(plot_idx, fname):
    ax = plt.subplot(len(sys.argv)-1, 5, plot_idx)
    plt.text(0.5, 0.5,fname,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax.transAxes,
             fontsize = 18)




def write_plot(plot_idx, l, get_col, P):

    if l is None or len(l.shape) < 2 or l.shape[0] < 2:
        return

    pl = np.zeros((l.shape[0], 3))
    for i in range(l.shape[0]):
        pl[i, 0] = l[i, DISTANCE_COL]
        pl[i, 1] = l[i, AZI_COL]
        pl[i, 2] = get_col(l[i,:])


    plt.subplot(len(sys.argv)-1, 5, plot_idx+1)
    plt.xlabel("distance (km)")
    plt.ylabel("b")
    plt.ylim([-.1, 0])
    plt.xlim([0, 10000 if P else 3000])
    plt.plot(pl[:, 0], pl[:, 2], 'ro')

#    plt.subplot(len(sys.argv)-1, 9, plot_idx+5)
#    plt.xlabel("azimuth (deg)")
#    plt.ylabel("b")
#    plt.ylim([20, 5])
#    plt.xlim([0, 360])
#    plt.plot(pl[:, 1], pl[:, 2], 'ro')

def write_grid(in_dirs, out_fname, get_col = None, P=True, vert=True):

    if get_col is None:
        if P and vert:
            b_col = VERT_P_FIT_B
        elif P and not vert:
            b_col = HORIZ_P_FIT_B
        elif not P and vert:
            b_col = VERT_S_FIT_B
        elif not P and not vert:
            b_col = HORIZ_S_FIT_B
        get_col = lambda x : x[b_col]

    pp = PdfPages(out_fname)
    plt.figure(figsize = (35,45))
    sta_idx = 0
    for base_coda_dir in in_dirs:

        fname = os.path.join(base_coda_dir, 'all_data')
        all_data, bands = read_shape_data(fname)

        plot_idx = sta_idx * 5 + 1

        for (band_idx, band) in enumerate(bands):
            try:
                band_data = extract_band(all_data, band_idx)
                clean_data = clean_points(band_data, P=P, vert=vert)

                write_plot(plot_idx, clean_data, get_col, P)
                plot_idx = plot_idx + 1
            except:
                print traceback.format_exc()
                continue

        label_station(sta_idx*5+1, fname.split("_")[1])
        sta_idx = sta_idx + 1

    pp.savefig()
    pp.close()

def main():
    write_grid(sys.argv[1:], 'logs/shape_params_all_stations_P.pdf', P=True, vert=True)
    write_grid(sys.argv[1:], 'logs/shape_params_all_stations_S.pdf', P=False, vert=False)

    cursor = db.connect().cursor()
    sites = read_sites(cursor)
    st  = 1237680000
    et = st + 3600*24
    site_up = read_uptime(cursor, st, et)
    detections, arid2num = read_detections(cursor, st, et, arrival_table="leb_arrival", noarrays=True)
    phasenames, phasetimedef = read_phases(cursor)
    earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
    netmodel = learn.load_netvisa("parameters", st, et, detections, site_up, sites, phasenames, phasetimedef)

#    get_col = lambda x : pred_arrtime(cursor, x, netmodel, P_PHASEID_COL, VERT_P_FIT_PEAK_OFFSET)
#    write_grid(sys.argv[1:], 'logs/pred_offset_all_stations_P.pdf', get_col=get_col, P=True, vert=True)
#    get_col = lambda x : pred_arrtime(cursor, x, netmodel, S_PHASEID_COL, HORIZ_S_FIT_PEAK_OFFSET)#
#    write_grid(sys.argv[1:], 'logs/pred_offset_all_stations_S.pdf', get_col=get_col, P=False, vert=False)



#    get_col = lambda x : x[VERT_P_FIT_PEAK_OFFSET] - VERT_P_FIT_PHASE_START_TIME]
#    write_grid(sys.argv[1:], 'logs/peak_offset_all_stations_P.pdf', get_col=get_col, P=True, vert=True)
#    get_col = lambda x : x[HORIZ_S_FIT_PEAK_OFFSET] - x[HORIZ_S_FIT_PHASE_START_TIME]
#    write_grid(sys.argv[1:], 'logs/peak_offset_all_stations_S.pdf', get_col=get_col, P=False, vert=False)

#    get_col = lambda x : x[VERT_P_FIT_PEAK_HEIGHT]
#    write_grid(sys.argv[1:], 'logs/peak_height_all_stations_P.pdf', get_col=get_col, P=True, vert=True)
#    get_col = lambda x : x[HORIZ_S_FIT_PEAK_HEIGHT]
#    write_grid(sys.argv[1:], 'logs/peak_height_all_stations_S.pdf', get_col=get_col, P=False, vert=False)



#    get_col = lambda x : x[VERT_P_FIT_HEIGHT] - x[VERT_P_FIT_B] * (x[VERT_P_FIT_CODA_START_OFFSET] - x[VERT_P_FIT_PEAK_OFFSET])
#    write_grid(sys.argv[1:], 'logs/fit_height_all_stations_P.pdf', P=True, vert=True)
#    get_col = lambda x : x[HORIZ_S_FIT_HEIGHT] - x[HORIZ_S_FIT_B] * (x[HORIZ_S_FIT_CODA_START_OFFSET] - x[HORIZ_S_FIT_PEAK_OFFSET])
#    write_grid(sys.argv[1:], 'logs/fit_height_all_stations_S.pdf', get_col=get_col, P=False, vert=False)

if __name__ == "__main__":
    main()
