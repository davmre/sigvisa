import os, errno, sys, time, traceback
import numpy as np, scipy
from guppy import hpy; hp = hpy()

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


import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.plot_coda_decays import *

def arrival_peak_offset(trace, window_start_offset, window_end_offset = None):
    srate = trace.stats.sampling_rate

    if window_end_offset is None:
        window_end_offset = window_start_offset + 15

    i = np.floor(window_start_offset*srate)
    j = np.floor(window_end_offset*srate)

    print window_start_offset, window_end_offset, i, j, srate, trace.data.shape

    pt = np.argmax(trace.data[i:j]) / srate
    return (pt +window_start_offset, trace.data[(pt+window_start_offset) * srate ])

def c_cost(smoothed, phaseids, params):

#    noise_floor = params[-1]
#    params = np.reshape(params[:-1], (len(phaseids), -1))
    noise_floor = smoothed.stats.noise_floor
    params = np.reshape(params, (len(phaseids), -1))

    for i, pid in enumerate(phaseids):
        if np.isnan(params[i, PEAK_HEIGHT_PARAM]) or np.isnan(params[i, CODA_HEIGHT_PARAM]):
            return np.float('inf')
        if params[i, PEAK_HEIGHT_PARAM] < 1:
            return np.float('inf')
        if params[i, CODA_HEIGHT_PARAM] > 1.1 * params[i, PEAK_HEIGHT_PARAM]:
            return np.float('inf')
        if params[i, CODA_DECAY_PARAM] >= 0 or params[i, CODA_DECAY_PARAM] >= 0:
            return np.float('inf')
        if params[i, PEAK_DECAY_PARAM] < 0 or params[i, PEAK_DECAY_PARAM] < 0:
            return np.float('inf')

    tr = imitate_envelope(smoothed, phaseids, params)
    c = logenv_l1_cost(smoothed.data, tr.data)

    return c


def fit_envelope(arrivals, smoothed):
    arr_bounds = [ (0, 15), (0, None) , (0, None), (0, None), (-.2, 0) ]
    arrivals = [arr for arr in arrivals if arr is not None]

    start_params = np.zeros((len(arrivals), NUM_PARAMS))
    bounds = []
    phaseids = []
    arr_times = np.zeros((len(arrivals), 1))
    for i, arr in enumerate(arrivals):
        time = arr[AR_TIME_COL]
        (peak_offset_time, peak_height) = arrival_peak_offset(smoothed, time - smoothed.stats.starttime_unix)

        start_params[i, PEAK_OFFSET_PARAM] = peak_offset_time
        start_params[i, PEAK_HEIGHT_PARAM] = peak_height
        start_params[i, PEAK_DECAY_PARAM] = .5
        start_params[i, CODA_HEIGHT_PARAM] = peak_height
        start_params[i, CODA_DECAY_PARAM] = -0.02

        bounds = bounds + arr_bounds
        phaseids.append(arr[AR_PHASEID_COL])
        arr_times[i] = time

    start_params = start_params[:, 1:].flatten()

    f = lambda params : c_cost(smoothed, phaseids, np.hstack([arr_times, np.reshape(params, (2, -1))]))

    best_params, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f, start_params, approx_grad=1, bounds=bounds)
    best_params = np.hstack([arr_times, np.reshape(best_params, (2, -1))])
    return best_params, phaseids, best_cost


def get_first_p_s_arrivals(cursor, event, siteid):
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
    sql_query="SELECT l.time, l.azimuth, l.snr, pid.id, sid.id FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and leba.arid=l.arid and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase order by l.sta" % (event[EV_EVID_COL], phase_condition, siteid)
    cursor.execute(sql_query)
    arrivals = np.array(cursor.fetchall())

    first_p_arrival = None
    for arrival in arrivals:
        if int(arrival[AR_PHASEID_COL]) in P_PHASEIDS:
            first_p_arrival = arrival
            break
    first_s_arrival = None
    for arrival in arrivals:
        if int(arrival[AR_PHASEID_COL]) in S_PHASEIDS:
            first_s_arrival = arrival
            break

    return (first_p_arrival, first_s_arrival)


def get_densest_azi(cursor, siteid):
    max_azi_count = -1
    max_azi = -1
    max_azi_condition = ""
    for azi in np.linspace(0, 330, 12):
        if azi == 330:
            azi_condition = "(l.azimuth between 0 and 30 or l.azimuth between 330 and 360)"
        else:
            azi_condition = "l.azimuth between %f and %f" % (azi, azi+60)
        sql_query="SELECT count(distinct(l.arid)) FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and sid.sta=l.sta and sid.id=%d and %s" % (siteid, azi_condition)
        cursor.execute(sql_query)
        azi_count = cursor.fetchall()
        if azi_count > max_azi_count:
            max_azi_count = azi_count
            max_azi = azi
            max_azi_condition = azi_condition
    print "max azi is", max_azi, "with count", max_azi_count
    return max_azi


def main():
# boilerplate initialization of various things
    siteid = int(sys.argv[1])
    cursor = db.connect().cursor()
    sites = read_sites(cursor)
    st  = 1237680000
    et = st + 3600*24
    site_up = read_uptime(cursor, st, et)
    detections, arid2num = read_detections(cursor, st, et, arrival_table="leb_arrival", noarrays=True)
    phasenames, phasetimedef = read_phases(cursor)
    earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
    netmodel = learn.load_netvisa("parameters", st, et, detections, site_up, sites, phasenames, phasetimedef)

    max_azi = get_densest_azi(cursor, siteid)


# want to select all events, with certain properties, which have a P or S phase detected at this station
    phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
    sql_query="SELECT distinct lebo.mb, lebo.lon, lebo.lat, lebo.evid, lebo.time, lebo.depth FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase order by l.sta" % (phase_condition, siteid)
    cursor.execute(sql_query)
    events = np.array(cursor.fetchall())

    bands = ['narrow_logenvelope_4.00_6.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_1.00_1.50', 'narrow_logenvelope_0.70_1.00']
    short_bands = [b[19:] for b in bands]

    runid = int(time.time())
    base_coda_dir = get_base_dir(siteid, None, runid)
    print "writing data to directory", base_coda_dir

    f = open(os.path.join(base_coda_dir, 'all_data'), 'w')
    for b in bands:
        f.write(b + " ")
    f.write("\n")

    for event in events:

        distance = utils.geog.dist_km((event[EV_LON_COL], event[EV_LAT_COL]), (sites[siteid-1][0], sites[siteid-1][1]))
        azimuth = utils.geog.azimuth((event[EV_LON_COL], event[EV_LAT_COL]), (sites[siteid-1][0], sites[siteid-1][1]))

        (first_p_arrival, first_s_arrival) = get_first_p_s_arrivals(cursor, event, siteid)

        if first_p_arrival is not None and first_s_arrival is not None and first_p_arrival[AR_TIME_COL] > first_s_arrival[AR_TIME_COL]:
            print "skipping evid %d because S comes before P..." % (event[EV_EVID_COL])
            continue

        try:
            (arrival_segment, noise_segment, other_arrivals, other_arrival_phases) = load_signal_slice(cursor, event[EV_EVID_COL], siteid, load_noise = True)
        except:
            print traceback.format_exc()
            continue

        for (band_idx, band) in enumerate(bands):
            short_band = short_bands[band_idx]


            vert_noise_floor = arrival_segment[0]["BHZ"][band].stats.noise_floor
            horiz_noise_floor = arrival_segment[0]["horiz_avg"][band].stats.noise_floor

            try:
                vert_smoothed, horiz_smoothed = smoothed_traces(arrival_segment, band)
            except:
                print traceback.format_exc()
                continue

            # DO THE FITTING
            fit_vert_params, phaseids, vert_cost = fit_envelope([first_p_arrival, first_s_arrival], vert_smoothed)
            fit_horiz_params, phaseids, horiz_cost = fit_envelope([first_p_arrival, first_s_arrival], horiz_smoothed)


            # plot!
            pdf_dir = get_dir(os.path.join(base_coda_dir, short_band))
            pp = PdfPages(os.path.join(pdf_dir, str(int(event[EV_EVID_COL])) + ".pdf"))
            gen_title = lambda event, fit: "%s evid %d siteid %d mb %f \n dist %f azi %f \n p: %s \n s: %s " % (band, event[EV_EVID_COL], siteid, event[EV_MB_COL], distance, azimuth, fit[0,:],fit[1,:])
            try:
                plot_channels(pp, vert_smoothed, fit_vert_params, phaseids, horiz_smoothed, fit_horiz_params, title = gen_title(event, fit_vert_params))
            except:
                print "error plotting:"
                print traceback.format_exc()
            print "wrote plot", os.path.join(pdf_dir, str(int(event[EV_EVID_COL])) + ".pdf")


            # write a row to the database
            #TODO

            pp.close()

        del arrival_segment
        del noise_segment
        del other_arrivals
        del other_arrival_phases
    #    print hp.heap()

    f.close()

if __name__ == "__main__":
    main()






