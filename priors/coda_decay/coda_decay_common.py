import os, errno, sys, time, traceback
import numpy as np, scipy, scipy.stats

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import sigvisa
import learn, sigvisa_util
import signals.SignalPrior
from signals.armodel.learner import ARLearner
from signals.armodel.model import ARModel, ErrorModel
from utils.waveform import *
from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.geog
import obspy.signal.util


P_PHASES = ['P', 'Pn']
P_PHASEIDS = [1,2]

S_PHASES = ['S', 'Sn']
S_PHASEIDS = [4, 5]

MIN_SEGMENT_LENGTH = 40

min_p_coda_length = 30
min_s_coda_length = 45

avg_cost_bound = 0.2

bands = ['narrow_envelope_2.00_3.00', 'narrow_envelope_4.00_6.00', 'narrow_envelope_1.00_1.50', 'narrow_envelope_0.70_1.00']
#bands = ["narrow_envelope_2.00_3.00",]
chans = ["BHZ","BHE", "BHN"]

(FIT_EVID, FIT_MB, FIT_LON, FIT_LAT, FIT_DEPTH, FIT_PHASEID, FIT_PEAK_DELAY, FIT_CODA_HEIGHT, FIT_CODA_DECAY, FIT_DISTANCE, FIT_AZIMUTH, FIT_NUM_COLS) = range(11+1)

(AR_TIME_COL, AR_AZI_COL, AR_SNR_COL, AR_PHASEID_COL, AR_SITEID_COL, AR_NUM_COLS) = range(5+1)

(HEURISTIC_FIT_B, HEURISTIC_FIT_HEIGHT, HEURISTIC_FIT_PHASE_START_TIME, HEURISTIC_FIT_PHASE_LENGTH, HEURISTIC_FIT_PEAK_OFFSET, HEURISTIC_FIT_PEAK_HEIGHT, HEURISTIC_FIT_CODA_START_OFFSET, HEURISTIC_FIT_CODA_LENGTH, HEURISTIC_FIT_MAX_CODA_LENGTH, HEURISTIC_FIT_AVG_COST, HEURISTIC_FIT_NUM_COLS) = range(10+1)

# params for the envelope model
ARR_TIME_PARAM, PEAK_OFFSET_PARAM, PEAK_HEIGHT_PARAM, PEAK_DECAY_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(6+1)

def phaseid_to_name(phaseid):
    for (ids, n) in ((P_PHASEIDS, P_PHASES), (S_PHASEIDS, S_PHASES)):
        if phaseid in ids:
            i = ids.index(phaseid)
            return n[i]
    raise Exception("unrecognized phaseids %s" % (phaseid,))

def get_dir(dname):

    try:
        os.makedirs(dname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise
    return dname


def get_base_dir(siteid, runid, label=None):
    if label is not None:
        return get_dir(os.path.join("logs", "codas_%d_%s_%s" % (siteid, label, runid)))
    else:
        return get_dir(os.path.join("logs", "codas_%d_%s" % (siteid, runid)))

def accept_fit(fit, min_coda_length=40, max_avg_cost=avg_cost_bound):
# print fit[HEURISTIC_FIT_B], fit[HEURISTIC_FIT_CODA_LENGTH], fit[HEURISTIC_FIT_AVG_COST]
    return fit[HEURISTIC_FIT_B] > -0.15 and fit[HEURISTIC_FIT_B] <= 0 and fit[HEURISTIC_FIT_CODA_LENGTH] >= (min_coda_length-0.1) and fit[HEURISTIC_FIT_AVG_COST] <= max_avg_cost

def load_shape_data(cursor, chan, short_band, siteid, runid, acost_threshold=10, min_azi=0, max_azi=360):
    sql_query = "select lebo.evid, lebo.mb, lebo.lon, lebo.lat, lebo.depth, pid.id, fit.peak_delay, fit.coda_height, fit.coda_decay, fit.dist, fit.azi from leb_origin lebo, leb_assoc leba, leb_arrival l, sigvisa_coda_fits fit, static_siteid sid, static_phaseid pid where fit.arid=l.arid and l.arid=leba.arid and leba.orid=lebo.orid and leba.phase=pid.phase and fit.chan='%s' and fit.band='%s' and sid.id=%d and sid.sta=l.sta and fit.runid=%d and fit.acost<%f and fit.peak_delay between -10 and 20 and fit.coda_decay>-0.2 and fit.azi between %f and %f" % (chan, short_band, siteid, runid, acost_threshold, min_azi, max_azi)
    cursor.execute(sql_query)
    shape_data = np.array(cursor.fetchall())
    return shape_data


def gen_logenvelope(length, sampling_rate, peak_height, gamma, b):
#    print length, sampling_rate, peak_height, gamma, b

    t = np.linspace(1/sampling_rate, length, length*sampling_rate)
    f = (gamma*-1)*np.log(t) + (b * t)

    offset = peak_height - f[0]
    f = f + offset

    return f

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def smooth_segment(segment, bands=None, chans=None, window_len=300):

    if chans is None:
        chans = segment[bands[0]].keys()
    if bands is None:
        bands = segment.keys()

    smoothed_segment = dict()
    for chan in chans:
        smoothed_segment[chan] = dict()
        for band in bands:

            try:
                tr = segment[chan][band]
            except:
                print "couldn't access band %s chan %s, skipping..." % (band, chan)
                continue

            smoothed = Trace(smooth(tr.data, window_len=window_len, window="hamming") , header=tr.stats.copy())
            smoothed.stats.npts = len(smoothed.data)
            smoothed_segment[chan][band] = smoothed

    return smoothed_segment

def load_signal_slice(cursor, evid, siteid, load_noise = False, learn_noise=False, bands=None, chans=None):
    sql_query="SELECT l.time, l.arid, pid.id FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba, static_phaseid pid where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d and pid.phase=leba.phase order by l.time" % (evid, siteid)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    other_arrival_phaseids = other_arrivals[:, 2]
    other_arrival_arids = other_arrivals[:, 1]
    other_arrivals = other_arrivals[:, 0]
    sql_query="SELECT leba.phase, l.arid FROM leb_arrival l, static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (evid, siteid)
    cursor.execute(sql_query)
    other_arrival_phases = np.array(cursor.fetchall())
    other_arrival_phases = other_arrival_phases[:,0]

    first_p_arrival = None
    first_p_phaseid = 1
    for i, arrival in enumerate(other_arrivals):
        if int(other_arrival_phaseids[i]) in P_PHASEIDS:
            first_p_arrival = arrival
            first_p_phaseid = other_arrival_phaseids[i]
            break
    first_s_arrival = None
    first_s_phaseid = None
    for i, arrival in enumerate(other_arrivals):
        if int(other_arrival_phaseids[i]) in S_PHASEIDS:
            first_s_arrival = arrival
            first_s_phaseid = other_arrival_phaseids[i]
            break

    try:
        traces = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals)- 100, np.max(other_arrivals) + 350, stalist=[siteid,])
        arrival_segment = sigvisa_util.extract_timeslice_at_station(traces, np.min(other_arrivals)-100, np.max(other_arrivals) + 350, siteid)
        if arrival_segment is None:
            return None
        sigvisa_util.compute_narrowband_envelopes(arrival_segment)


        if chans is None:
            chans = arrival_segment[0].keys()
        if bands is None:
            bands = arrival_segment[0][chans[0]].keys()

        noise_segment = None
        if load_noise:
            noise_segment = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals)-150, np.min(other_arrivals)-50, stalist=[siteid,])
            sigvisa_util.compute_narrowband_envelopes(noise_segment)

        for chan in chans:
            for band in bands:
                a = arrival_segment[0][chan][band]
                a.data = a.data[a.stats.sampling_rate*70: - a.stats.sampling_rate*20]
                a.stats.starttime_unix += 70
                a.stats.npts = len(a.data)
                a.stats.p_time = first_p_arrival
                a.stats.s_time = first_s_arrival
                a.stats.p_phaseid = first_p_phaseid
                a.stats.s_phaseid = first_s_phaseid
                if load_noise:
                    noise = noise_segment[0][chan][band]
                    noise.data = noise.data[noise.stats.sampling_rate*5 : -noise.stats.sampling_rate*5]

                    print "learning for band %s chan %s" % (band, chan)
                    a.stats.noise_floor = np.mean(np.log(noise.data))
                    a.stats.smooth_noise_floor = a.stats.noise_floor
                    if learn_noise:
                        ar_learner = ARLearner(noise.data, noise.stats.sampling_rate)
                        print ar_learner.c
                        #arrival_segment[0][chan][band].stats.noise_model = ar_learner.cv_select()
                        params, std = ar_learner.yulewalker(17)
                        print params, "std", std
                        em = ErrorModel(0, std)
                        a.stats.noise_model = ARModel(params, em, c=ar_learner.c)

                        smoothed_noise_data = smooth(noise.data, window_len=300, window="hamming")
                        ar_learner = ARLearner(smoothed_noise_data, noise.stats.sampling_rate)
                        print ar_learner.c
                        params, std = ar_learner.yulewalker(17)
                        em = ErrorModel(0, std)
                        a.stats.smooth_noise_model = ARModel(params, em, c=ar_learner.c)

                        """
                        if band == "narrow_envelope_2.00_3.00" and chan=="BHZ":
                            print "noise", noise.stats
                            print "signal",a.stats
                            print "model",a.stats.noise_model.params, a.stats.noise_model.em.std

                            print "writing noise..."
                            f = open('noise.dat', 'w')
                            for d in noise.data[17:]:
                                f.write(str(d) + "\n")
                            print "writing signal..."
                            f = open('signal.dat', 'w')
                            for d in a.data[17:]:
                                f.write(str(d) + "\n")

                            expected_noise = [ (sum([params[k] * noise.data[t-k-1] for k in range(len(params))])) for t in range(len(params), len(noise.data))]
                            expected_sig = [ (sum([params[k] * a.data[t-k-1] for k in range(len(params))])) for t in range(len(params), len(a.data))]

                            f = open('noisepred.dat', 'w')
                            for d in expected_noise:
                                f.write(str(d) + "\n")

                            f = open('signalpred.dat', 'w')
                            for d in expected_sig:
                                f.write(str(d) + "\n")

                                f = open('envelope.dat', 'w')
                            for d in arrival_segment[0][chan]["broadband_envelope"].data:
                                f.write(str(d) + "\n")

                            f = open('raw_signal.dat', 'w')
                            for d in arrival_segment[0][chan]["broadband"].data:
                                f.write(str(d) + "\n")

                            f = open('noise_envelope.dat', 'w')
                            for d in noise_segment[0][chan]["broadband_envelope"].data:
                                f.write(str(d) + "\n")

                            f = open('raw_noise.dat', 'w')
                            for d in noise_segment[0][chan]["broadband"].data:
                                f.write(str(d) + "\n")"""



    except:
        raise

    print "finished loading signal slice..."

    return (arrival_segment, noise_segment, other_arrivals, other_arrival_phases, other_arrival_arids)


def safe_lookup(l, idx):
    if l is not None:
        return l[idx]
    else:
        return -1

def load_event(cursor, evid):
    sql_query="SELECT lon, lat, depth, time, mb, orid, evid FROM leb_origin where evid=%d" % (evid)
    cursor.execute(sql_query)
    return np.array(cursor.fetchone())

def pred_arrtime(cursor, r, netmodel, phaseid_col, phase_arr_time_col):
    cursor.execute("select time, depth from leb_origin where evid=%d" % (r[EVID_COL]))
    (t, d) = cursor.fetchone()

    phase = int(r[phaseid_col])
    if phase == -1:
        phase = 4 if phaseid_col == S_PHASEID_COL else 0

    pred_arr_time = t + netmodel.mean_travel_time(r[LON_COL], r[LAT_COL], d, int(r[SITEID_COL])-1, phase-1)

    sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (r[EVID_COL], r[SITEID_COL])
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    other_arrivals = other_arrivals[:, 0]
    start_time = np.min(other_arrivals) - 30

    return (pred_arr_time - start_time) - r[phase_arr_time_col]


def logenv_linf_cost(true_env, logenv):
    c = np.max (np.abs(true_env - logenv))
    return c

def logenv_l1_cost(true_env, logenv):
    n = len(true_env)
    n2 = len(logenv)
    if n != n2:
        if np.abs(n-n2) > 5:
            print "warning: comparing unequal-length traces (%d vs %d)" % (n, n2)
        n = np.min([n, n2])
    c = np.sum (np.abs(true_env[:n] - logenv[:n]))
    return c

def logenv_ar_cost(true_env, logenv):
    diff = true_env - logenv

    ar_n = 3
    ar_params = [0.1, 0.1, 0.8]
    ll = 0

    last_n = diff[0:ar_n]
    for x in diff:
        pred = np.sum(last_n * ar_params)
        ll = ll - (x-pred)**2

    return ll


def plot_heat(pp, f, n=20, center=None, width=None, lonbounds=None, latbounds=None, title=""):

    if lonbounds is not None and latbounds is not None:
        min_lon = lonbounds[0]
        max_lon = lonbounds[1]
        min_lat = latbounds[0]
        max_lat = latbounds[1]
    elif center is not None and width is not None:
        min_lon = center[0]-width/2.0
        max_lon = center[0]+width/2.0
        min_lat = center[1]-width/2.0
        max_lat = center[1]+width/2.0
    else:
        raise RuntimeError("Heat map requires either a bounding box, or a center and a width")

    bmap = draw_earth("",
                  #"NET-VISA posterior density, NEIC(white), LEB(yellow), "
                  #"SEL3(red), NET-VISA(blue)",
                  projection="cyl",
                  resolution="l",
                  llcrnrlon = min_lon, urcrnrlon = max_lon,
                  llcrnrlat = min_lat, urcrnrlat = max_lat,
                  nofillcontinents=True,
                      figsize=(8,8))

    lon_arr = np.linspace(min_lon, max_lon, n)
    lat_arr = np.linspace(min_lat, max_lat, n)


    max_val = np.float("-inf")
    max_lon = 0
    max_lat = 0
    out = np.zeros((len(lon_arr), len(lat_arr)))
    for loni, lon in enumerate(lon_arr):
        for lati, lat in enumerate(lat_arr):
            out[loni, lati] = f(lon, lat)

            if out[loni, lati] > max_val:
                max_lon = lon
                max_lat = lat
                max_val = out[loni, lati]

    minlevel = scipy.stats.scoreatpercentile(out.flatten(), 20)
    levels = np.linspace(minlevel, np.max(out), 10)

    draw_density(bmap, lon_arr, lat_arr, out, levels = levels, colorbar=True)

    plt.title(title)

    return bmap, (max_lon, max_lat)


def logsub_noise(log_height, log_noise):
    return np.log ( np.exp(log_height) - np.exp(log_noise) )

def subtract_traces(tr, to_subtract):
    l = np.min([len(tr.data), len(to_subtract.data)])

    newdata = tr.data[:l] - to_subtract.data[:l]
    newtrace = Trace(newdata, header=tr.stats.copy())
    newtrace.stats.npts = len(newtrace.data)
    return newtrace

def get_template(sigmodel, trace, phaseids, params, logscale=False, sample=False):
    srate = trace.stats['sampling_rate']
    st = trace.stats.starttime_unix
    et = st + trace.stats.npts/srate
    siteid = trace.stats.siteid
    c = sigvisa.canonical_channel_num(trace.stats.channel)
    b = sigvisa.canonical_band_num(trace.stats.band)
    if not sample:
        env = sigmodel.generate_trace(st, et, int(siteid), int(b), int(c), srate, phaseids, params)
    else:
        env = sigmodel.sample_trace(st, et, int(siteid), int(b), int(c), srate, phaseids, params)
    env.data = np.log(env.data) if logscale else env.data
    return env


