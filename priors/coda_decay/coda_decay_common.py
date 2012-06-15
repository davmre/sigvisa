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

bands = ["narrow_envelope_2.00_3.00",]
chans = ["BHZ","BHE", "BHN"]

(EVID_COL, SITEID_COL, BANDID_COL, P_PHASEID_COL, S_PHASEID_COL, DISTANCE_COL, AZI_COL, LON_COL, LAT_COL, MB_COL, VERT_P_FIT_B, VERT_P_FIT_HEIGHT, VERT_P_FIT_PHASE_START_TIME, VERT_P_FIT_PHASE_LENGTH, VERT_P_FIT_PEAK_OFFSET, VERT_P_FIT_PEAK_HEIGHT, VERT_P_FIT_CODA_START_OFFSET, VERT_P_FIT_CODA_LENGTH, VERT_P_FIT_MAX_CODA_LENGTH, VERT_P_FIT_AVG_COST, HORIZ_P_FIT_B, HORIZ_P_FIT_HEIGHT, HORIZ_P_FIT_PHASE_START_TIME, HORIZ_P_FIT_PHASE_LENGTH, HORIZ_P_FIT_PEAK_OFFSET, HORIZ_P_FIT_PEAK_HEIGHT, HORIZ_P_FIT_CODA_START_OFFSET, HORIZ_P_FIT_CODA_LENGTH, HORIZ_P_FIT_MAX_CODA_LENGTH, HORIZ_P_FIT_AVG_COST, VERT_S_FIT_B, VERT_S_FIT_HEIGHT, VERT_S_FIT_PHASE_START_TIME, VERT_S_FIT_PHASE_LENGTH, VERT_S_FIT_PEAK_OFFSET, VERT_S_FIT_PEAK_HEIGHT, VERT_S_FIT_CODA_START_OFFSET, VERT_S_FIT_CODA_LENGTH, VERT_S_FIT_MAX_CODA_LENGTH, VERT_S_FIT_AVG_COST, HORIZ_S_FIT_B, HORIZ_S_FIT_HEIGHT, HORIZ_S_FIT_PHASE_START_TIME, HORIZ_S_FIT_PHASE_LENGTH, HORIZ_S_FIT_PEAK_OFFSET, HORIZ_S_FIT_PEAK_HEIGHT, HORIZ_S_FIT_CODA_START_OFFSET, HORIZ_S_FIT_CODA_LENGTH, HORIZ_S_FIT_MAX_CODA_LENGTH, HORIZ_S_FIT_AVG_COST, VERT_NOISE_FLOOR_COL, HORIZ_NOISE_FLOOR_COL, DEPTH_COL, EVTIME_COL, NUM_COLS) = range(54+1)

(EV_MB_COL, EV_LON_COL, EV_LAT_COL, EV_EVID_COL, EV_TIME_COL, EV_DEPTH_COL, EV_NUM_COLS) = range(6+1)

(AR_TIME_COL, AR_AZI_COL, AR_SNR_COL, AR_PHASEID_COL, AR_SITEID_COL, AR_NUM_COLS) = range(5+1)

(FIT_B, FIT_HEIGHT, FIT_PHASE_START_TIME, FIT_PHASE_LENGTH, FIT_PEAK_OFFSET, FIT_PEAK_HEIGHT, FIT_CODA_START_OFFSET, FIT_CODA_LENGTH, FIT_MAX_CODA_LENGTH, FIT_AVG_COST, FIT_NUM_COLS) = range(10+1)

# params for the envelope model
ARR_TIME_PARAM, PEAK_OFFSET_PARAM, PEAK_HEIGHT_PARAM, PEAK_DECAY_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(6+1)

def phaseid_to_name(phaseid):
    for (ids, n) in ((P_PHASEIDS, P_PHASES), (S_PHASEIDS, S_PHASES)):
        if phaseid in ids:
            i = ids.index(phaseid)
            return n[i]
    raise Exception("unrecognized phaseids %s" % (phaseid,))

def add_depth_time(cursor, r):
    print r.shape
    z =  np.zeros( (r.shape[0], 1) )
    print z.shape
    r = np.hstack((r,  z, z))
    print r.shape

    for (idx, row) in enumerate(r):
        cursor.execute("select time, depth from leb_origin where evid=%d" % (row[EVID_COL]))
        (t, d) = cursor.fetchone()
        r[idx, DEPTH_COL] = d
        r[idx, EVTIME_COL] = t
    return r


def accept_fit(fit, min_coda_length=40, max_avg_cost=avg_cost_bound):
#    print fit[FIT_B], fit[FIT_CODA_LENGTH], fit[FIT_AVG_COST]
    return fit[FIT_B] > -0.15 and fit[FIT_B] <= 0 and fit[FIT_CODA_LENGTH] >= (min_coda_length-0.1) and fit[FIT_AVG_COST] <= max_avg_cost

def clean_points(X, P=True, vert=True):

    if X is None:
        return np.array(())

    if len(X.shape) < 2:
        X = np.reshape(X, (1, X.shape[0]))

    n = X.shape[0]

#    f = lambda (a, ma) : True
#    if filter_azi:
#        f = lambda (a,ma) : np.mod(a-ma, 360) < 60

    new_X = None
    for row in X:
        fit = fit_from_row(row, P, vert)
        if accept_fit(fit, min_coda_length = (min_p_coda_length if P else min_s_coda_length)):
            if new_X is None:
                new_X = row
            else:
                new_X = np.vstack([new_X, row])

    return new_X


def read_shape_params(fname):
    f = open(fname, 'r')
    params = dict()

    for line in f:
        s = line[:-1].split()
        k = " ".join(s[0:4])
        params[k] = [float(s[4]), float(s[5])]

    return params

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

def read_shape_data(fname):
    f = open(fname, 'r')
    all_data = None

    bands = f.readline()[:-1].split()

    line_no = 2
    for line in f:
        new_row = np.array(  map( lambda x : float(x), (line[:-1]).split()))

        if all_data is None:
            all_data = new_row
        else:
            try:
                all_data = np.vstack([all_data, new_row])
            except:
                print "error reading line %d: %s" % (line_no, line)
                continue
            finally:
                line_no += 1
    return all_data, bands



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

def extract_band(all_data, idx):

    band_data = None

    for row in all_data:
#        print row
        if int(row[BANDID_COL]) == idx:
            if band_data is None:
                band_data = row
            else:
                band_data = np.vstack([band_data, row])

    print band_data.shape
    return band_data


def safe_lookup(l, idx):
    if l is not None:
        return l[idx]
    else:
        return -1

def fit_from_row(row, P, vert):
    idx = VERT_P_FIT_B
    if P and not vert:
        idx += FIT_NUM_COLS
    elif not P and vert:
        idx += FIT_NUM_COLS*2
    elif not P and not vert:
        idx += FIT_NUM_COLS*3
    return row[idx:idx+FIT_NUM_COLS]

def load_event(cursor, evid):
    sql_query="SELECT lebo.mb, lebo.lon, lebo.lat, lebo.evid, lebo.time, lebo.depth FROM leb_origin lebo where lebo.evid=%d" % (evid)
    cursor.execute(sql_query)
    return np.array(cursor.fetchone())

def row_to_ev(cursor, row):
    return load_event(cursor, row[EVID_COL])



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

def construct_output_generators(cursor, netmodel, P, vert):
    if P and vert:
        b_col = VERT_P_FIT_B
    elif P and not vert:
        b_col = HORIZ_P_FIT_B
    elif not P and vert:
        b_col = VERT_S_FIT_B
    elif not P and not vert:
        b_col = HORIZ_S_FIT_B
    gen_decay = lambda x :x[b_col]
    gen_onset = lambda x : pred_arrtime(cursor, x, netmodel, P_PHASEID_COL if P else S_PHASEID_COL, b_col + (VERT_P_FIT_PEAK_OFFSET - VERT_P_FIT_B))
#    gen_amp = lambda x : (  (x[b_col + (VERT_P_FIT_HEIGHT - VERT_P_FIT_B)] - x[b_col] * (x[b_col + (VERT_P_FIT_CODA_START_OFFSET - VERT_P_FIT_B)] - x[b_col + (VERT_P_FIT_PEAK_OFFSET - VERT_P_FIT_B)]))  - x[VERT_NOISE_FLOOR_COL if vert else HORIZ_NOISE_FLOOR_COL]  )/ x[MB_COL]

    gen_amp = lambda x : np.log (  np.exp( x[b_col + (VERT_P_FIT_PEAK_HEIGHT - VERT_P_FIT_B)] ) - np.exp ( x[VERT_NOISE_FLOOR_COL if vert else HORIZ_NOISE_FLOOR_COL])  ) -  x[MB_COL]

    return b_col, gen_decay, gen_onset, gen_amp



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

def get_template(sigmodel, trace, phaseids, params, logscale=False):
    srate = trace.stats['sampling_rate']
    st = trace.stats.starttime_unix
    et = st + trace.stats.npts/srate
    siteid = trace.stats.siteid
    env = sigmodel.generate_segment(st, et, siteid, srate, phaseids, params)
    env = env[trace.stats.channel][trace.stats.band]
    env.data = np.log(env.data) if logscale else env.data
    return env


