import os, errno, sys, time, traceback, hashlib
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

def load_event(cursor, evid):
    sql_query = "SELECT lon, lat, depth, time, mb, orid, evid from leb_origin where evid=%d" % (evid)
    cursor.execute(sql_query)
    return np.array(cursor.fetchone())

def load_event_arrivals(cursor, evid, siteid):
    sql_query = "select l.time from leb_arrival l, leb_origin lebo, leb_assoc leba, static_siteid sid where l.arid=leba.arid and leba.orid=lebo.orid and lebo.evid=%d and l.sta=sid.sta and sid.id=%d" % (evid, siteid)
    cursor.execute(sql_query)
    return np.array(cursor.fetchall())[:,0]


def filter_shape_data(fit_data, chan=None, short_band=None, siteid=None, runid=None, phaseids=None, evids=None, min_azi=0, max_azi=360, min_mb=0, max_mb=100, min_dist=0, max_dist=20000):

    new_data = []
    for row in fit_data:
        if chan is not None:
            if row[FIT_CHAN] != chan:
                continue
        if short_band is not None:
            b  = sigvisa.canonical_band_num(short_band)
            if int(row[FIT_BANDID]) != b:
                continue
        if siteid is not None:
            if int(row[FIT_SITEID]) != siteid:
                continue
        if runid is not None:
            if int(row[FIT_RUNID]) != runid:
                continue
        if phaseids is not None:
            if int(row[FIT_PHASEID]) not in phaseids:
                continue
        if evids is not None:
            if int(row[FIT_EVID]) not in evids:
                continue
        if row[FIT_AZIMUTH] > max_azi or row[FIT_AZIMUTH] < min_azi:
            continue
        if row[FIT_MB] > max_mb or row[FIT_MB] < min_mb:
            continue
        if row[FIT_DISTANCE] > max_dist or row[FIT_DISTANCE] < min_dist:
            continue

        new_data.append(row)
    return np.array(new_data)


def load_shape_data(cursor, chan=None, short_band=None, siteid=None, runids=None, phaseids=None, evids=None, exclude_evids=None, acost_threshold=20, min_azi=0, max_azi=360, min_mb=0, max_mb=100, min_dist=0, max_dist=20000):

    chan_cond = "and fit.chan='%s'" % (chan) if chan is not None else ""
    band_cond = "and fit.band='%s'" % (short_band) if short_band is not None else ""
    site_cond = "and sid.id=%d" % (siteid) if siteid is not None else ""
    run_cond = "and (" + " or ".join(["fit.runid = %d" % runid for runid in runids]) + ")" if runids is not None else ""
    phase_cond = "and (" + " or ".join(["pid.id = %d" % phaseid for phaseid in phaseids]) + ")" if phaseids is not None else ""
    evid_cond = "and (" + " or ".join(["lebo.evid = %d" % evid for evid in evids]) + ")" if evids is not None else ""
    evid_cond = "and (" + " or ".join(["lebo.evid != %d" % evid for evid in exclude_evids]) + ")" if exclude_evids is not None else ""

    sql_query = "select distinct lebo.evid, lebo.mb, lebo.lon, lebo.lat, lebo.depth, pid.id, fit.peak_delay, fit.coda_height, fit.coda_decay, sid.id, fit.dist, fit.azi, fit.band from leb_origin lebo, leb_assoc leba, leb_arrival l, sigvisa_coda_fits fit, static_siteid sid, static_phaseid pid where fit.arid=l.arid and l.arid=leba.arid and leba.orid=lebo.orid and leba.phase=pid.phase and sid.sta=l.sta %s %s %s %s %s %s and fit.acost<%f and fit.peak_delay between -10 and 20 and fit.coda_decay>-0.2 and fit.azi between %f and %f and lebo.mb between %f and %f and fit.dist between %f and %f" % (chan_cond, band_cond, site_cond, run_cond, phase_cond, evid_cond, acost_threshold, min_azi, max_azi, min_mb, max_mb, min_dist, max_dist)

    fname = "db_cache/%s.txt" % str(hashlib.md5(sql_query).hexdigest())
    try:

        shape_data = np.loadtxt(fname, dtype=float)
    except:
        cursor.execute(sql_query)
        shape_data = np.array(cursor.fetchall(), dtype=object)
        print shape_data.size
        shape_data[:, FIT_BANDID] = np.asarray([sigvisa.canonical_band_num(band) for band in shape_data[:, FIT_BANDID]])
        shape_data = np.array(shape_data, dtype=float)
        np.savetxt(fname, shape_data)

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

def load_signal_slice(cursor, evid, siteid, load_noise = False, learn_noise=False, bands=None, chans=None, earthmodel=None):
    sql_query="SELECT l.time, l.arid, pid.id FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba, static_phaseid pid where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d and pid.phase=leba.phase order by l.time" % (evid, siteid)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())

    if len(other_arrivals) > 0:
        other_arrival_phaseids = other_arrivals[:, 2]
        other_arrival_arids = other_arrivals[:, 1]
        other_arrivals = other_arrivals[:, 0]
        sql_query="SELECT leba.phase, l.arid FROM leb_arrival l, static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (evid, siteid)
        cursor.execute(sql_query)
        other_arrival_phases = np.array(cursor.fetchall())
        other_arrival_phases = other_arrival_phases[:,0]
    else:
        if earthmodel is None:
            raise Exception("trying to load signal slice for undetected event, but no earthmodel was passed!")
        event = load_event(cursor, evid)
        other_arrival_phaseids = [1,5]
        other_arrival_phases = ['P', 'S']
        other_arrivals, p = predict_event_arrivals(cursor, earthmodel, evid, siteid, other_arrival_phaseids)
        other_arrival_arids = None

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

        if len(traces) == 0:
            raise Exception("no traces found for siteid %d between %f and %f" % (siteid,  np.min(other_arrivals)- 100, np.max(other_arrivals) + 350))
            #import pdb
            #pdb.set_trace()

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

                    a.stats.noise_floor = np.mean(noise.data)
                    a.stats.smooth_noise_floor = a.stats.noise_floor
                    if learn_noise:
#                        print "learning for band %s chan %s" % (band, chan)
                        ar_learner = ARLearner(noise.data, noise.stats.sampling_rate)
#                        print ar_learner.c
                        #arrival_segment[0][chan][band].stats.noise_model = ar_learner.cv_select()
                        params, std = ar_learner.yulewalker(17)
#                        print params, "std", std
                        em = ErrorModel(0, std)
                        a.stats.noise_model = ARModel(params, em, c=ar_learner.c)

                        smoothed_noise_data = smooth(noise.data, window_len=300, window="hamming")
                        ar_learner = ARLearner(smoothed_noise_data, noise.stats.sampling_rate)
#                        print ar_learner.c
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




def plot_heat(pp, f, n=20, center=None, width=None, lonbounds=None, latbounds=None, title="", fname=None):

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
    out = np.zeros((len(lon_arr), len(lat_arr)))

    data_file = None
    if fname is not None:
        get_dir(os.path.dirname(fname))

        # if file exists, open it and read the saved values
        try:
            data_file = open(fname, 'r')
            meta_info = data_file.readline()
            fmin_lon, fmax_lon, fmin_lat, fmax_lat, fn = [float(x) for x in meta_info.split()]
            if not (almost_equal(fmin_lon, min_lon) and
                    almost_equal(fmax_lon, max_lon) and
                    almost_equal(fmin_lat, min_lat) and
                    almost_equal(fmax_lat, max_lat) and
                    int(fn) == n):
                print '%f %f %f %f %d\n' % (min_lon, max_lon, min_lat, max_lat, n)
                print '%f %f %f %f %d\n' % (fmin_lon, fmax_lon, fmin_lat, fmax_lat, fn)
                raise Exception("output file %s already exists, but the parameters don't match!")

            for l in data_file:
                v = l.split()
                loni = int(v[0])
                lati = int(v[1])
                fval = float(v[2])
                out[loni, lati] = fval
            data_file.close()

        # otherwise, start a new file
        except IOError:
            data_file = open(fname, 'w')
            data_file.write('%f %f %f %f %d\n' % (min_lon, max_lon, min_lat, max_lat, n))
            data_file.close()

        data_file = open(fname, 'a')
        print "saving heat map values to %s" % fname

    max_val = np.float("-inf")
    max_lon = 0
    max_lat = 0
    for loni, lon in enumerate(lon_arr):
        for lati, lat in enumerate(lat_arr):

            # if the function value at this location wasn't in the file, compute it and save to file
            if out[loni, lati] == 0:
                out[loni, lati] = f(lon, lat)

                if data_file is not None:
                    data_file.write('%d %d %f\n' % (loni, lati, out[loni, lati]))
                    data_file.flush()
            else:
                print "using value loaded from file: %f at (%.2f, %.2f)" % (out[loni, lati], lon, lat)

            if out[loni, lati] > max_val:
                max_lon = lon
                max_lat = lat
                max_val = out[loni, lati]

    minlevel = scipy.stats.scoreatpercentile(out.flatten(), 20)
    levels = np.linspace(minlevel, np.max(out), 10)

    draw_density(bmap, lon_arr, lat_arr, out, levels = levels, colorbar=True)

    plt.title(title)

    return bmap, (max_lon, max_lat), out, lon_arr, lat_arr

def logsub_noise(log_height, log_noise):
    return np.log ( np.exp(log_height) - np.exp(log_noise) )

def subtract_traces(tr, to_subtract):
    l = np.min([len(tr.data), len(to_subtract.data)])
    newdata = tr.data[:l] - to_subtract.data[:l]
    newtrace = Trace(newdata, header=tr.stats.copy())
    newtrace.stats.npts = len(newtrace.data)
    return newtrace

