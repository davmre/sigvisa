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


import utils.nonparametric_regression as nr
from utils.plot_multi_station_params import *
from utils.predict_envelope_shape import read_shape_data, read_shape_params


S_PHASES = ['S', 'Sn']
P_PHASES = ['P']
LOVE_PHASES = ['LR']
phasenames = eval(sys.argv[2])

chan_name = "horiz_avg"
mag_threshold = 4.0
min_coda_length = 40

if sys.argv[2] == "P_PHASES":
    mag_threshold = 4
    chan_name = "BHZ"
    min_coda_length = 20

label = '_' + sys.argv[2]


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

def arrival_peak_offset(trace, window_start_time, window_end_time):
    srate = trace.stats.sampling_rate

    i = np.floor((window_start_time)*srate)
    j = np.floor(np.min([window_start_time+15, window_end_time])*srate)

    print window_start_time, window_end_time, i, j, srate, trace.data.shape

    pt = np.argmax(trace.data[i:j]) / srate
    return (pt +window_start_time, trace.data[(pt+window_start_time) * srate ])

def gen_logenvelope(length, sampling_rate, peak_height, gamma, b):
    t = np.linspace(1/sampling_rate, length, length*sampling_rate)
    f = (gamma*-1)*np.log(t) + (b * t)

    offset = peak_height - f[0]
    f = f + offset

    return f

def logenv_linf_cost(true_env, logenv):
    c = np.max (np.abs(true_env - logenv))
    return c

def logenv_l1_cost(true_env, logenv):
    c = np.sum (np.abs(true_env - logenv))
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

def fit_logenvelope(trace, peak_offset_time, peak_height, coda_length):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[peak_offset_time*srate : (peak_offset_time + coda_length)*srate]

    cost = lambda(height, b): logenv_l1_cost(true_env, gen_logenvelope(len(true_env)/srate, srate, height, 0, b))
    mcost = lambda(height, b): logenv_linf_cost(true_env, gen_logenvelope(len(true_env)/srate, srate, height, 0, b))

    bounds = ((peak_height-2, peak_height+1), (-.1, 0),)
    results = scipy.optimize.brute(cost, bounds, Ns=15, full_output=0)

    avg_cost = cost(results)/len(true_env)
    max_cost = mcost(results)

    #b = results[0]
    #results = (0., b)
    
    return results, avg_cost, max_cost


def unfair_para_predict(params_dict, arrival, band, distance):
    evid = int(arrival[5])
    siteid = int(arrival[10])
    phaseid = int(arrival[7])

    phase = "P" if phaseid else "S"
    s = "r" if distance < 1000 else "t"

    key = "%s %d %s %s" % (s, siteid, phase, band[19:])
    v = params_dict[key]
    print "returning [%s] = %f, %f" % (key, v[0], v[1])

    return v

def fair_nonpara_predict(arrival, lb, w):

    evid = int(arrival[5])

    if lb is None:
        return None

    x0 = [arrival[2], arrival[3]]
    X = np.array([ lb[i, [LON_COL, LAT_COL]] for i in range(lb.shape[0]) if lb[i, EVID_COL] != evid ] )
    y = np.array([ lb[i, B_COL] for i in range(lb.shape[0]) if lb[i, EVID_COL] != evid ] )

    if y.shape[0] == 0:
        return None

    ll_kernel = lambda ( ll1, ll2) : np.exp(-1 * utils.geog.dist_km(ll1, ll2)/ (w**2))
    y0 = nr.kernel_predict(x0, X, y, kernel=ll_kernel)
    print "returning", y0
    return y0

def plot_envelopes(pp, trace, peak_offset_time, peak_height, phase_start_time, noise_floor, coda_length, gamma, b, arrival, netmodel, band, distance, lb, shape_params, all_det_times = None, all_det_labels = None, title = None):
    srate = trace.stats['sampling_rate']

    siteid = int(arrival[10])
    phaseid = int(arrival[7])
    

    fake_stats = trace.stats.copy()
    fake_stats['starttime_unix'] += peak_offset_time
    fake_trace = Trace(gen_logenvelope(coda_length, srate, gamma, 0, b), fake_stats)
#    fake_trace = Trace(gen_logenvelope(coda_length, srate, peak_height, gamma, b), fake_stats)
    fake_trace.stats.npts = len(fake_trace.data)

#    pred_stats = trace.stats.copy()
#    pred_stats['starttime_unix'] = arrival[8] + netmodel.mean_travel_time(arrival[2], arrival[3], arrival[9], siteid-1, phaseid-1)
#    pred_para_b = unfair_para_predict(shape_params, arrival, band, distance)[0]
#    pred_nonpara_b = fair_nonpara_predict(arrival, lb, 30)

#    pred_trace_para = Trace(gen_logenvelope(coda_length, srate, gamma, 0, pred_para_b), pred_stats)
#    pred_trace_para.stats.npts = len(pred_trace_para.data)

#    if pred_nonpara_b is not None:
#        pred_trace_nonpara = Trace(gen_logenvelope(coda_length, srate, gamma, 0, pred_nonpara_b), pred_stats.copy())
#    else: 
#        pred_trace_nonpara = Trace(np.array(()), pred_stats.copy())
#    pred_trace_nonpara.stats.npts = len(pred_trace_nonpara.data)


    plot.plot_traces([trace, fake_trace], title=title, all_det_times=all_det_times, all_det_labels=all_det_labels, formats = ["k-", "r-"], linewidths = [5,5])
#    plot.plot_traces([trace, fake_trace, pred_trace_para, pred_trace_nonpara], title=title + " pb %f npb %f" % (pred_para_b, pred_nonpara_b), all_det_times=all_det_times, all_det_labels=all_det_labels, formats = ["k-", "r-", "b:", "g:"], linewidths = [5,5,1,1])
    maxtrc, mintrc = float(max(trace.data)), float(min(trace.data))
    plt.bar(left = trace.stats['starttime_unix'] + phase_start_time - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)
    plt.bar(left = trace.stats['starttime_unix'] + (peak_offset_time + coda_length) - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)

    xvals = [trace.stats.starttime_unix, trace.stats.starttime_unix + trace.stats.npts/srate]
    plt.plot(xvals, [noise_floor, noise_floor], "g-")

    pp.savefig()


def find_coda_length(trace, peak_offset_time, phase_end_time, noise_floor):
    srate = trace.stats.sampling_rate

#    last_window_avg = np.mean(trace.data[peak_offset_time*srate + 5*srate : peak_offset_time*srate + 10*srate])

#    num_windows = np.floor((phase_end_time - peak_offset_time)/5)
#    print "pot", peak_offset_time
#    print num_windows
#    print last_window_avg
#    for i in np.linspace((peak_offset_time+10)*srate, (peak_offset_time+ (num_windows+2)*5)*srate, num_windows):
#        window_avg = np.mean(trace.data[i : i + 5*srate])
#        if np.isnan(window_avg):
#            return i/srate - peak_offset_time

#        if (window_avg > last_window_avg +0.1):
#            print "returning i/srate", i/srate, window_avg, last_window_avg, num_windows, peak_offset_time*srate + 10*srate, peak_offset_time*srate + (num_windows)*5*srate, peak_offset_time, phase_end_time, i/srate - peak_offset_time
#            return i/srate - peak_offset_time
#        print "last_avg, avg", last_window_avg, window_avg
#        last_window_avg = window_avg

#    return phase_end_time - peak_offset_time



# choose noise floor as the largest of the last 5 five-second windows
#    num_back_windows = int(np.min([5, np.floor((phase_end_time - peak_offset_time)/5)]))
#    boundaries = np.floor(np.linspace((phase_end_time - 5*num_back_windows)*srate, phase_end_time*srate, num_back_windows+1))
#    print boundaries
#    noise_floor = np.max([ np.mean(trace.data[int(boundaries[n]):int(boundaries[n+1])]) for n in range(num_back_windows) ])

#    print "windows", num_back_windows, noise_floor

#    num_windows = np.floor((phase_end_time - peak_offset_time)/5)
#    for i in np.linspace((peak_offset_time + 10)*srate, (peak_offset_time + (num_windows+2)* 5)*srate, num_windows):
#        window_avg = np.mean(trace.data[i : i + 5*srate])
#        print window_avg, " vs ", noise_floor
#        if window_avg < noise_floor + 0.1:
#            return (i+5)/srate - peak_offset_time
#    return phase_end_time - peak_offset_time


# choose coda end time as the first 5s window when we fit a positive slope?

# or, do some sort of clustering of slopes - but this is only good if it means we tolerate one or two moments of positive slope in exchange for a much longer period of consistent negative slope...

    num_windows = np.floor((phase_end_time - peak_offset_time - 10 )/5) -1
    for i in np.linspace((peak_offset_time + 10)*srate, (phase_end_time - 10)*srate, num_windows):
        height = trace.data[i]
        cost = lambda(b): logenv_l1_cost(trace.data[i:i+10*srate], gen_logenvelope(10, srate, height, 0, b))
        bounds = ((-.01, 0.1),)
        results = scipy.optimize.brute(cost, bounds, Ns=10, full_output=0)
        b = results[0]

        if b > 0 or np.mean(trace.data[i:i+10*srate]) < noise_floor:
            return (i)/srate - peak_offset_time

    return phase_end_time - peak_offset_time

siteid = int(sys.argv[1])
cursor = db.connect().cursor()
sites = read_sites(cursor)
st  = 1237680000
et = st + 3600*24

site_up = read_uptime(cursor, st, et)
detections, arid2num = read_detections(cursor, st, et, arrival_table="leb_arrival", noarrays=True)
phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in phasenames]) + ")"

phasenames, phasetimedef = read_phases(cursor)
earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
netmodel = learn.load_netvisa("parameters", st, et, detections, site_up, sites, phasenames, phasetimedef)

max_azi_count = -1
max_azi = -1
max_azi_condition = ""
for azi in np.linspace(0, 330, 12):
    if azi == 330:
        azi_condition = "(l.azimuth between 0 and 30 or l.azimuth between 330 and 360)"
    else:
        azi_condition = "l.azimuth between %f and %f" % (azi, azi+60)
    sql_query="SELECT count(distinct(l.arid)) FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>%f and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.id=%d and %s" % (mag_threshold, phase_condition, siteid, azi_condition)
    cursor.execute(sql_query)
    azi_count = cursor.fetchall()
    if azi_count > max_azi_count:
        max_azi_count = azi_count
        max_azi = azi
        max_azi_condition = azi_condition

print "max azi is", max_azi, "with count", max_azi_count

sql_query="SELECT l.time, lebo.mb, lebo.lon, lebo.lat, l.azimuth, lebo.evid, l.snr, pid.id, lebo.time, lebo.depth, sid.id FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>%f and leba.arid=l.arid and l.snr > 1 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase order by l.sta limit 0, 100" % (mag_threshold, phase_condition, siteid)
cursor.execute(sql_query)
arrivals = np.array(cursor.fetchall())

print arrivals

bands = ['narrow_logenvelope_4.00_6.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_1.00_1.50', 'narrow_logenvelope_0.70_1.00']
#bands = ['narrow_logenvelope_0.10_0.20', 'narrow_logenvelope_0.70_1.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_4.00_6.00']
#bands = ['narrow_logenvelope_2.00_3.00']

bands_b = np.zeros((arrivals.shape[0], len(bands)))

#ldata = read_shape_data('logs/station_%d%s_decay.data' % (siteid, label))
ldata = dict()
print "read data with keys", ldata.keys()
shape_params = read_shape_params('parameters/CodaDecay.txt')

f = open('logs/station_%d%s_decay.data' % (siteid, label), 'w')
for (band_idx, band) in enumerate(bands):
    try:
        learned = []

        #lb = ldata[band]
        lb = np.array(())

        f.write(band + "\n")

        pp = PdfPages('logs/synthetic_%s_%d%s.pdf' % (band, siteid, label))
        pp2 = PdfPages('logs/synthetic_%s_%d%s_selected.pdf' % (band, siteid, label))

        for (arrival_idx, arrival) in enumerate(arrivals):
            start_time = arrival[0] - 5
            end_time = arrival[0] + 200


            sql_query="SELECT l.time FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (arrival[5], siteid)
            cursor.execute(sql_query)
            other_arrivals = np.array(cursor.fetchall())
            sql_query="SELECT l.iphase FROM sane_leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (arrival[5], siteid)    
            cursor.execute(sql_query)
            other_arrival_phases = np.array(cursor.fetchall())

            phase_length = 200
            if other_arrivals.shape[0] > 0:
                for a in other_arrivals:
                    if a[0] > arrival[0]:
                        phase_length = a[0] - arrival[0]
                        break
            #print other_arrivals
            #print "loading with end %f + max(%f, %f) = %f" % (np.max(other_arrivals), phase_length, 100, np.max(other_arrivals) + np.max([phase_length, 100]))

            traces = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals)-30, np.max(other_arrivals) + np.max([phase_length, 100]), stalist=[siteid,])
            arrival_segment = sigvisa_util.extract_timeslice_at_station(traces, np.min(other_arrivals)-30, np.max(other_arrivals) + np.max([phase_length, 100]), siteid)
            if arrival_segment is None:
                continue
            sigvisa_util.compute_narrowband_envelopes(arrival_segment)

            noise_segment = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals)-150, np.min(other_arrivals)-50, stalist=[siteid,])

            sigvisa_util.compute_narrowband_envelopes(noise_segment)
            noise_trace = noise_segment[0][chan_name][band]
            noise_floor = np.mean(noise_trace.data)
        
            distance = utils.geog.dist_km((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))
            det_azimuth = arrival[4]
            azimuth = utils.geog.azimuth((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))
            bazimuth = utils.geog.azimuth((sites[siteid-1][0], sites[siteid-1][1]), (arrival[2], arrival[3]))
        
            try:
                trace = arrival_segment[0][chan_name][band]
            except:
                print "couldn't load trace, skipping..."
                continue
            phase_start_time = arrival[0] - trace.stats.starttime_unix
            npts = trace.stats.npts
            srate = trace.stats.sampling_rate
            
            if npts < srate * 15:
                print "minimum segment length 15s, skipping segment with", npts/srate
                continue

            smoothed = Trace(smooth(trace.data, window_len=300, window="flat") , header=trace.stats.copy())
            smoothed.stats.npts = len(smoothed.data)
            npts = smoothed.stats.npts
            
            phase_length = np.min([phase_length, npts/srate - phase_start_time])

            try:
                (peak_offset_time, peak_height) = arrival_peak_offset( smoothed, phase_start_time, phase_start_time + phase_length )
            except:
                print traceback.format_exc()
                print "skipping segment because it does not contain the arrival we care about"
                continue

            if (npts/srate - peak_offset_time) < 10:
                print "skipping segment because peak is within 10s of end"
                continue
        
            try:
                coda_length = np.min([phase_length - (peak_offset_time - phase_start_time), find_coda_length(smoothed, peak_offset_time, peak_offset_time - (peak_offset_time - phase_start_time)  + phase_length, noise_floor  )])
            except:
                print "error finding coda length"
                print traceback.format_exc()
                print phase_length, peak_offset_time, phase_start_time
                continue

            (gamma, b), avg_cost, max_cost = fit_logenvelope(smoothed, peak_offset_time, peak_height, coda_length)

            print "fit params", gamma, b, "at station", siteid, ", event at distance", distance, "at rel avg cost", avg_cost/peak_height, "rel max cost", max_cost/peak_height


            try:
                plot_envelopes(pp, smoothed, peak_offset_time, peak_height, phase_start_time, noise_floor, coda_length, gamma, b, arrival, netmodel, band, distance, lb, shape_params, all_det_times = other_arrivals[:,0], all_det_labels = other_arrival_phases, title="%s evid %d siteid %d mb %f \n dist %f peak %f b %f \n acost %f mcost %f azi %f noise %f" % (band, arrival[5], siteid, arrival[1], distance, gamma, b, avg_cost/peak_height, max_cost/peak_height, azimuth, noise_floor))
            except:
                print "error plotting:"
                print traceback.format_exc()
                continue

            f.write("%d %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (arrival[5], siteid, arrival[7], distance, azimuth, det_azimuth, arrival[2], arrival[3], b, peak_height, gamma, arrival[1], arrival[6], coda_length, avg_cost/peak_height, max_cost/peak_height, max_azi, noise_floor))

            if b > -0.15 and b <= 0 and coda_length > min_coda_length and (avg_cost/peak_height < 0.06) and (max_cost/peak_height < 0.2) and np.mod(azimuth-max_azi, 360) < 60:
                bands_b[arrival_idx, band_idx] = b
                learned.append((distance, azimuth, gamma, b))
                print "learning: %f %f %f %f" % (b, coda_length, avg_cost/peak_height, max_cost/peak_height)
                print azimuth, max_azi
                try:
                    plot_envelopes(pp2, smoothed, peak_offset_time, peak_height, phase_start_time, noise_floor, coda_length, gamma, b, arrival, netmodel, band, distance, lb, shape_params, all_det_times = other_arrivals[:,0], all_det_labels = other_arrival_phases, title="%s evid %d siteid %d mb %f \n dist %f peak %f b %f \n acost %f mcost %f azi %f noise %f" % (band, arrival[5], siteid, arrival[1], distance, gamma, b, avg_cost/peak_height, max_cost/peak_height, azimuth, noise_floor))
                except:
                    print "error plotting:"
                    print traceback.format_exc()
                    continue



            else:
                print "not learning: %f %f %f %f" % (b, coda_length, avg_cost/peak_height, max_cost/peak_height)
                print azimuth, max_azi

        l = np.array(learned)
        print l

        try:

#            plt.figure()
#            plt.xlabel("distance (km)")
#            plt.ylabel("gamma")
#            plt.plot(l[:, 0], l[:, 2], 'ro')
#            pp.savefig()

            plt.figure()
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            plt.plot(l[:, 0], l[:, 3], 'ro')
            pp.savefig()

            plt.figure()
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            plt.plot(l[:, 0], l[:, 3], 'ro')
            pp2.savefig()

 #           plt.figure()
 #           plt.xlabel("azimuth (deg)")
 #           plt.ylabel("gamma")
 #           plt.plot(l[:, 1], l[:, 2], 'ro')
 #           pp.savefig()

            plt.figure()
            plt.xlabel("azimuth (deg)")
            plt.ylabel("b")
            plt.xlim([0, 360])
            plt.plot(l[:, 1], l[:, 3], 'ro')
            pp.savefig()

            plt.figure()
            plt.xlabel("azimuth (deg)")
            plt.ylabel("b")
            plt.xlim([0, 360])
            plt.plot(l[:, 1], l[:, 3], 'ro')
            pp2.savefig()


        except:
            print "error plotting learned params"
            print traceback.format_exc()
            continue

    finally:
        try:
            pp.close()          
            pp2.close()
        except:
            pass

f.close()


pp = PdfPages('logs/synthetic_bs_%d%s.pdf' % (siteid, label))

try:
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):

            plt.figure()
            plt.xlabel(bands[i])
            plt.ylabel(bands[j])
            plt.plot(bands_b[:, i], bands_b[:, j], 'ro')
            pp.savefig()


    for base in range(len(bands)):
        plt.figure()
        plt.title("Ratios against %s" % (bands[base]))
        success = 0
        for i in range(bands_b.shape[0]):
            try:
                ratios = [bands_b[i, j]/ bands_b[i, base] for j in range(len(bands))]
                print ratios
                plt.plot(range(len(ratios)), ratios, "b-")
                success = success+1
            except:
                print "exception", ratios
                print traceback.format_exc()
                continue
        if success  > 0:
            pp.savefig()

finally:
    pp.close()

print bands_b







