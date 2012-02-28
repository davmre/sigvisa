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


S_PHASES = ['S', 'Sn']
P_PHASES = ['P']
LOVE_PHASES = ['LR']
phasenames = eval(sys.argv[2])

chan_name = "horiz_avg"
mag_threshold = 4.0
if sys.argv[2] == "P_PHASES":
    mag_threshold = 4.5
    chan_name = "BHZ"

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
    j = np.floor((window_end_time)*srate)

#    print window_start_time, window_end_time, i, j, srate, trace.data.shape

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

def plot_envelopes(pp, trace, peak_offset_time, peak_height, phase_start_time, coda_length, gamma, b, all_det_times = None, all_det_labels = None, title = None):
    srate = trace.stats['sampling_rate']
    fake_stats = trace.stats.copy()
    fake_stats['starttime_unix'] += peak_offset_time

    fake_trace = Trace(gen_logenvelope(coda_length, srate, gamma, 0, b), fake_stats)
#    fake_trace = Trace(gen_logenvelope(coda_length, srate, peak_height, gamma, b), fake_stats)
    fake_trace.stats.npts = len(fake_trace.data)

    plot.plot_traces(trace, fake_trace, title=title, all_det_times=all_det_times, all_det_labels=all_det_labels)
    maxtrc, mintrc = float(max(trace.data)), float(min(trace.data))
    plt.bar(left = trace.stats['starttime_unix'] + phase_start_time - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)
    plt.bar(left = trace.stats['starttime_unix'] + (peak_offset_time + coda_length) - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)

    pp.savefig()


def find_coda_length(trace, peak_offset_time, phase_end_time):
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

        if b > -0.001:
            return (i)/srate - peak_offset_time
    return phase_end_time - peak_offset_time




siteid = int(sys.argv[1])
cursor = db.connect().cursor()
sites = read_sites(cursor)
phase_condition = "(" + " or ".join(["iphase='%s'" % (pn) for pn in phasenames]) + ")"

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

sql_query="SELECT l.time, lebo.mb, lebo.lon, lebo.lat, l.azimuth, lebo.evid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>%f and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and %s order by l.sta" % (mag_threshold, phase_condition, siteid, max_azi_condition)
cursor.execute(sql_query)
arrivals = np.array(cursor.fetchall())

print arrivals

segments = []

bands = ['narrow_logenvelope_4.00_6.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_1.00_1.50', 'narrow_logenvelope_0.70_1.00']
#bands = ['narrow_logenvelope_0.10_0.20', 'narrow_logenvelope_0.70_1.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_4.00_6.00']
#bands = ['narrow_logenvelope_2.00_3.00']

bands_b = np.zeros((arrivals.shape[0], len(bands)))

f = open('logs/station_%d%s_decay.data' % (siteid, label), 'w')
for (band_idx, band) in enumerate(bands):
    try:
        learned = []

        pp = PdfPages('logs/synthetic_%s_%d%s.pdf' % (band, siteid, label))

        for (arrival_idx, arrival) in enumerate(arrivals):
            start_time = arrival[0] - 5
            end_time = arrival[0] + 200


            sql_query="SELECT l.time FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (arrival[5], siteid)
            cursor.execute(sql_query)
            other_arrivals = np.array(cursor.fetchall())
            sql_query="SELECT l.iphase FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (arrival[5], siteid)    
            cursor.execute(sql_query)
            other_arrival_phases = np.array(cursor.fetchall())

            phase_length = 200
            if other_arrivals.shape[0] > 0:
                for a in other_arrivals:
                    if a[0] > arrival[0]:
                        phase_length = a[0] - arrival[0]
                        break

            arrival_segment = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals)-30, np.max([arrival[0]+phase_length,  np.max(other_arrivals)]), stalist=[siteid,])
            sigvisa_util.compute_narrowband_envelopes(arrival_segment)
            segments = segments + arrival_segment
        
            distance = utils.geog.dist_km((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))

            azimuth = arrival[4]
        
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
                print "skipping segment because it does not contain the arrival we care about"
                continue


            if (npts/srate - peak_offset_time) < 10:
                print "skipping segment because peak is within 10s of end"
                continue
        
            try:
                coda_length = np.min([phase_length - (peak_offset_time - phase_start_time), find_coda_length(smoothed, peak_offset_time, peak_offset_time - (peak_offset_time - phase_start_time)  + phase_length  )])
            except:
                print "error finding coda length"
                print traceback.format_exc()
                print phase_length, peak_offset_time, phase_start_time
                continue

            if coda_length < 15:
                print "coda length %f < 15s, skipping" % (coda_length,)
                continue

            (gamma, b), avg_cost, max_cost = fit_logenvelope(smoothed, peak_offset_time, peak_height, coda_length)

            print "fit params", gamma, b, "at station", siteid, ", event at distance", distance, "at rel avg cost", avg_cost/peak_height, "rel max cost", max_cost/peak_height

            if avg_cost/peak_height > 0.06 or max_cost/peak_height > 0.15:
                print "cost too high, skipping" 
                continue

            try:
                plot_envelopes(pp, smoothed, peak_offset_time, peak_height, phase_start_time, coda_length, gamma, b, all_det_times = other_arrivals[:,0], all_det_labels = other_arrival_phases, title="%s evid %d siteid %d mb %f \n dist %f peak %f b %f \n acost %f mcost %f azi %f" % (band, arrival[5], siteid, arrival[1], distance, gamma, b, avg_cost/peak_height, max_cost/peak_height, azimuth))
            except:
                print "error plotting:"
                print traceback.format_exc()
                continue

            if b > -0.15 and b <= 0:
                bands_b[arrival_idx, band_idx] = b
                learned.append((distance, azimuth, gamma, b))

        print " learned params for band: ", band
        for l in learned:
            print l


        l = np.array(learned)
        
        f.write(band + "\n")
        for idx in range(l.shape[0]):
            f.write("%f %f %f\n" % (l[idx, 0], l[idx, 1], l[idx, 3]))
        
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


        except:
            print "error plotting learned params"
            continue

    finally:
        pp.close()          

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







