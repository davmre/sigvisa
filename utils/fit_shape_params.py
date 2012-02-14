import os, sys
import numpy as np, scipy
import matplotlib.pyplot as plt

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import priors.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util


S_PHASES = ['S']
P_PHASES = ['P']
LOVE_PHASES = ['LR']
phasenames = S_PHASES

def arrival_peak_offset(trace, window_start_time, window_end_time):
    srate = trace.stats.sampling_rate

    i = np.floor((window_start_time)*srate)
    j = np.floor((window_end_time)*srate)

    print window_start_time, window_end_time, i, j, srate, trace.data.shape

    pt = np.argmax(trace.data[i:j]) / srate
    return (pt +window_start_time, trace.data[(pt+window_start_time) * srate ])

def gen_logenvelope(length, sampling_rate, peak_height, gamma, b):
    t = np.linspace(1/sampling_rate, length, length*sampling_rate)
    f = (gamma*-1)*np.log(t) + (b * t)

    offset = peak_height - f[0]
    f = f + offset

    return f

def logenv_l1_cost(true_env, logenv):
    c = np.sum (np.abs(true_env - logenv))
    return c


def fit_logenvelope(trace, peak_offset_time, peak_height, coda_length):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[peak_offset_time*srate : (peak_offset_time + coda_length)*srate]

    cost = lambda(gamma, b): logenv_l1_cost(true_env, gen_logenvelope(len(true_env)/srate, srate, peak_height, gamma, b))

    bounds = ((0, 2), (-.03, 0))
    results = scipy.optimize.brute(cost, bounds, Ns=15, full_output=0)
    print "fit at cost", cost(results)
    
    return results

def plot_envelopes(pp, trace, peak_offset_time, peak_height, phase_start_time, coda_length, gamma, b, all_det_times = None, all_det_labels = None, title = None):
    srate = trace.stats['sampling_rate']
    fake_stats = trace.stats.copy()
    fake_stats['starttime_unix'] += peak_offset_time

    fake_trace = Trace(gen_logenvelope(coda_length, srate, peak_height, gamma, b), fake_stats)
    fake_trace.stats.npts = len(fake_trace.data)

    plot.plot_traces(trace, fake_trace, title=title, all_det_times=all_det_times, all_det_labels=all_det_labels)
    maxtrc, mintrc = float(max(trace.data)), float(min(trace.data))
    plt.bar(left = trace.stats['starttime_unix'] + phase_start_time - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)
    plt.bar(left = trace.stats['starttime_unix'] + (peak_offset_time + coda_length) - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)

    pp.savefig()

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

    num_back_windows = np.min([5, np.floor((phase_end_time - peak_offset_time)/5)])
    boundaries = np.linspace((phase_end_time - 5*num_back_windows)*srate, phase_end_time*srate, num_back_windows+1)
    noise_floor = np.max([ np.mean(trace.data[boundaries[n]:boundaries[n+1]]) for n in range(num_back_windows) ])

    print "windows", num_back_windows, noise_floor

    num_windows = np.floor((phase_end_time - peak_offset_time)/5)
    for i in np.linspace((peak_offset_time + 10)*srate, (peak_offset_time + (num_windows+2)* 5)*srate, num_windows):
        window_avg = np.mean(trace.data[i : i + 5*srate])
        print window_avg, " vs ", noise_floor
        if window_avg < noise_floor + 0.1:
            return i/srate - peak_offset_time
    return phase_end_time - peak_offset_time


siteid = int(sys.argv[1])

cursor = db.connect().cursor()

sites = read_sites(cursor)
print sites[siteid-1]


phase_condition = "(" + " or ".join(["iphase='%s'" % (pn) for pn in phasenames]) + ")"
sql_query="SELECT l.time, lebo.mb, lebo.lon, lebo.lat, lebo.evid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4.0 and leba.arid=l.arid and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d order by l.sta" % (phase_condition, siteid)
cursor.execute(sql_query)
arrivals = np.array(cursor.fetchall())




segments = []



bands = ['narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_0.70_1.00']


for band in bands:
    try:
        learned = []

        pp = PdfPages('logs/synthetic_%s_%d.pdf' % (band, siteid))

        for arrival in arrivals:
            start_time = arrival[0] - 5
            end_time = arrival[0] + 200

            print start_time

            sql_query="SELECT l.time FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (arrival[4], siteid)
            cursor.execute(sql_query)
            other_arrivals = np.array(cursor.fetchall())
            sql_query="SELECT l.iphase FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (arrival[4], siteid)    
            cursor.execute(sql_query)
            other_arrival_phases = np.array(cursor.fetchall())
            print other_arrivals
            print other_arrival_phases

            phase_length = 200
            if other_arrivals.shape[0] > 0:
                for a in other_arrivals:
                    print "a is ", a[0]
                    if a[0] > arrival[0]:
                        phase_length = a[0] - arrival[0]
                        print "pl is ", phase_length
                        break

            print arrival[0]
            print phase_length
            print [arrival[0]+phase_length,  np.max(other_arrivals)]

            arrival_segment = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals), np.max([arrival[0]+phase_length,  np.max(other_arrivals)]), stalist=[siteid,])
            sigvisa_util.compute_narrowband_envelopes(arrival_segment)
            segments = segments + arrival_segment
        
            distance = utils.geog.dist_km((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))
        
            try:
                trace = arrival_segment[0]['horiz_avg'][band]
            except:
                print "couldn't load trace, skipping..."
                continue
            phase_start_time = arrival[0] - trace.stats.starttime_unix
            npts = trace.stats.npts
            srate = trace.stats.sampling_rate
            
            if npts < srate * 15:
                print "minimum segment length 15s, skipping segment with", npts/srate
                continue

            smoothed = Trace(smooth(trace.data, window_len=60, window="flat") , header=trace.stats.copy())
            smoothed.stats.npts = len(smoothed.data)
            npts = smoothed.stats.npts

            try:
                (peak_offset_time, peak_height) = arrival_peak_offset( smoothed, phase_start_time, phase_start_time + phase_length )
            except:
                print "skipping segment because it does not contain the arrival we care about"
                continue


            if (npts/srate - peak_offset_time) < 10:
                print "skipping segment because peak is within 10s of end"
                continue
        
            coda_length = np.min([phase_length - (peak_offset_time - phase_start_time), find_coda_length(smoothed, peak_offset_time, peak_offset_time - (peak_offset_time - phase_start_time)  + phase_length  )])

            (gamma, b) = fit_logenvelope(smoothed, peak_offset_time, peak_height, coda_length)

            print "fit params", gamma, b, "at station", siteid, ", event at distance", distance

            plot_envelopes(pp, smoothed, peak_offset_time, peak_height, phase_start_time, coda_length, gamma, b, all_det_times = other_arrivals[:,0], all_det_labels = other_arrival_phases, title="%s evid %d siteid %d mb %f \n dist %f gamma %f b %f" % (band, arrival[4], siteid, arrival[1], distance, gamma, b))


            if distance < 7000 and b > -0.06 and b <= 0:
                learned.append((distance, gamma, b))

        print " learned params for band: ", band
        for l in learned:
            print l


        l = np.array(learned)
        
        print l


        plt.figure()
        plt.xlabel("distance (km)")
        plt.ylabel("gamma")
        plt.plot(l[:, 0], l[:, 1], 'ro')
        pp.savefig()

        plt.figure()
        plt.xlabel("distance (km)")
        plt.ylabel("b")
        plt.plot(l[:, 0], l[:, 2], 'ro')
        pp.savefig()

    finally:
        pp.close()          









