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

def arrival_peak_offset(trace):
    window_end = trace.stats['sampling_rate'] * 45
    pt = np.argmax(trace.data[0:window_end]) / trace.stats['sampling_rate']
    return (pt, trace.data[pt * trace.stats['sampling_rate'] ])

def gen_logenvelope(length, sampling_rate, peak_height, gamma, b):
    t = np.linspace(1/sampling_rate, length/sampling_rate, length)
    f = (gamma*-1)*np.log(t) + (b * t)

    offset = peak_height - f[0]
    f = f + offset

    return f

def logenv_l1_cost(true_env, logenv):
    c = np.sum (np.abs(true_env - logenv))
    return c


def fit_logenvelope(trace, peak_time, peak_height, coda_length):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[peak_time*srate : coda_length]
    length = coda_length - srate*peak_time
    cost = lambda(gamma, b, peak): logenv_l1_cost(true_env, gen_logenvelope(length, srate, peak, gamma, b))

    bounds = ((0, 2), (-.03, 0), (0, 5))
    results = scipy.optimize.brute(cost, bounds, Ns=15, full_output=0)
    print "fit at cost", cost(results)
    
    return results

def plot_envelopes(pp, trace, peak_time, peak_height, coda_length, gamma, b, title = None):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[peak_time*srate :]
    length = len(trace.data) - srate*peak_time
    fake_stats = trace.stats.copy()
    fake_stats['starttime_unix'] += peak_time
    fake_stats['npts'] = length
    fake_trace = Trace(gen_logenvelope(length, srate, peak_height, gamma, b), fake_stats)

    plot.plot_traces(trace, fake_trace, title=title, all_det_times = [trace.stats['starttime_unix'] + coda_length/srate])
#    pp.savefig()
#    plot.plot_trace(fake_trace, title= "SYNTHETIC ENVELOPE (2-3Hz, site %d, gamma %f b %f)" % (siteid, gamma, b))
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


def find_coda_length(trace):
    srate = trace.stats.sampling_rate
    npts = trace.stats.npts
    noise_floor = np.mean(trace.data[npts - 5*srate:])

    for i in np.linspace(15*srate, npts - 5*srate, npts/(5*srate)-3):
        window_avg = np.mean(trace.data[i : i + 5*srate])
        if (window_avg < noise_floor) or np.abs(window_avg - noise_floor) < 0.1:
            return i
    return npts

siteid = int(sys.argv[1])

cursor = db.connect().cursor()

sites = read_sites(cursor)
print sites[siteid-1]

sql_query="SELECT l.time, lebo.mb, lebo.lon, lebo.lat, lebo.evid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4.0 and leba.arid=l.arid and lebo.orid=leba.orid and (iphase='S' or iphase='Sn') and sid.sta=l.sta and sid.statype='ss' and sid.id=%d order by l.sta" % (siteid)
cursor.execute(sql_query)
arrivals = np.array(cursor.fetchall())




segments = []



bands = ['narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_0.70_1.00']
for band in bands:

    learned = []

    pp = PdfPages('logs/synthetic_%s_%d_fit_amp.pdf' % (band, siteid))

    for arrival in arrivals:
        start_time = arrival[0] - 5
        end_time = arrival[0] + 200
        arrival_segment = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, stalist=[siteid,])
        sigvisa_util.compute_narrowband_envelopes(arrival_segment)
        segments = segments + arrival_segment
        
        distance = utils.geog.dist_km((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))
        
        trace = arrival_segment[0]['horiz_avg'][band]
        
        smoothed = Trace(smooth(trace.data, window_len=60, window="flat") , header=trace.stats.copy())
        smoothed.stats.npts = len(smoothed.data)
        (peak_time, peak_height) = arrival_peak_offset( smoothed )

        coda_length = find_coda_length(smoothed)

        (gamma, b, peak) = fit_logenvelope(smoothed, peak_time, peak_height, coda_length)

        print "fit params", gamma, b, "at station", siteid, ", event at distance", distance

        plot_envelopes(pp, smoothed, peak_time, peak, coda_length, gamma, b, title="%s evid %d siteid %d mb %f \n dist %f gamma %f b %f" % (band, arrival[4], siteid, arrival[1], distance, gamma, b))

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


    pp.close()          









