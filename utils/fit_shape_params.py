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


def fit_logenvelope(trace, peak_time, peak_height):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[peak_time*srate :]
    length = len(trace.data) - srate*peak_time
    cost = lambda(gamma, b): logenv_l1_cost(true_env, gen_logenvelope(length, srate, peak_height, gamma, b))

    bounds = ((-2, 2), (-.05, 0.05))
    results = scipy.optimize.brute(cost, bounds, Ns=20, full_output=0, finish=None)
    print "fit at cost", cost(results)
    
    return results

def plot_envelopes(pp, trace, peak_time, peak_height, gamma, b, title = None):
    srate = trace.stats['sampling_rate']
    true_env = trace.data[peak_time*srate :]
    length = len(trace.data) - srate*peak_time
    fake_stats = trace.stats.copy()
    fake_stats['starttime_unix'] += peak_time
    fake_stats['npts'] = length
    fake_trace = Trace(gen_logenvelope(length, srate, peak_height, gamma, b), fake_stats)

    plot.plot_traces(trace, fake_trace, title=title)
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

siteid = int(sys.argv[1])

cursor = db.connect().cursor()

sites = read_sites(cursor)
print sites[siteid-1]

sql_query="SELECT l.time, lebo.mb, lebo.lon, lebo.lat, lebo.evid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and lebo.orid=leba.orid and (iphase='S' or iphase='Sn' or iphase='ScP') and sid.sta=l.sta and sid.statype='ss' and sid.id=%d order by l.sta" % (siteid)
cursor.execute(sql_query)
arrivals = np.array(cursor.fetchall())




segments = []



bands = ['narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_0.70_1.00']
for band in bands:

    learned = []

    pp = PdfPages('logs/synthetic_%s_%d.pdf' % (band, siteid))

    for arrival in arrivals:
        start_time = arrival[0] - 5
        end_time = arrival[0] + 200
        arrival_segment = sigvisa_util.load_and_process_traces(cursor, start_time, end_time, stalist=[siteid,])
        sigvisa_util.compute_narrowband_envelopes(arrival_segment)
        segments = segments + arrival_segment
        
        distance = utils.geog.dist_km((arrival[2], arrival[3]), (sites[siteid-1][0], sites[siteid-1][1]))
        
        trace = arrival_segment[0]['horiz_avg'][band]
        
        smoothed = Trace(smooth(trace.data, window_len=5, window="flat") , header=trace.stats.copy())
        smoothed.stats.npts = len(smoothed.data)
        (peak_time, peak_height) = arrival_peak_offset( smoothed )

        (gamma, b) = fit_logenvelope(trace, peak_time, peak_height)

    #gamma = 0.2
    #b = -0.01
        print "fit params", gamma, b, "at station", siteid, ", event at distance", distance
        plot_envelopes(pp, smoothed, peak_time, peak_height, gamma, b, title="%s evid %d siteid %d mb %f \n dist %f gamma %f b %f" % (band, arrival[4], siteid, arrival[1], distance, gamma, b))

        learned.append((distance, gamma, b))

    print " learned params for band: ", band
    for l in learned:
        print l

    pp.close()          









