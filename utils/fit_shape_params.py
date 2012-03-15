import os, sys, time, traceback
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


(EVID_COL, SITEID_COL, P_PHASEID_COL, S_PHASEID_COL, DISTANCE_COL, AZI_COL, LON_COL, LAT_COL, MB_COL, VERT_P_B_COL, VERT_P_PEAK_COL, VERT_P_PEAKFIT_COL, VERT_P_ONSET_COL, VERT_P_CODA_LEN_COL, VERT_P_AVG_COST_COL, VERT_P_MAX_COST_COL, VERT_P_ACCEPT, VERT_S_B_COL, VERT_S_PEAK_COL, VERT_S_PEAKFIT_COL, VERT_S_ONSET_COL, VERT_S_CODA_LEN_COL, VERT_S_AVG_COST_COL, VERT_S_MAX_COST_COL, VERT_S_ACCEPT_COL, HORIZ_P_B_COL, HORIZ_P_PEAK_COL, HORIZ_P_PEAKFIT_COL, HORIZ_P_ONSET_COL, HORIZ_P_CODA_LEN_COL, HORIZ_P_AVG_COST_COL, HORIZ_P_MAX_COST_COL, HORIZ_P_ACCEPT_COL, HORIZ_S_B_COL, HORIZ_S_PEAK_COL, HORIZ_S_PEAKFIT_COL, HORIZ_S_ONSET_COL, HORIZ_S_CODA_LEN_COL, HORIZ_S_AVG_COST_COL, HORIZ_S_MAX_COST_COL, HORIZ_S_ACCEPT_COL, MIN_AZI_COL, VERT_NOISE_FLOOR_COL, HORIZ_NOISE_FLOOR_COL, NUM_COLS) = range(44+1)

P_PHASES = ['P', 'Pn']
P_PHASEIDS = [1,2]

S_PHASES = ['S', 'Sn']
S_PHASEIDS = [4, 5]

min_p_coda_length = 30
min_s_coda_length = 45

max_cost_bound = 0.15

full_coda = False
full_label = "notfull"

if len(sys.argv) == 3 and sys.argv[2] == "full":
    full_coda = True
    full_label = "full"
    max_cost_bound = 0.25

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

    # print window_start_time, window_end_time, i, j, srate, trace.data.shape

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


def fit_phase_coda(phase_arrival, smoothed, other_arrivals, other_arrival_phases, noise_floor):
    npts = smoothed.stats.npts
    srate = smoothed.stats.sampling_rate
    stime = smoothed.stats.starttime_unix

    phase_length = 200
    if other_arrivals.shape[0] > 0:
        for (a, pa) in zip(other_arrivals, other_arrival_phases):
            if a > phase_arrival[AR_TIME_COL] and pa != "LR":
                phase_length = np.min([a - phase_arrival[AR_TIME_COL], phase_length])
                
    # compute phase arrival times relative to the trace start time
    phase_start_time = phase_arrival[AR_TIME_COL] - stime
    phase_length = np.min([phase_length, npts/srate - phase_start_time])
    
    try:
        (peak_offset_time, peak_height) = arrival_peak_offset(smoothed, phase_start_time, phase_start_time + phase_length )
    except:
        print "segment does not contain the arrival we care about"
        return None

    if (npts/srate - peak_offset_time) < 10:
        print "skipping segment because peak is within 10s of end"
        return None
        
    try:
        coda_length = find_coda_length(smoothed, peak_offset_time, peak_offset_time - (peak_offset_time - phase_start_time)  + phase_length, noise_floor)
        coda_length = np.min([coda_length, phase_length - (peak_offset_time - phase_start_time)])
    except:
        print "error finding coda length"
        print traceback.format_exc()
        print phase_length, peak_offset_time, phase_start_time
        return None

    (fit_height, b), avg_cost, max_cost = fit_logenvelope(smoothed, peak_offset_time, peak_height, coda_length)
    
    return (b, fit_height, phase_start_time, phase_length, peak_offset_time, peak_height, coda_length, avg_cost/peak_height, max_cost/peak_height)

(FIT_B, FIT_HEIGHT, FIT_PHASE_START_TIME, FIT_PHASE_LENGTH, FIT_PEAK_OFFSET, FIT_PEAK_HEIGHT, FIT_CODA_LENGTH, FIT_AVG_COST, FIT_MAX_COST) = range(9)    

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


def plot_channels(pp, vert_trace, vert_noise_floor, vert_fits, vert_formats, horiz_trace, horiz_noise_floor, horiz_fits, horiz_formats, arrival, event, netmodel, band, distance, lb, shape_params, all_det_times = None, all_det_labels = None, title = None):
    plt.figure(figsize = (8, 8))

    bhz_axes = plt.subplot(2, 1, 1)

    if title is not None:
        plt.title(title, fontsize=12)

    plot_envelopes(bhz_axes, vert_trace, vert_noise_floor, vert_fits, vert_formats, arrival, event, netmodel, band, distance, lb, shape_params, all_det_times, all_det_labels)
    horiz_axes = plt.subplot(2, 1, 2, sharex=bhz_axes, sharey = bhz_axes)
    plot_envelopes(horiz_axes, horiz_trace, horiz_noise_floor, horiz_fits, horiz_formats, arrival, event, netmodel, band, distance, lb, shape_params, all_det_times, all_det_labels)
        
    pp.savefig()

def plot_envelopes(axes, trace, noise_floor, fits, formats, arrival, event, netmodel, band, distance, lb, shape_params, all_det_times = None, all_det_labels = None):
    srate = trace.stats['sampling_rate']

    siteid = int(arrival[AR_SITEID_COL])
    phaseid = int(arrival[AR_PHASEID_COL])
 
    traces = [trace,]
    formats = ["k-",] + formats
    linewidths = [5,]

    for fit in fits:
        if fit is not None:
            stats = trace.stats.copy()
            stats['starttime_unix'] += fit[FIT_PEAK_OFFSET]
            fit_trace = Trace(gen_logenvelope(fit[FIT_CODA_LENGTH], srate, fit[FIT_HEIGHT], 0, fit[FIT_B]), stats)
            fit_trace.stats.npts = len(fit_trace.data)
            traces.append(fit_trace)
#            formats.append("r-")
            linewidths.append(5)

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


    plot.plot_traces_subplot(axes, traces, all_det_times=all_det_times, all_det_labels=all_det_labels, formats = formats, linewidths = linewidths)
#    plot.plot_traces([trace, fake_trace, pred_trace_para, pred_trace_nonpara], title=title + " pb %f npb %f" % (pred_para_b, pred_nonpara_b), all_det_times=all_det_times, all_det_labels=all_det_labels, formats = ["k-", "r-", "b:", "g:"], linewidths = [5,5,1,1])
    maxtrc, mintrc = float(max(trace.data)), float(min(trace.data))
#    plt.bar(left = trace.stats['starttime_unix'] + phase_start_time - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)
#    plt.bar(left = trace.stats['starttime_unix'] + (peak_offset_time + coda_length) - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)

    xvals = [trace.stats.starttime_unix, trace.stats.starttime_unix + trace.stats.npts/srate]
    axes.plot(xvals, [noise_floor, noise_floor], "g-")




def find_coda_length(trace, peak_offset_time, phase_end_time, noise_floor):
    srate = trace.stats.sampling_rate

# choose coda end time as the first 5s window when we fit a positive slope

    num_windows = np.floor((phase_end_time - peak_offset_time - 10 )/5) -1
    first_strike = False

    for i in np.linspace((peak_offset_time + 10)*srate, (phase_end_time - 10)*srate, num_windows):
        height = trace.data[i]
        cost = lambda(b): logenv_l1_cost(trace.data[i:i+10*srate], gen_logenvelope(10, srate, height, 0, b))
        bounds = ((-.1, 0.1),)
        results = scipy.optimize.brute(cost, bounds, Ns=10, full_output=0)
        b = results[0]

        if not full_coda:

            if (b > 0 and first_strike) or np.mean(trace.data[i:i+10*srate]) < noise_floor(i/srate):
                return (i)/srate - peak_offset_time

            if (b > 0):
                first_strike = True

        else:
            if np.mean(trace.data[i:i+10*srate]) < noise_floor(i/srate) + 0.5:
                return (i)/srate - peak_offset_time


    return phase_end_time - peak_offset_time

def safe_lookup(l, idx):
    if l is not None:
        return l[idx]
    else:
        return -1

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


phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in phasenames]) + ")"
max_azi_count = -1
max_azi = -1
max_azi_condition = ""
for azi in np.linspace(0, 330, 12):
    if azi == 330:
        azi_condition = "(l.azimuth between 0 and 30 or l.azimuth between 330 and 360)"
    else:
        azi_condition = "l.azimuth between %f and %f" % (azi, azi+60)
    sql_query="SELECT count(distinct(l.arid)) FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.id=%d and %s" % (phase_condition, siteid, azi_condition)
    cursor.execute(sql_query)
    azi_count = cursor.fetchall()
    if azi_count > max_azi_count:
        max_azi_count = azi_count
        max_azi = azi
        max_azi_condition = azi_condition
print "max azi is", max_azi, "with count", max_azi_count


# want to select all events, with certain properties, which have a P or S phase detected at this station
phase_condition = "(" + " or ".join(["leba.phase='%s'" % (pn) for pn in S_PHASES + P_PHASES]) + ")"
print phase_condition
(EV_MB_COL, EV_LON_COL, EV_LAT_COL, EV_EVID_COL, EV_TIME_COL, EV_DEPTH_COL, EV_NUM_COLS) = range(6+1)
sql_query="SELECT distinct lebo.mb, lebo.lon, lebo.lat, lebo.evid, lebo.time, lebo.depth FROM leb_arrival l , static_siteid sid, static_phaseid pid, leb_origin lebo, leb_assoc leba where l.time between 1238889600 and 1245456000 and lebo.mb>4 and leba.arid=l.arid and l.snr > 2 and lebo.orid=leba.orid and %s and sid.sta=l.sta and sid.statype='ss' and sid.id=%d and pid.phase=leba.phase order by l.sta limit 0,10" % (phase_condition, siteid)
cursor.execute(sql_query)
events = np.array(cursor.fetchall())


print events.shape



#arrivals = np.array(cursor.fetchall())
#print arrivals

bands = ['narrow_logenvelope_4.00_6.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_1.00_1.50', 'narrow_logenvelope_0.70_1.00']
#bands = ['narrow_logenvelope_0.10_0.20', 'narrow_logenvelope_0.70_1.00', 'narrow_logenvelope_2.00_3.00', 'narrow_logenvelope_4.00_6.00']
#bands = ['narrow_logenvelope_2.00_3.00']

#ldata = read_shape_data('logs/station_%d%s_decay.data' % (siteid, label))
ldata = dict()
print "read data with keys", ldata.keys()
shape_params = read_shape_params('parameters/CodaDecay.txt')



f = open('logs/station_%d_%s_decay.data' % (siteid,full_label), 'w')
lb = np.array(())

short_bands = [b[19:] for b in bands]
pps = [PdfPages('logs/codas_%d_%s_%s.pdf' % (siteid, sb, full_label)) for sb in short_bands]
learned_p = [[] for b in bands]
learned_s = [[] for b in bands]
learned_sp = [[] for b in bands]

(AR_TIME_COL, AR_AZI_COL, AR_SNR_COL, AR_PHASEID_COL, AR_SITEID_COL, AR_NUM_COLS) = range(5+1)

for event in events:

    distance = utils.geog.dist_km((event[EV_LON_COL], event[EV_LAT_COL]), (sites[siteid-1][0], sites[siteid-1][1]))
    azimuth = utils.geog.azimuth((event[EV_LON_COL], event[EV_LAT_COL]), (sites[siteid-1][0], sites[siteid-1][1]))

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

    if first_p_arrival is not None and first_s_arrival is not None and first_p_arrival[AR_TIME_COL] > first_s_arrival[AR_TIME_COL]:
        print "skipping evid %d because S comes before P..." % (event[EV_EVID_COL])
        continue 

    sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (event[EV_EVID_COL], siteid)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    other_arrivals = other_arrivals[:, 0]
    sql_query="SELECT leba.phase, l.arid FROM leb_arrival l, static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (event[EV_EVID_COL], siteid)
    cursor.execute(sql_query)
    other_arrival_phases = np.array(cursor.fetchall())
    other_arrival_phases = other_arrival_phases[:,0]

    try:
        traces = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals)-30, np.max(other_arrivals) + 150, stalist=[siteid,])
        arrival_segment = sigvisa_util.extract_timeslice_at_station(traces, np.min(other_arrivals)-30, np.max(other_arrivals) + 150, siteid)
        if arrival_segment is None:
            continue
        sigvisa_util.compute_narrowband_envelopes(arrival_segment)

        # test to make sure we have the necessary channels
        tr1 = arrival_segment[0]["BHZ"][bands[0]]
        tr2 = arrival_segment[0]["horiz_avg"][bands[0]]

        noise_segment = sigvisa_util.load_and_process_traces(cursor, np.min(other_arrivals)-150, np.min(other_arrivals)-50, stalist=[siteid,])
        sigvisa_util.compute_narrowband_envelopes(noise_segment)

        tr1 = None
        tr2 = None
        del traces

    except:
        print traceback.format_exc()        
        continue

    for (band_idx, band) in enumerate(bands):
        short_band = short_bands[band_idx]
        pp = pps[band_idx]

        vert_noise_floor = np.mean(noise_segment[0]["BHZ"][band])
        horiz_noise_floor = np.mean(noise_segment[0]["horiz_avg"][band])
        
            # load the vertical and horizontal channels (intended for
            # analyzing P and S arrivals respectively), and establish
            # that they have the same start time, length, and sampling
            # rate.
        try:
            vert_trace = arrival_segment[0]["BHZ"][band]
            horiz_trace = arrival_segment[0]["horiz_avg"][band]
        except:
            print "couldn't load trace, skipping..."
            continue
        if np.abs(vert_trace.stats.starttime_unix - horiz_trace.stats.starttime_unix) > 0.01:
            raise Exception("vertical and horizontal channels have different start times!")
        if vert_trace.stats.npts != horiz_trace.stats.npts:
            raise Exception("vertical and horizontal channels have different lengths!")
        if vert_trace.stats.sampling_rate != horiz_trace.stats.sampling_rate:
            raise Exception("vertical and horizontal channels have different sampling rates!")

        # reject segments too short to do an accurate coda fit
        npts = vert_trace.stats.npts
        srate = vert_trace.stats.sampling_rate
        if npts < srate * 15:
            print "minimum segment length 15s, skipping segment with", npts/srate
            continue

        # compute smoothed traces
        vert_smoothed = Trace(smooth(vert_trace.data, window_len=300, window="hamming") , header=vert_trace.stats.copy())
        vert_smoothed.stats.npts = len(vert_smoothed.data)
        horiz_smoothed = Trace(smooth(horiz_trace.data, window_len=300, window="hamming") , header=horiz_trace.stats.copy())
        horiz_smoothed.stats.npts = len(horiz_smoothed.data)
        

        # DO THE FITTING
        vnf = lambda t : vert_noise_floor
        hnf = lambda t : horiz_noise_floor
        fit_p_vert = None
        fit_p_horiz = None
        fit_s_vert = None
        fit_s_horiz = None
        accept_p_vert = False
        accept_p_horiz = False
        accept_s_vert = False
        accept_s_horiz = False
        if first_p_arrival is not None:
            fit_p_vert = fit_phase_coda(first_p_arrival, vert_smoothed, other_arrivals, other_arrival_phases, vnf)
            fit_p_horiz = fit_phase_coda(first_p_arrival, horiz_smoothed, other_arrivals, other_arrival_phases, hnf)
            if fit_p_vert[FIT_B] > -0.15 and fit_p_vert[FIT_B] <= 0 and fit_p_vert[FIT_CODA_LENGTH] > min_p_coda_length and (fit_p_vert[FIT_AVG_COST] < 0.07) and (fit_p_vert[FIT_MAX_COST] < max_cost_bound):
                accept_p_vert = True
            if fit_p_horiz[FIT_B] > -0.15 and fit_p_horiz[FIT_B] <= 0 and fit_p_horiz[FIT_CODA_LENGTH] > min_p_coda_length and (fit_p_horiz[FIT_AVG_COST] < 0.07) and (fit_p_horiz[FIT_MAX_COST] < max_cost_bound):
                accept_p_horiz = True

# and np.mod(azimuth-max_azi, 360) < 60:


        if first_s_arrival is not None:

            # if we got a good fit to the P coda, use the continuing P coda as a secondary noise floor for the S coda
            if accept_p_vert:
                vnf = lambda t : max(vert_noise_floor, fit_p_vert[FIT_HEIGHT] + fit_p_vert[FIT_B]*(t - fit_p_vert[FIT_PEAK_OFFSET]))
            if accept_p_horiz:
                hnf = lambda t : max(horiz_noise_floor, fit_p_horiz[FIT_HEIGHT] + fit_p_horiz[FIT_B]*(t - fit_p_horiz[FIT_PEAK_OFFSET]))

            fit_s_vert = fit_phase_coda(first_s_arrival, vert_smoothed, other_arrivals, other_arrival_phases, vnf)
            fit_s_horiz = fit_phase_coda(first_s_arrival, horiz_smoothed, other_arrivals, other_arrival_phases, hnf)
            if fit_s_vert[FIT_B] > -0.15 and fit_s_vert[FIT_B] <= 0 and fit_s_vert[FIT_CODA_LENGTH] > min_s_coda_length and (fit_s_vert[FIT_AVG_COST] < 0.07) and (fit_s_vert[FIT_MAX_COST] < max_cost_bound):
                accept_s_vert = True
            if fit_s_horiz[FIT_B] > -0.15 and fit_s_horiz[FIT_B] <= 0 and fit_s_horiz[FIT_CODA_LENGTH] > min_s_coda_length and (fit_s_horiz[FIT_AVG_COST] < 0.07) and (fit_s_horiz[FIT_MAX_COST] < max_cost_bound):
                accept_s_horiz = True

#        print first_p_arrival
#        print first_s_arrival
#        print "p vert" , fit_p_vert
#        print "s horiz", fit_s_horiz

        if accept_p_vert:
            learned_p[band_idx].append((distance, azimuth, fit_p_vert[FIT_B]))
        if accept_s_horiz:
            learned_s[band_idx].append((distance, azimuth, fit_s_horiz[FIT_B]))

        if accept_p_vert and accept_s_horiz:
            learned_sp[band_idx].append((fit_p_horiz[FIT_B], fit_s_horiz[FIT_B]))

        gen_title = lambda event, arrival, p_fit, s_fit: "%s evid %d siteid %d mb %f \n dist %f azi %f \n p_b %f p_acost %f p_mcost %f p_len %f \n s_b %f s_acost %f s_mcost %f s_len %f " % (band, event[EV_EVID_COL], siteid, event[EV_MB_COL], distance, azimuth, safe_lookup(p_fit, FIT_B), safe_lookup(p_fit, FIT_AVG_COST), safe_lookup(p_fit, FIT_MAX_COST), p_fit[FIT_CODA_LENGTH], safe_lookup(s_fit, FIT_B), safe_lookup(s_fit, FIT_AVG_COST), safe_lookup(s_fit, FIT_MAX_COST), s_fit[FIT_CODA_LENGTH]) 
        try:
            plot_channels(pp, vert_smoothed, vert_noise_floor, [fit_p_vert, fit_s_vert], ["g-" if accept_p_vert else "r-", "g-" if accept_s_vert else "r-"], horiz_smoothed, horiz_noise_floor, [fit_p_horiz, fit_s_horiz], ["g-" if accept_p_horiz else "r-", "g-" if accept_s_horiz else "r-"], arrival, event, netmodel, band, distance, lb, shape_params, all_det_times = other_arrivals, all_det_labels = other_arrival_phases, title = gen_title(event, arrival, fit_p_vert, fit_s_horiz))
        except:
            print "error plotting:"
            print traceback.format_exc()

        f.write("%d %d %s %d %d %f %f %f %f %f " % (event[EV_EVID_COL], siteid, short_band, first_p_arrival[AR_PHASEID_COL] if first_p_arrival is not None else -1, first_s_arrival[AR_PHASEID_COL] if first_s_arrival is not None else -1, distance, azimuth, event[EV_LON_COL], event[EV_LAT_COL], event[EV_MB_COL]))
        write_fit = lambda f, fit, accept: f.write("%f %f %f %f %f %f %f %d" % (fit[FIT_B], fit[FIT_PEAK_HEIGHT], fit[FIT_PEAK_OFFSET] - fit[FIT_PHASE_START_TIME], fit[FIT_HEIGHT], fit[FIT_CODA_LENGTH], fit[FIT_AVG_COST], fit[FIT_MAX_COST], 1 if accept else 0))
        if first_p_arrival is not None:
            write_fit(f, fit_p_vert, accept_p_vert)
            write_fit(f, fit_p_horiz, accept_p_horiz)
        else:
            f.write("-1 -1 -1 -1 -1 -1 -1 ")
            f.write("-1 -1 -1 -1 -1 -1 -1 ")
        if first_s_arrival is not None:
            write_fit(f, fit_s_vert, accept_s_vert)
            write_fit(f, fit_s_horiz, accept_s_horiz)
        else:
            f.write("-1 -1 -1 -1 -1 -1 -1 ")
            f.write("-1 -1 -1 -1 -1 -1 -1 ")

        f.write("%f %f %f\n" % (max_azi, vert_noise_floor, horiz_noise_floor))

    del arrival_segment
    del noise_segment
    del other_arrivals
    del other_arrival_phases
    
for (band_idx, band) in enumerate(bands):
    try:

        lp = np.array(learned_p[band_idx])
        ls = np.array(learned_s[band_idx])
        lsp = np.array(learned_sp[band_idx])

        pp = pps[band_idx]

        if (len(lp.shape) == 2):
            plt.figure()
            plt.title("P codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            plt.plot(lp[:, 0], lp[:, 2], 'ro')
            pp.savefig()

        if (len(ls.shape) == 2):
            plt.figure()
            plt.title("S codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            plt.plot(ls[:, 0], ls[:, 2], 'ro')
            pp.savefig()

        if (len(lp.shape) == 2):
            plt.figure()
            plt.title("P codas / azimuth")
            plt.xlabel("azimuth (deg)")
            plt.ylabel("b")
            plt.xlim([0, 360])
            plt.plot(lp[:, 1], lp[:, 2], 'ro')
            pp.savefig()

        if (len(ls.shape) == 2):
            plt.figure()
            plt.title("S codas / azimuth")
            plt.xlabel("azimuth (deg)")
            plt.ylabel("b")
            plt.xlim([0, 360])
            plt.plot(ls[:, 1], ls[:, 2], 'ro')
            pp.savefig()

        if (len(lsp.shape) == 2):
            plt.figure()
            plt.title("P vs S")
            plt.xlabel("b (P coda)")
            plt.ylabel("b (S coda)")
            plt.plot(lsp[:, 0], lsp[:, 1], 'ro')
            pp.savefig()

    except:
        print "error plotting learned params for band", band
        print traceback.format_exc()
    finally:
        pp.close()

f.close()










