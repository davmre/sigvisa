import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import signals.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util

from priors.coda_decay.coda_decay_common import *




def print_params(params):
    n = params.shape[0]
    for i in range(n):
        print "%d: st: %.1f pdelay: %.1f pheight: %.2f pdecay: %.4f cheight: %.2f cdecay: %.4f" % (i, params[i, 0], params[i, 1], params[i, 2], params[i, 3], params[i, 4], params[i, 5])

def set_dummy_wiggles(sigmodel, tr, phaseids):
    c = sigvisa.canonical_channel_num(tr.stats.channel)
    b = sigvisa.canonical_band_num(tr.stats.band)
    for pid in phaseids:
        sigmodel.set_wiggle_process(tr.stats.siteid, b, c, pid, 1, 0.05, np.array([.8,-.2]))

def set_noise_process(sigmodel, tr):
    c = sigvisa.canonical_channel_num(tr.stats.channel)
    b = sigvisa.canonical_band_num(tr.stats.band)
    arm = tr.stats.noise_model
    sigmodel.set_noise_process(tr.stats.siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))

def set_noise_processes(sigmodel, seg):
    for chan in seg.keys():
        c = sigvisa.canonical_channel_num(chan)
        for band in seg[chan].keys():
            b = sigvisa.canonical_band_num(band)
            siteid = seg[chan][band].stats.siteid
            try:
                arm = seg[chan][band].stats.noise_model
            except KeyError:
#                print "no noise model found for chan %s band %s, not setting.." % (chan, band)
                continue
            sigmodel.set_noise_process(siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))

def c_cost(sigmodel, smoothed, phaseids, params, iid=False):

#    noise_floor = params[-1]
#    params = np.reshape(params[:-1], (len(phaseids), -1))

    params = np.reshape(params, (len(phaseids), -1))

    for i, pid in enumerate(phaseids):
        if np.isnan(params[i, PEAK_HEIGHT_PARAM]) or np.isnan(params[i, CODA_HEIGHT_PARAM]):
            return np.float('inf')
        if params[i, CODA_HEIGHT_PARAM] > params[i, PEAK_HEIGHT_PARAM] + 1:
            return np.float('inf')
        if params[i, CODA_DECAY_PARAM] >= 0:
            return np.float('inf')
        if params[i, PEAK_DECAY_PARAM] < 0:
            return np.float('inf')

#    print "trying heights ppeak %f pcoda %f speak %f scoda %f" % (params[0, PEAK_HEIGHT_PARAM], params[0, CODA_HEIGHT_PARAM], params[1, PEAK_HEIGHT_PARAM], params[1, CODA_HEIGHT_PARAM] )

    # we assume the noise params are already set...

    print "params"
    print_params(params)

    if iid:
        env = get_template(sigmodel, smoothed, phaseids, params, logscale=True)
        c = logenv_l1_cost(np.log(smoothed.data), env.data)
    else:
        c = -1 *sigmodel.trace_likelihood(smoothed, phaseids, params);

    print "cost", c

    return c

# params with peak but without arrtime
def remove_peak(pp):
    newp = np.zeros((pp.shape[0], NUM_PARAMS-3))
    newp[:, 0] = pp[:, PEAK_OFFSET_PARAM-1]
    newp[:, 1] = pp[:, CODA_HEIGHT_PARAM-1]
    newp[:, 2] = pp[:, CODA_DECAY_PARAM-1]
    return newp

def restore_peak(peakless_params):
    p = peakless_params
    newp = np.zeros((p.shape[0], NUM_PARAMS-1))
    newp[:, 0] = p[:, 0]
    newp[:, 1] = p[:, 1]
    newp[:, 2] = 1
    newp[:, 3] = p[:, 1]
    newp[:, 4] = p[:, 2]
    return newp

def load_template_params(cursor, evid, chan, band, runid, siteid):
    sql_query = "select l.time, fit.peak_delay, fit.peak_height, fit.peak_decay, fit.coda_height, fit.coda_decay, fit.acost, pid.id from sigvisa_coda_fits fit, leb_assoc leba, leb_origin lebo, static_siteid sid, static_phaseid pid, leb_arrival l where lebo.orid=leba.orid and leba.arid=fit.arid and leba.phase=pid.phase and l.arid=leba.arid and l.sta=sid.sta and lebo.evid=%d and fit.chan='%s' and fit.band='%s' and fit.runid=%d and sid.id=%d" % (evid, chan, band, runid, siteid)
    cursor.execute(sql_query)
    rows = np.array(cursor.fetchall())
    for (i, row) in enumerate(rows):
        if row[2] is None:
            row[2] = row[4]
            row[3] = 1

    try:
        fit_params = np.asfarray(rows[:, 0:6])
        phaseids = list(rows[:, 7])
        fit_cost = rows[0,6]
    except IndexError:
        return None, None, None
    return fit_params, phaseids, fit_cost

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
    env.stats.noise_floor = trace.stats.noise_floor
    return env


