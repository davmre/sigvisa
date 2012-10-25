"""

Code for computing the cost/likelihood of an envelope template with respect to a waveform.

"""


import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils.geog
import obspy.signal.util

from signals.coda_decay_common import *

import sigvisa_c

def print_params(params):
    n = params.shape[0]
    for i in range(n):
        print "%d: st: %.1f pdelay: %.1f pheight: %.2f pdecay: %.4f cheight: %.2f cdecay: %.4f" % (i, params[i, 0], params[i, 1], params[i, 2], params[i, 3], params[i, 4], params[i, 5])

def set_dummy_wiggles(sigmodel, tr, phaseids):
    c = sigvisa_c.canonical_channel_num(tr.stats.channel)
    b = sigvisa_c.canonical_band_num(tr.stats.band)
    for pid in phaseids:
        sigmodel.set_wiggle_process(tr.stats.siteid, b, c, pid, 1, 0.05, np.array([.8,-.2]))

def set_noise_process(sigmodel, tr):
    c = sigvisa_c.canonical_channel_num(tr.stats.channel)
    b = sigvisa_c.canonical_band_num(tr.stats.band)
    arm = tr.stats.noise_model
    sigmodel.set_noise_process(tr.stats.siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))

def set_noise_processes(sigmodel, seg):
    for chan in seg.keys():
        c = sigvisa_c.canonical_channel_num(chan)
        for band in seg[chan].keys():
            b = sigvisa_c.canonical_band_num(band)
            siteid = seg[chan][band].stats.siteid
            try:
                arm = seg[chan][band].stats.noise_model
            except KeyError:
#                print "no noise model found for chan %s band %s, not setting.." % (chan, band)
                continue
            sigmodel.set_noise_process(siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))


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


def c_cost(wave, params, iid=False, sigmodel=None):

    if sigmodel is None:
        sigmodel = Sigvisa().sigmodel

#    noise_floor = params[-1]
#    params = np.reshape(params[:-1], (len(phaseids), -1))

    phases = params[0]
    vals = params[1]

    for i, pid in enumerate(phases):
        if np.isnan(vals[i, PEAK_HEIGHT_PARAM]) or np.isnan(vals[i, CODA_HEIGHT_PARAM]):
            return np.float('inf')
        if vals[i, CODA_HEIGHT_PARAM] > vals[i, PEAK_HEIGHT_PARAM] + 1:
            return np.float('inf')
        if vals[i, CODA_DECAY_PARAM] >= 0:
            return np.float('inf')
        if vals[i, PEAK_DECAY_PARAM] < 0:
            return np.float('inf')

#    print "trying heights ppeak %f pcoda %f speak %f scoda %f" % (params[0, PEAK_HEIGHT_PARAM], params[0, CODA_HEIGHT_PARAM], params[1, PEAK_HEIGHT_PARAM], params[1, CODA_HEIGHT_PARAM] )

    # we assume the noise params are already set...


    if iid:
        env = get_template(sigmodel, wave, params, logscale=True)
        c = logenv_l1_cost(np.log(wave.data), env.data)
    else:
        c = -1 *sigmodel.trace_likelihood(wave, params);

    print "cost", c

    return c

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

def get_template(sigmodel, trace, params, logscale=False, sample=False):
    srate = trace.stats['sampling_rate']
    st = trace.stats.starttime_unix
    et = st + trace.stats.npts/srate
    siteid = trace.stats.siteid
    c = sigvisa_c.canonical_channel_num(trace.stats.channel)
    b = sigvisa_c.canonical_band_num(trace.stats.band)
    if not sample:
        env = sigmodel.generate_trace(st, et, int(siteid), int(b), int(c), srate, params)
    else:
        env = sigmodel.sample_trace(st, et, int(siteid), int(b), int(c), srate, params)
    env.data = np.log(env.data) if logscale else env.data
    env.stats.noise_floor = trace.stats.noise_floor
    return env


