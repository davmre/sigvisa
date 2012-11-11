import os, errno, sys, time, traceback, hashlib
import numpy as np, scipy, scipy.stats

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
from sigvisa import Sigvisa
from source.event import Event
import sigvisa_c
from signals.armodel.learner import ARLearner
from signals.armodel.model import ARModel, ErrorModel
from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.geog
import obspy.signal.util

# from signals.template_models.paired_exp import *

def ensure_dir_exists(dname):
    try:
        os.makedirs(dname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise
    return dname

def get_base_dir(sta, run_name, label=None):
    if label is not None:
        return ensure_dir_exists(os.path.join("logs", "codas_%s_%s_%s" % (sta, label, run_name)))
    else:
        return ensure_dir_exists(os.path.join("logs", "codas_%s_%s" % (sta, run_name)))

def get_next_runid(cursor):
    sql_query = "select max(runid) from sigvisa_coda_fitting_runs"
    cursor.execute(sql_query)
    r = cursor.fetchone()[0]
    if r is None:
        runid = 1
    else:
        runid = r+1
    return runid

def get_fitting_runid(cursor, run_name, iteration, create_if_new=True):
    sql_query = "select runid from sigvisa_coda_fitting_runs where run_name='%s' and iter=%d" % (run_name, iteration)
    cursor.execute(sql_query)
    result = cursor.fetchone()
    if result is None:
        if not create_if_new:
            raise Exception("no existing runid for iteration %d of run %s!")
        runid = get_next_runid(cursor)
        sql_query = "insert into sigvisa_coda_fitting_runs (runid, run_name, iter) values (%d, '%s', %d)" % (runid, run_name, iteration)
        cursor.execute(sql_query)
    else:
        runid = result[0]
    return runid

def read_fitting_run(cursor, runid):
    sql_query = "select run_name, iter from sigvisa_coda_fitting_runs where runid=%d" % runid
    cursor.execute(sql_query)
    info = cursor.fetchone()
    if info is None:
        raise Exception("no entry in DB for runid %d" % runid)
    else:
        (run_name, iteration) = info
    return (run_name, iteration)

def get_last_iteration(cursor, run_name):
    iters = read_fitting_run_iterations(cursor, run_name)
    last_iter = iters[-1][0] if len(iters) > 0 else 0
    return last_iter

def read_fitting_run_iterations(cursor, run_name):
    sql_query = "select iter, runid from sigvisa_coda_fitting_runs where run_name='%s'" % run_name
    cursor.execute(sql_query)
    r = cursor.fetchall()
    return sorted(r)

def load_template_params(evid, sta, chan, band, run_name, iteration):
    s = Sigvisa()

    runid = get_fitting_runid(s.cursor, run_name, iteration, create_if_new=False)

    pieces = band.split('_')
    lowband = float(pieces[1])
    highband = float(pieces[2])

    sql_query = "select round(atime,4), peak_delay, coda_height, coda_decay, acost, phase from sigvisa_coda_fits where sta='%s' and evid=%d and chan='%s' and lowband=%f and highband=%f and runid=%d" % (sta, evid, chan, lowband, highband, runid)
    s.cursor.execute(sql_query)
    rows = s.cursor.fetchall()
    try:
        all_phases = s.phases
        fit_params =np.asfarray([row[0:4] for row in rows])
        phases = tuple([r[5] for r in rows])
        tmp = sorted(zip(phases, range(len(phases))), key = lambda z : all_phases.index(z[0]))
        (phases, permutation) = zip(*tmp)
        fit_params = fit_params[permutation, :]

        fit_cost = rows[0][4]
    except IndexError as e:
        print e
        return (None, None), None
    return (phases, fit_params), fit_cost

def store_template_params(wave, template_params, method_str, iid, fit_cost, run_name, iteration):
    s  = Sigvisa()

    runid = get_fitting_runid(s.cursor, run_name, iteration)

    sta = wave['sta']
    siteid = wave['siteid']
    chan = wave['chan']
    band = wave['band']
    st = wave['stime']
    et = wave['etime']
    time_len = wave['len']
    event = Event(evid=wave['evid'])

    pieces = band.split('_')
    lowband = float(pieces[1])
    highband = float(pieces[2])

    distance = utils.geog.dist_km((event.lon, event.lat), (s.sites[siteid-1][0], s.sites[siteid-1][1]))
    azimuth = utils.geog.azimuth((s.sites[siteid-1][0], s.sites[siteid-1][1]), (event.lon, event.lat))

    (phases, fit_params) = template_params

    PE_ARR_TIME_PARAM, PE_PEAK_OFFSET_PARAM, PE_CODA_HEIGHT_PARAM, PE_CODA_DECAY_PARAM, PE_NUM_PARAMS = range(4+1)

    for (i, phase) in enumerate(phases):
        sql_query = "INSERT INTO sigvisa_coda_fits (runid, evid, sta, chan, lowband, highband, phase, atime, peak_delay, coda_height, coda_decay, optim_method, iid, stime, etime, acost, dist, azi) values (%d, %d, '%s', '%s', %f, %f, '%s', %f, %f, %f, %f, '%s', %d, %f, %f, %f, %f, %f)" % (runid, event.evid, sta, chan, lowband, highband, phase, fit_params[i, PE_ARR_TIME_PARAM], fit_params[i, PE_PEAK_OFFSET_PARAM], fit_params[i, PE_CODA_HEIGHT_PARAM], fit_params[i, PE_CODA_DECAY_PARAM], method_str, 1 if iid else 0, st, et, fit_cost/time_len, distance, azimuth)
        try:
            s.cursor.execute(sql_query)
        except Exception as e:
            print e
            print "DB error inserting fits (probably duplicate key), continuing..."
            pass

def filter_shape_data(fit_data, chan=None, short_band=None, siteid=None, runid=None, phaseids=None, evids=None, min_azi=0, max_azi=360, min_mb=0, max_mb=100, min_dist=0, max_dist=20000):

    new_data = []
    for row in fit_data:
        if chan is not None:
            if row[FIT_CHAN] != chan:
                continue
        if short_band is not None:
            b  = sigvisa_c.canonical_band_num(short_band)
            if int(row[FIT_BANDID]) != b:
                continue
        if siteid is not None:
            if int(row[FIT_SITEID]) != siteid:
                continue
        if runid is not None:
            if row[FIT_RUNID] != run_name:
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


def load_wiggle_models(cursor, sigmodel, filename):
    f = open(filename, 'r')
    for line in f:
        entries = line.split()
        siteid = sta_to_siteid(entries[0], cursor)
        phaseid = phasename_to_id(entries[1])
        c = sigvisa_c.canonical_channel_num(entries[2])
        b = sigvisa_c.canonical_band_num(entries[3])
        mean = float(entries[4])
        std = float(entries[5])
        order = int(entries[6])
        params = [float(x) for x in entries[7:7+order]]
        sigmodel.set_wiggle_process(siteid, b, c, phaseid, mean, std, np.asfarray(params))


def load_shape_data(cursor, chan=None, short_band=None, siteid=None, runids=None, phaseids=None, evids=None, exclude_evids=None, acost_threshold=20, min_azi=0, max_azi=360, min_mb=0, max_mb=100, min_dist=0, max_dist=20000):

    chan_cond = "and fit.chan='%s'" % (chan) if chan is not None else ""
    band_cond = "and fit.band='%s'" % (short_band) if short_band is not None else ""
    site_cond = "and sid.id=%d" % (siteid) if siteid is not None else ""
    run_cond = "and (" + " or ".join(["fit.runid = %d" % int(runid) for runid in runids]) + ")" if runids is not None else ""
    phase_cond = "and (" + " or ".join(["pid.id = %d" % phaseid for phaseid in phaseids]) + ")" if phaseids is not None else ""
    evid_cond = "and (" + " or ".join(["lebo.evid = %d" % evid for evid in evids]) + ")" if evids is not None else ""
    evid_cond = "and (" + " or ".join(["lebo.evid != %d" % evid for evid in exclude_evids]) + ")" if exclude_evids is not None else ""

    sql_query = "select distinct lebo.evid, lebo.mb, lebo.lon, lebo.lat, lebo.depth, pid.id, fit.peak_delay, fit.coda_height, fit.coda_decay, sid.id, fit.dist, fit.azi, fit.band from leb_origin lebo, leb_assoc leba, leb_arrival l, sigvisa_c_coda_fits fit, static_siteid sid, static_phaseid pid where fit.arid=l.arid and l.arid=leba.arid and leba.orid=lebo.orid and leba.phase=pid.phase and sid.sta=l.sta %s %s %s %s %s %s and fit.acost<%f and fit.peak_delay between -10 and 20 and fit.coda_decay>-0.2 and fit.azi between %f and %f and lebo.mb between %f and %f and fit.dist between %f and %f" % (chan_cond, band_cond, site_cond, run_cond, phase_cond, evid_cond, acost_threshold, min_azi, max_azi, min_mb, max_mb, min_dist, max_dist)

    fname = "db_cache/%s.txt" % str(hashlib.md5(sql_query).hexdigest())
    try:

        shape_data = np.loadtxt(fname, dtype=float)
    except:
        cursor.execute(sql_query)
        shape_data = np.array(cursor.fetchall(), dtype=object)
        print shape_data.size
        shape_data[:, FIT_BANDID] = np.asarray([sigvisa_c.canonical_band_num(band) for band in shape_data[:, FIT_BANDID]])
        shape_data = np.array(shape_data, dtype=float)
        np.savetxt(fname, shape_data)

    return shape_data


