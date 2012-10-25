import os, errno, sys, time, traceback, hashlib
import numpy as np, scipy, scipy.stats

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plot
import sigvisa_c
from signals.armodel.learner import ARLearner
from signals.armodel.model import ARModel, ErrorModel
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
            b  = sigvisa_c.canonical_band_num(short_band)
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
    run_cond = "and (" + " or ".join(["fit.runid = %d" % runid for runid in runids]) + ")" if runids is not None else ""
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


