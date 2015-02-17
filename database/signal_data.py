import os
import errno
import sys
import time
import traceback
import hashlib
import time
import re
import numpy as np
import scipy
import scipy.stats
import cPickle as pickle

from sigvisa.database.dataset import *
from sigvisa.database import db

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
import sigvisa.utils.geog as geog
import obspy.signal.util



# from sigvisa.models.templates.paired_exp import *

(FIT_EVID, FIT_MB, FIT_LON, FIT_LAT, FIT_DEPTH, FIT_PHASEID,
 FIT_SITEID, FIT_DISTANCE, FIT_AZIMUTH,
 FIT_LOWBAND, FIT_ATIME, FIT_PEAK_DELAY, FIT_CODA_HEIGHT,
 FIT_PEAK_DECAY, FIT_CODA_DECAY, FIT_AMP_TRANSFER, FIT_NUM_COLS) = range(16 + 1)
WIGGLE_PARAM0 = FIT_ATIME

class RunNotFoundException(Exception):
    pass

class NoDataException(Exception):
    pass


def ensure_dir_exists(dname):
    try:
        os.makedirs(dname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise
    return dname


def get_base_dir(sta, run_name, label=None):
    if label is not None:
        return ensure_dir_exists(os.path.join("logs", "codas_%s_%s_%s" % (sta, label, run_name)))
    else:
        return ensure_dir_exists(os.path.join("logs", "codas_%s_%s" % (sta, run_name)))


def get_next_runid(cursor):
    sql_query = "select max(runid) from sigvisa_coda_fitting_run"
    cursor.execute(sql_query)
    r = cursor.fetchone()[0]
    if r is None:
        runid = 1
    else:
        runid = r + 1
    return runid


def get_fitting_runid(cursor, run_name, iteration, create_if_new=True):
    sql_query = "select runid from sigvisa_coda_fitting_run where run_name='%s' and iter=%d" % (run_name, iteration)
    cursor.execute(sql_query)
    result = cursor.fetchone()
    if result is None:
        if not create_if_new:
            raise RunNotFoundException("no existing runid for iteration %d of run %s!" % (iteration, run_name))
        runid = get_next_runid(cursor)
        sql_query = "insert into sigvisa_coda_fitting_run (runid, run_name, iter) values (%d, '%s', %d)" % (
            runid, run_name, iteration)
        cursor.execute(sql_query)
    else:
        runid = result[0]
    return runid


def read_fitting_run(cursor, runid):
    sql_query = "select run_name, iter from sigvisa_coda_fitting_run where runid=%d" % runid
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
    sql_query = "select iter, runid from sigvisa_coda_fitting_run where run_name='%s'" % run_name
    cursor.execute(sql_query)
    r = np.reshape(np.array(cursor.fetchall()), (-1, 2))
    return np.array(sorted(r))


def get_fitid(cursor, evid, sta, chan, band, run_name=None, iteration=None, runid=None):

    if runid is None:
        runid = get_fitting_runid(cursor, run_name, iteration, create_if_new=False)

    sql_query = "select fitid from sigvisa_coda_fit where sta='%s' and evid=%d and chan='%s' and band='%s' and runid=%d" % (
        sta, evid, chan, band, runid)
    cursor.execute(sql_query)
    fitid = cursor.fetchone()[0]
    return fitid


def filter_and_sort_template_params(unsorted_phases, unsorted_params, filter_list):
    phase_indices = zip(unsorted_phases, range(len(unsorted_phases)))
    phase_indices = [(p, i) for (p, i) in phase_indices if p in filter_list]
    tmp = sorted(phase_indices, key=lambda z: filter_list.index(z[0]))
    (phases, permutation) = zip(*tmp)
    fit_params = unsorted_params[permutation, :]
    return (phases, fit_params)


def benchmark_fitting_run(cursor, runid, return_raw_data=False):
    sql_query = "select acost, elapsed from sigvisa_coda_fit where runid=%d" % runid
    cursor.execute(sql_query)
    results = np.array(cursor.fetchall())
    acosts = results[:, 0]
    times = results[:, 1]
    if return_raw_data:
        return acosts, times
    else:
        return np.mean(acosts), np.mean(times)


def load_template_params_by_fitid(cursor, fitid, return_cost=True):

    s = Sigvisa()
    sql_query = "select phase, round(arrival_time,16), round(peak_offset, 16), round(coda_height, 16), round(coda_decay, 16) from sigvisa_coda_fit_phase where fitid=%d" % (
        fitid)
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    try:
        fit_params = np.asfarray([row[1:5] for row in rows])
        phases = tuple([r[0] for r in rows])
        (phases, fit_params) = filter_and_sort_template_params(phases, fit_params, filter_list=s.phases)
    except IndexError as e:
        print e
        return (None, None), None

    if return_cost:
        sql_query = "select acost from sigvisa_coda_fit where fitid=%d" % fitid
        print sql_query
        cursor.execute(sql_query)
        fit_cost = cursor.fetchone()[0]
        return (phases, fit_params), fit_cost
    else:
        return (phases, fit_params)


def load_template_params(cursor, evid, sta, chan, band, run_name=None, iteration=None, runid=None):
    fitid = get_fitid(cursor, evid, sta, chan, band, run_name=None, iteration=None, runid=None)
    p, c = load_template_params_by_fitid(cursor, fitid)
    return p, c, fitid


def execute_and_return_id(dbconn, query, idname, **kwargs):
    cursor = dbconn.cursor()
    if "cx_Oracle" in str(type(dbconn)):
        import cx_Oracle
        myseq = cursor.var(cx_Oracle.NUMBER)
        query += " returning %s into :rbfhaj" % (idname,)
        cursor.execute(query, rbfhaj=myseq, **kwargs)
        lrid = int(myseq.getvalue())
    elif "MySQLdb" in str(type(dbconn)):
        mysql_query = re.sub(r":(\w+)\s*([,)])", r"%(\1)s\2", query)
        cursor.execute(mysql_query, args=kwargs)
        lrid = cursor.lastrowid

    cursor.close()
    dbconn.commit()
    return lrid


def sql_param_condition(chan=None, band=None, site=None, runids=None, phases=None, evids=None, exclude_evids=None, max_acost=200, min_azi=0, max_azi=360, min_mb=0, max_mb=100, min_dist=0, max_dist=20000, require_human_approved=False, min_amp=-10, wiggle_family=None):
    """

    assumes "from leb_origin lebo, sigvisa_coda_fit_phase fp, sigvisa_coda_fit fit"

    """

    chan_cond = "and fit.chan='%s'" % (chan) if chan is not None else ""
    band_cond = "and fit.band='%s'" % (band) if band is not None else ""
    site_cond = "and fit.sta='%s'" % (site) if site is not None else ""
    run_cond = "and (" + " or ".join(["fit.runid = %d" % int(runid) for runid in runids]) + ")" if runids is not None else ""
    phase_cond = "and (" + " or ".join(["fp.phase = '%s'" % phase for phase in phases]) + ")" if phases is not None else ""
    evid_cond = "and (" + " or ".join(["fit.evid = %d" % evid for evid in evids]) + ")" if evids is not None else ""
    evid_cond = "and (" + " or ".join(
        ["fit.evid != %d" % evid for evid in exclude_evids]) + ")" if exclude_evids is not None else ""
    approval_cond = "and human_approved=2" if require_human_approved else ""
    cost_cond = "and fit.acost<%f" % max_acost if np.isfinite(max_acost) else ""

    wiggle_cond = "and fp.wiggle_family='%s'" % (wiggle_family) if wiggle_family is not None else ""

    cond = "fp.fitid = fit.fitid and fp.coda_height > %f %s %s %s %s %s %s and fit.azi between %f and %f and fit.evid=lebo.evid and lebo.mb between %f and %f and fit.dist between %f and %f %s %s %s" % (min_amp, chan_cond, band_cond, site_cond, run_cond, phase_cond, evid_cond, min_azi, max_azi, min_mb, max_mb, min_dist, max_dist, approval_cond, cost_cond, wiggle_cond)

    return cond


class SavedFit(object):
    def __init__(self, ev, phase, sta, band,
                 dist, azi, arrival_time,
                 peak_offset, coda_height,
                 peak_decay, coda_decay, messages):
        self.ev = ev
        self.phase = phase
        self.sta = sta
        self.band = band
        self.dist = dist
        self.azi = azi
        self.arrival_time=arrival_time
        self.peak_offset=peak_offset
        self.coda_height=coda_height
        self.peak_decay=peak_decay
        self.coda_decay=coda_decay
        self.messages=messages

def load_training_messages(cursor, **kwargs):
    cond = sql_param_condition(**kwargs)

    sql_query = "select distinct lebo.evid, fp.phase, fit.sta, fit.dist, fit.azi, fit.band, fp.arrival_time, fp.peak_offset, fp.coda_height, fp.peak_decay, fp.coda_decay, fit.runid, fp.message_fname from leb_origin lebo, sigvisa_coda_fit_phase fp, sigvisa_coda_fit fit where %s" % (cond)

    s = Sigvisa()
    ensure_dir_exists(os.path.join(s.homedir, "db_cache"))
    fname = os.path.join(s.homedir, "db_cache", "%s.txt" % str(hashlib.md5(sql_query).hexdigest()))
    try:
        with open(fname, 'rb') as f:
            fits = pickle.load(f)
    except:
        cursor.execute(sql_query)
        message_data = np.array(cursor.fetchall(), dtype=object)

        fits = []
        for row in message_data:
            evid, phase, sta, dist, azi, band, atime, peak_offset, coda_height, peak_decay, coda_decay, runid, message_fname = row

            messages = None
            if message_fname is not None and message_fname.endswith("msg"):
                message_dir = os.path.join(s.homedir, "training_messages", "runid_%d" % runid)
                message_full_fname = os.path.join(message_dir, message_fname)
                with open(message_full_fname, 'r') as f:
                    message_str = f.read()
                messages = eval(message_str, {'array': np.array})

            ev = get_event(evid=evid, cursor=cursor)
            fit = SavedFit(ev=ev, phase=phase, sta=sta, band=band, dist=dist,azi=azi, arrival_time=atime, peak_offset=peak_offset, coda_height=coda_height, peak_decay=peak_decay, coda_decay=coda_decay, messages=messages)
            fits.append(fit)

        if len(fits) > 0:
            with open(fname, 'wb') as f:
                pickle.dump(fits, f, 2)
        else:
            raise NoDataException("found no wiggle data matching query %s" % sql_query)
    return fits
