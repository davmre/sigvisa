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
 FIT_CODA_DECAY, FIT_AMP_TRANSFER, FIT_NUM_COLS) = range(15 + 1)
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


def sql_param_condition(chan=None, band=None, site=None, runids=None, phases=None, evids=None, exclude_evids=None, max_acost=200, min_azi=0, max_azi=360, min_mb=0, max_mb=100, min_dist=0, max_dist=20000, require_human_approved=False, min_amp=-10):
    """

    assumes "from leb_origin lebo, sigvisa_coda_fit_phase fp, sigvisa_coda_fit fit"

    """

    chan_cond = "and fit.chan='%s'" % (chan) if chan is not None else ""
    band_cond = "and fit.band='%s'" % (band) if band is not None else ""
    site_cond = "and fit.sta='%s'" % (site) if site is not None else ""
    run_cond = "and (" + " or ".join(["fit.runid = %d" % int(runid) for runid in runids]) + ")" if runids is not None else ""
    phase_cond = "and (" + " or ".join(["fp.phase = '%s'" % phase for phase in phases]) + ")" if phases is not None else ""
    evid_cond = "and (" + " or ".join(["lebo.evid = %d" % evid for evid in evids]) + ")" if evids is not None else ""
    evid_cond = "and (" + " or ".join(
        ["lebo.evid != %d" % evid for evid in exclude_evids]) + ")" if exclude_evids is not None else ""
    approval_cond = "and human_approved=2" if require_human_approved else ""
    cost_cond = "and fit.acost<%f" % max_acost if np.isfinite(max_acost) else ""

    cond = "fp.fitid = fit.fitid and fp.coda_height > %f %s %s %s %s %s %s and fit.azi between %f and %f and fit.evid=lebo.evid and lebo.mb between %f and %f and fit.dist between %f and %f %s %s" % (min_amp, chan_cond, band_cond, site_cond, run_cond, phase_cond, evid_cond, min_azi, max_azi, min_mb, max_mb, min_dist, max_dist, approval_cond, cost_cond)

    return cond


def load_wiggle_data(cursor, basisid, **kwargs):

    from sigvisa.models.wiggles.wiggle_models import WiggleGenerator

    cond = sql_param_condition(**kwargs)

    sql_query = "select distinct lebo.evid, lebo.mb, lebo.lon, lebo.lat, lebo.depth, fp.phase, fit.sta, fit.dist, fit.azi, fit.band, wiggle.params from leb_origin lebo, sigvisa_coda_fit_phase fp, sigvisa_coda_fit fit, sigvisa_wiggle wiggle  where wiggle.fpid=fp.fpid and wiggle.basisid=%d and %s" % (basisid, cond)

    ensure_dir_exists(os.path.join(os.getenv('SIGVISA_HOME'), "db_cache"))
    fname = os.path.join(os.getenv('SIGVISA_HOME'), "db_cache", "%s.txt" % str(hashlib.md5(sql_query).hexdigest()))
    fname_sta = os.path.join(os.getenv('SIGVISA_HOME'), "db_cache", "%s_sta.txt" % str(hashlib.md5(sql_query).hexdigest()))
    try:
        wiggle_data = np.loadtxt(fname, dtype=float)
        sta_data = np.loadtxt(fname_sta, dtype=str)
    except:
        cursor.execute(sql_query)
        wiggle_data = np.array(cursor.fetchall(), dtype=object)
        print wiggle_data.size

        if wiggle_data.shape[0] > 0:
            s = Sigvisa()
            sta_data = np.array(wiggle_data[:, FIT_SITEID], dtype=str)
            wiggle_data[:, FIT_SITEID] = -1
            wiggle_data[:, FIT_PHASEID] = np.asarray([s.phaseids[phase] for phase in wiggle_data[:, FIT_PHASEID]])
            wiggle_data[:, FIT_LOWBAND] = [b.split('_')[1] for b in wiggle_data[:, FIT_LOWBAND]]

            wiggle_params = np.array([WiggleGenerator.decode_params(encoded = p) for p in wiggle_data[:, -1] ])
            wiggle_data = np.array(wiggle_data[:, :-1], dtype=float)
            wiggle_data = np.hstack([wiggle_data, wiggle_params])
            np.savetxt(fname, wiggle_data)
            np.savetxt(fname_sta, sta_data, "%s")
        else:
            raise NoDataException("found no wiggle data matching query %s" % sql_query)
    return wiggle_data, sta_data

def load_shape_data(cursor, **kwargs):

    cond = sql_param_condition(**kwargs)

    sql_query = "select distinct lebo.evid, lebo.mb, lebo.lon, lebo.lat, lebo.depth, fp.phase, fit.sta, fit.dist, fit.azi, fit.band, fp.arrival_time, fp.peak_offset, fp.coda_height, fp.coda_decay, fp.amp_transfer from  leb_origin lebo, sigvisa_coda_fit_phase fp, sigvisa_coda_fit fit where %s" % (cond)

    ensure_dir_exists(os.path.join(os.getenv('SIGVISA_HOME'), "db_cache"))
    fname = os.path.join(os.getenv('SIGVISA_HOME'), "db_cache", "%s.txt" % str(hashlib.md5(sql_query).hexdigest()))
    fname_sta = os.path.join(os.getenv('SIGVISA_HOME'), "db_cache", "%s_sta.txt" % str(hashlib.md5(sql_query).hexdigest()))
    try:
        shape_data = np.loadtxt(fname, dtype=float)
        sta_data = np.loadtxt(fname_sta, dtype=str)
    except:
        print sql_query
        cursor.execute(sql_query)
        print "sql done"
        shape_data = np.array(cursor.fetchall(), dtype=object)
        print shape_data.shape

        if shape_data.shape[0] > 0:
            s = Sigvisa()
            sta_data = np.array(shape_data[:, FIT_SITEID], dtype=str)
            shape_data[:, FIT_SITEID] = -1 #np.asarray([s.name_to_siteid_minus1[sta] + 1 for sta in shape_data[:, FIT_SITEID]])
            shape_data[:, FIT_PHASEID] = np.asarray([s.phaseids[phase] for phase in shape_data[:, FIT_PHASEID]])
            shape_data[:, FIT_LOWBAND] = [b.split('_')[1] for b in shape_data[:, FIT_LOWBAND]]
            shape_data = np.array(shape_data, dtype=float)
            np.savetxt(fname, shape_data)
            np.savetxt(fname_sta, sta_data, "%s")
        else:
            raise NoDataException("found no shape data matching query %s" % sql_query)
    return shape_data, sta_data


def insert_wiggle(dbconn, p):
    cursor = dbconn.cursor()
    sql_query = "insert into sigvisa_wiggle (fpid, basisid, timestamp, params) values (:fpid, :basisid, :timestamp, :params)"

    if "cx_Oracle" in str(type(dbconn)):
        import cx_Oracle
        binary_var = cursor.var(cx_Oracle.BLOB)
        binary_var.setvalue(0, p['params'])
        p['params'] = binary_var
        wiggleid = cursor.var(cx_Oracle.NUMBER)
        sql_query += " returning wiggleid into :x"
        p['x'] = wiggleid
        cursor.setinputsizes(params=cx_Oracle.BLOB)
        cursor.execute(sql_query, p)
        wiggleid = int(wiggleid.getvalue())

    elif "MySQLdb" in str(type(dbconn)):
        mysql_query = re.sub(r":(\w+)\s*([,)])", r"%(\1)s\2", sql_query)
        cursor.execute(mysql_query, p)
        wiggleid = cursor.lastrowid

    cursor.close()

    return wiggleid


def read_wiggle(cursor, wiggleid):
    sql_query = "select * from sigvisa_wiggle where wiggleid=%d" % (wiggleid)
    cursor.execute(sql_query)
    return cursor.fetchone()
