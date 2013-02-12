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
from sigvisa.models.noise.armodel.learner import ARLearner
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
import sigvisa.utils.geog as geog
import obspy.signal.util

# from sigvisa.models.templates.paired_exp import *

(FIT_EVID, FIT_MB, FIT_LON, FIT_LAT, FIT_DEPTH, FIT_PHASEID, FIT_ATIME, FIT_PEAK_DELAY, FIT_CODA_HEIGHT, FIT_CODA_DECAY,
 FIT_AMP_TRANSFER, FIT_SITEID, FIT_DISTANCE, FIT_AZIMUTH, FIT_LOWBAND, FIT_NUM_COLS) = range(15 + 1)


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
            raise Exception("no existing runid for iteration %d of run %s!" % (iteration, run_name))
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
    sql_query = "select phase, round(param1,16), round(param2, 16), round(param3, 16), round(param4, 16) from sigvisa_coda_fit_phase where fitid=%d" % (
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


def store_template_params(wave, template_params, optim_param_str, iid, hz, acost, run_name, iteration, elapsed):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    runid = get_fitting_runid(cursor, run_name, iteration)

    sta = wave['sta']
    siteid = wave['siteid']
    chan = wave['chan']
    band = wave['band']
    st = wave['stime']
    et = wave['etime']
    event = get_event(evid=wave['evid'])

    distance = geog.dist_km((event.lon, event.lat), (s.sites[siteid - 1][0], s.sites[siteid - 1][1]))
    azimuth = geog.azimuth((s.sites[siteid - 1][0], s.sites[siteid - 1][1]), (event.lon, event.lat))

    optim_param_str = optim_param_str.replace("'", "''")

    sql_query = "INSERT INTO sigvisa_coda_fit (runid, evid, sta, chan, band, optim_method, iid, stime, etime, hz, acost, dist, azi, timestamp, elapsed) values (%d, %d, '%s', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f)" % (
        runid, event.evid, sta, chan, band, optim_param_str, 1 if iid else 0, st, et, hz, acost, distance, azimuth, time.time(), elapsed)

    fitid = execute_and_return_id(s.dbconn, sql_query, "fitid")

    (phases, fit_params) = template_params

    for (i, phase) in enumerate(phases):

        transfer = fit_params[i, 2] - event.source_logamp(band, phase)

        phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, param1, param2, param3, param4, amp_transfer) values (%d, '%s', 'paired_exp', %f, %f, %f, %f, %f)" % (
            fitid, phase, fit_params[i, 0], fit_params[i, 1], fit_params[i, 2], fit_params[i, 3], transfer)
        cursor.execute(phase_insert_query)
    return fitid


def load_shape_data(cursor, chan=None, band=None, sta=None, runids=None, phases=None, evids=None, exclude_evids=None, max_acost=200, min_azi=0, max_azi=360, min_mb=0, max_mb=100, min_dist=0, max_dist=20000, require_human_approved=False, min_amp=-10):

    chan_cond = "and fit.chan='%s'" % (chan) if chan is not None else ""
    band_cond = "and fit.band='%s'" % (band) if band is not None else ""
    site_cond = "and fit.sta='%s'" % (sta) if sta is not None else ""
    run_cond = "and (" + " or ".join(["fit.runid = %d" % int(runid) for runid in runids]) + ")" if runids is not None else ""
    phase_cond = "and (" + " or ".join(["fp.phase = '%s'" % phase for phase in phases]) + ")" if phases is not None else ""
    evid_cond = "and (" + " or ".join(["lebo.evid = %d" % evid for evid in evids]) + ")" if evids is not None else ""
    evid_cond = "and (" + " or ".join(
        ["lebo.evid != %d" % evid for evid in exclude_evids]) + ")" if exclude_evids is not None else ""
    approval_cond = "and human_approved==2" if require_human_approved else ""

    sql_query = "select distinct lebo.evid, lebo.mb, lebo.lon, lebo.lat, lebo.depth, fp.phase, fp.param1, fp.param2, fp.param3, fp.param4, fp.amp_transfer, fit.sta, fit.dist, fit.azi, fit.band from leb_origin lebo, sigvisa_coda_fit_phase fp, sigvisa_coda_fit fit where fp.fitid = fit.fitid and fit.acost<%f and fp.param3 > %f %s %s %s %s %s %s and fit.azi between %f and %f and fit.evid=lebo.evid and lebo.mb between %f and %f and fit.dist between %f and %f %s" % (
        max_acost, min_amp, chan_cond, band_cond, site_cond, run_cond, phase_cond, evid_cond, min_azi, max_azi, min_mb, max_mb, min_dist, max_dist, approval_cond)

    ensure_dir_exists(os.path.join(os.getenv('SIGVISA_HOME'), "db_cache"))
    fname = os.path.join(os.getenv('SIGVISA_HOME'), "db_cache", "%s.txt" % str(hashlib.md5(sql_query).hexdigest()))
    try:
        shape_data = np.loadtxt(fname, dtype=float)
    except:
        cursor.execute(sql_query)
        shape_data = np.array(cursor.fetchall(), dtype=object)
        print shape_data.size

        if shape_data.shape[0] > 0:
            s = Sigvisa()
            shape_data[:, FIT_SITEID] = np.asarray([s.name_to_siteid_minus1[sta] + 1 for sta in shape_data[:, FIT_SITEID]])
            shape_data[:, FIT_PHASEID] = np.asarray([s.phaseids[phase] for phase in shape_data[:, FIT_PHASEID]])
            shape_data[:, FIT_LOWBAND] = [b.split('_')[1] for b in shape_data[:, FIT_LOWBAND]]
            shape_data = np.array(shape_data, dtype=float)
            np.savetxt(fname, shape_data)
        else:
            raise Exception("found no shape data matching query %s" % sql_query)
    return shape_data


def insert_wiggle(dbconn, p):
    cursor = dbconn.cursor()
    sql_query = "insert into sigvisa_wiggle (fpid, stime, etime, srate, timestamp, type, log, meta0, meta1, meta2, params) values (:fpid, :stime, :etime, :srate, :timestamp, :type, :log, :meta0, :meta1, :meta2, :params)"

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
        raise Exception("blob insertion for mysql not yet implemented")

    return wiggleid


def read_wiggle(cursor, wiggleid):
    sql_query = "select * from sigvisa_wiggle where wiggleid=%d" % (wiggleid)
    cursor.execute(sql_query)
    return cursor.fetchone()


def insert_model(dbconn, fitting_runid, template_shape, param, site, chan, band, phase, model_type, model_fname, training_set_fname, training_ll, require_human_approved, max_acost, n_evids, min_amp):
    return execute_and_return_id(dbconn, "insert into sigvisa_template_param_model (fitting_runid, template_shape, param, site, chan, band, phase, model_type, model_fname, training_set_fname, n_evids, training_ll, timestamp, require_human_approved, max_acost, min_amp) values (:fr,:ts,:param,:site,:chan,:band,:phase,:mt,:mf,:tf, :ne, :tll,:timestamp, :require_human_approved, :max_acost, :min_amp)", "modelid", fr=fitting_runid, ts=template_shape, param=param, site=site, chan=chan, band=band, phase=phase, mt=model_type, mf=model_fname, tf=training_set_fname, tll=training_ll, timestamp=time.time(), require_human_approved='t' if require_human_approved else 'f', max_acost=max_acost, ne=n_evids, min_amp=min_amp)


def save_gsrun_to_db(d, segments, em, tm):
    s = Sigvisa()
    sql_query = "insert into sigvisa_gridsearch_run (evid, timestamp, elapsed, lon_nw, lat_nw, lon_se, lat_se, pts_per_side, likelihood_method, phases, wiggle_model_type, heatmap_fname, max_evtime_proposals, true_depth) values (:evid, :timestamp, :elapsed, :lon_nw, :lat_nw, :lon_se, :lat_se, :pts_per_side, :likelihood_method, :phases, :wiggle_model_type, :heatmap_fname, :max_evtime_proposals, :true_depth)"
    gsid = execute_and_return_id(s.dbconn, sql_query, "gsid", **d)

    for seg in segments:
        for chan in em.chans:
            for band in em.bands:
                gswid = execute_and_return_id(
                    s.dbconn, "insert into sigvisa_gsrun_wave (gsid, sta, chan, band, stime, etime, hz) values (%d, '%s', '%s', '%s', %f, %f, %f)" % (gsid, seg['sta'], chan, band,
                                                                                                                                                      seg['stime'], seg['etime'], seg.dummy_chan(chan)['srate']), 'gswid')
                for phase in em.phases:
                    for param in tm.models.keys():
                        modelid = tm.models[param][seg['sta']][phase][chan][band].modelid
                        gsmid = execute_and_return_id(
                            s.dbconn, "insert into sigvisa_gsrun_tmodel (gswid, modelid) values (%d, %d)" % (gswid, modelid), 'gsmid')

    s.dbconn.commit()
