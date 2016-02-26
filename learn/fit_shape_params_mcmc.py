import os
import errno
import sys
import time
import traceback
import numpy as np
import numpy.ma as ma
import pyublas
import scipy
import uuid
from collections import defaultdict

from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.run_mcmc import run_open_world_MH, MCMCLogger
from sigvisa.models.signal_model import update_arrivals
from sigvisa.models.noise.noise_util import model_path
import sigvisa.utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import Sigvisa
from sigvisa.signals.io import *

from sigvisa.graph.sigvisa_graph import SigvisaGraph


def set_templates_from_fitid(sg, eid, fitid, wave):
    s = Sigvisa()
    phase_sql_query = "select fpid, phase, template_model, arrival_time, peak_offset, coda_height, peak_decay, coda_decay, mult_wiggle_std, wiggle_family from sigvisa_coda_fit_phase where fitid=%d" % fitid
    phase_details = s.sql(phase_sql_query)
    phases = [p[1] for p in phase_details]
    templates = {}

    for (phase, p) in zip(phases, phase_details):
        tparams = {'arrival_time': p[3], 'peak_offset': p[4], 'coda_height': p[5], 'coda_decay': p[7]}
        if p[2]=="lin_polyexp":
            tparams['peak_decay'] = p[6]
        if not sg.raw_signals:
            tparams['mult_wiggle_std'] = p[8]
        templates[phase] = tparams

    for phase in templates.keys():
        sg.set_template(eid=eid, sta=wave['sta'], band=wave['band'],
                        chan=wave['chan'], phase=phase,
                        values = templates[phase])
        print "setting template", eid, phase, "to", templates[phase]


def get_previous_fitid(runid, evid, sta):

    s = Sigvisa()

    try:
        sql = "select fitid from sigvisa_coda_fit where evid=%d and sta='%s' and runid=%d" % (evid, sta, runid)
        r = s.sql(sql)
        fitid = int(r[0][0])
    except:
        sta = s.get_default_sta(sta)
        sql = "select fitid from sigvisa_coda_fit where evid=%d and sta='%s' and runid=%d" % (evid, sta, runid)
        r = s.sql(sql)
        fitid = int(r[0][0])
        
    return fitid


def setup_graph(event, sta, chan, band,
                tm_shape, tm_type, wm_family, wm_type, phases,
                init_run_name, init_iteration, fit_hz=5, uatemplate_rate=1e-4,
                smoothing=0, dummy_fallback=False, raw_signals=False, init_templates=False):

    """
    Set up the graph with the signal for a given training event.
    """

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    try:
        input_runid = get_fitting_runid(cursor, init_run_name, init_iteration, create_if_new = False)
        runids = (input_runid,)
        print "input_runid", input_runid
    except RunNotFoundException:
        runids = ()

    sg = SigvisaGraph(template_model_type=tm_type, template_shape=tm_shape,
                      wiggle_model_type=wm_type, wiggle_family=wm_family,
                      phases=phases, 
                      runids = runids,
                      uatemplate_rate=uatemplate_rate,
                      min_mb=1.0,
                      dummy_fallback=dummy_fallback,
                      raw_signals=raw_signals)

    filter_str = band
    if not raw_signals:
        filter_str += ";env"

    wave = load_event_station_chan(event.evid, sta, chan, cursor=cursor, exclude_other_evs=True, phases=None if phases=="leb" else phases, pre_s=100.0).filter(filter_str)
    cursor.close()
    if smoothing > 0:
        wave = wave.filter('smooth_%d' % smoothing)
    if fit_hz != wave['srate']:
        wave = wave.filter('hz_%.2f' % fit_hz)

    if len(mask_blocks(wave.data.mask)) > 2:
        raise Exception("wave contains missing data")

    if (not raw_signals) and  (np.sum(wave.data < 0.0001) > 10):
        raise Exception("wave contains regions of zeros")

    sg.add_wave(wave=wave)
    sg.add_event(ev=event)

    if init_templates:
        fitid = get_previous_fitid(input_runid, event.evid, sta)
        set_templates_from_fitid(sg, 1, fitid, wave)

    #sg.fix_arrival_times()


    phases = sg.ev_arriving_phases(1, wave["sta"])
    assert( "P"  in phases or "Pg"  in phases or "Pn"  in phases or "pP"  in phases)
        

    return sg


def optimize_template_params(sigvisa_graph,  wn, tmpl_optim_params):

    nm1 = wn.nm_node.prior_nm.copy()
    #means = wn.signal_component_means()
    #noise_mean = means['noise']
    #noise_var = wn.signal_component_means(return_stds_instead=True)['noise']**2
    #nm2 = sample_posterior_armodel_from_signal(noise_mean, noise_var, wn.prior_nm), None

    if sigvisa_graph.template_shape=="lin_polyexp":
        nphases = int(len(sigvisa_graph.template_nodes)/6) # HACK

        v1 = np.array([ 0., 2.0, -0.5, -1, -3, 0.6] * nphases)
        v2 = np.array([ 2., -1, -1.5, -1, -1.5, 0.2] * nphases)
        v3 = np.array([ -3., 2.3, 0.5, -1, -3, 0.4] * nphases)
        init_vs = [(v1, nm1), (v2, nm1), (v3, nm1)]
    else:
        raise Exception("don't know how to initialize params for template shape %s" % sigvisa_graph.template_shape)

    best_vals = None
    best_prob = -np.inf
    best_nm = nm1
    for v, nm in init_vs:
        #wn.set_noise_model(nm, nmid)
        sigvisa_graph.set_all(values=v, node_list=sigvisa_graph.template_nodes)
        sigvisa_graph.optimize_templates(optim_params=tmpl_optim_params)
        v_result = sigvisa_graph.get_all(sigvisa_graph.template_nodes)
        v_p = sigvisa_graph.current_log_p()

        if v_p > best_prob:
            best_prob = v_p
            best_vals = v_result
            best_nm = nm



    #nm, nmid = best_nm
    #wn.set_noise_model(nm, nmid)
    sigvisa_graph.set_all(values=best_vals, node_list=sigvisa_graph.template_nodes)

def multiply_scalar_gaussian(m1, v1, m2, v2):
    """
    Unnormalized product of two Gaussian densities: precisions add, means are averaged weighted by precision.
    """

    prec = (1.0/v1 + 1.0/v2)
    v = 1.0/prec if ( np.abs(prec) > 1e-30) else 1e30
    m = v * (m1/v1 + m2/v2)
    return m, v

def compute_template_messages(sg, wn, logger, burnin=50, target_eid=None):
    """
    After an MCMC run, compute a Gaussian approximation to the
    posterior distribution on template parameters. Then divide out the
    (hopefully Gaussian) prior to yield the upwards message.

    Also return the template parameters with the highest posterior
    probability: these will be used to compute the wiggle posterior.
    """


    gp_messages = defaultdict(dict)
    best_vals = dict()
    for (eid, phase) in wn.arrivals():
        if target_eid is not None and eid != target_eid: 
            continue
        tvals, labels, lps = logger.load_template_vals(eid, phase, wn)

        tnodes = sg.get_template_nodes(eid, wn.sta, phase, wn.band, wn.chan)
        for p, (k, n) in tnodes.items():
            try:
                idx = labels.index(p)
            except ValueError:
                continue
            vals = tvals[:,idx]
            if n.deterministic():
                parent_key = n.default_parent_key()
                vals = np.array([n.invert(v, parent_key=parent_key) for v in vals])
                n = n.parents[parent_key]
                if p == "coda_height":
                    p = "amp_transfer"
                elif p == "arrival_time":
                    p = "tt_residual"

            m = np.mean(vals[burnin:])
            v = np.var(vals[burnin:]) + 1e-6 # avoid zero-variance problems

            pv = n._parent_values()
            if n.modeled_as_joint():
                # worry: using the conditional dist means that possibly messages are too weak.
                # whereas using the prior means they are too strong.
                # should figure out the right thing to do here. 
                d = n.joint_conditional_dist()
                prior_mean = d.predict()
                prior_var = d.variance()
            else:
                prior_mean = n.model.predict(cond=pv)
                prior_var = n.model.variance(cond=pv, include_obs=True)

            print "param", p, "prior", (prior_mean, prior_var), "posterior", (m, v)
            mm, mv = multiply_scalar_gaussian(m, v, prior_mean, -prior_var)
            if mv > 0:
                gp_messages[(eid, phase)][p] = float(mm), float(mv)
            else:
                gp_messages[(eid, phase)][p] = float(mm), -float(mv)
            gp_messages[(eid, phase)][p+"_posterior"] = m, v

        best_lp_idx = np.argmax(lps)
        best_vals[(eid, phase)] = (tvals[best_lp_idx,1:], [tnodes[lbl][1] for lbl in labels[1:] if lbl in tnodes])

    return gp_messages, best_vals

def compute_wavelet_messages(sg, wn, target_eid=None):

    gp_messages = dict()
    gp_posteriors = dict()

    wn._parent_values() # update the SSM to include current templates
    ell, marginals, step_ells = wn.tssm.all_filtered_cssm_coef_marginals(wn.get_value().data)

    for i, (eid, phase, scale, sidx, npts, component_type) in enumerate(wn.tssm_components):

        if target_eid is not None and eid != target_eid:
            continue
        
        posterior_means, posterior_vars = marginals[i]
        if len(posterior_means)==0:
            continue

        ssm = wn.tssm.get_component(i)
        prior_means, prior_vars = ssm.get_coef_prior()

        message_means = posterior_means.copy()
        message_vars = posterior_vars.copy()
        for j, (prm, prv, psm, psv) in enumerate(zip(prior_means, prior_vars, posterior_means, posterior_vars)):

            if psv < 1e-8:
                print "large posterior!"
                import pdb; pdb.set_trace()

            message_means[j], message_vars[j] = multiply_scalar_gaussian(psm, psv, prm, -prv)

            if message_vars[j] < 0:
                print "negative message!"
                import pdb; pdb.set_trace()

        gp_posteriors[(eid, phase)] = posterior_means, posterior_vars
        gp_messages[(eid, phase)] = message_means, message_vars

    return gp_messages, gp_posteriors

def run_fit(sigvisa_graph, fit_hz, tmpl_optim_params, output_runid, 
            steps, burnin, init_templates=False, enable_uatemplates=False):

    # initialize the MCMC by finding a good set of template params
    wn = sigvisa_graph.station_waves.values()[0][0]
    if not init_templates:
        # if we didn't already initialize from previously fit templates,
        # start by running an optimization to find reasonable values
        optimize_template_params(sigvisa_graph, wn, tmpl_optim_params)


    # run MCMC to sample from the posterior on template params
    st = time.time()

    logger = MCMCLogger(run_dir="scratch/mcmc_fit_%s/" % (str(uuid.uuid4())), write_template_vals=True, dump_interval_s=1000, transient=True, serialize_interval_s=1000, print_interval_s=5)
    run_open_world_MH(sigvisa_graph, steps=steps, enable_event_moves=False, enable_event_openworld=False, enable_phase_openworld=False, enable_template_openworld=enable_uatemplates, logger=logger, disable_moves=['atime_xc', 'constpeak_atime_xc'], tmpl_birth_rate=0.1)
    et = time.time()
    logger.dump(sigvisa_graph)

    # compute template posterior, and set the graph state to the best template params
    messages, best_tmvals = compute_template_messages(sigvisa_graph, wn, logger, burnin=burnin, target_eid=1)
    for (vals, nodes) in best_tmvals.values():
        for (v, n) in zip(vals, nodes):
            n.set_value(v)

    # compute wavelet posterior
    wavelet_messages, wavelet_posteriors = compute_wavelet_messages(sigvisa_graph, wn, target_eid=1)

    for k, v in wavelet_messages.items():
        messages[k][sigvisa_graph.wiggle_family] = v
        messages[k][sigvisa_graph.wiggle_family + "_posterior"] = wavelet_posteriors[k]


    tops=repr(tmpl_optim_params)[1:-1]
    fitids = save_template_params(sigvisa_graph, 1, wn.mw['evid'],
                                  tmpl_optim_param_str = "mcmc",
                                  wiggle_optim_param_str = "mcmc",
                                  elapsed=et - st, 
                                  runid=output_runid,
                                  messages= messages)
    return fitids[0]

def save_template_params(sg, eid, evid,
                         tmpl_optim_param_str,
                         wiggle_optim_param_str,
                         elapsed,
                         runid, messages):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    fitids = []
    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    run_wiggle_dir = os.path.join(wiggle_dir, "runid_" + str(runid))
    ensure_dir_exists(run_wiggle_dir)

    event = get_event(evid=evid)

    wns = [wn for wns in sg.station_waves.values() for wn in wns]
    for wave_node in wns:
        eids = set([eeid for (eeid, phase) in wave_node.arrivals()])
        if eid not in eids: continue

        wave = wave_node.mw

        sta = wave['sta']
        chan = wave['chan']
        band = wave['band']

        smooth = 0
        for fstr in wave['filter_str'].split(';'):
            if 'smooth' in fstr:
                smooth = int(fstr[7:])

        st = wave['stime']
        et = wave['etime']


        env_signals = "env" in wave["filter_str"]

        pv = wave_node._parent_values()
        arrivals = update_arrivals(pv)

        slon, slat, _, _, _, _, _ = s.earthmodel.site_info(sta, st)
        distance = geog.dist_km((event.lon, event.lat), (slon, slat))
        azimuth = geog.azimuth((slon, slat), (event.lon, event.lat))

        tmpl_optim_param_str = tmpl_optim_param_str.replace("'", "''")
        wiggle_optim_param_str = wiggle_optim_param_str.replace("'", "''")
        optim_log = wiggle_optim_param_str.replace("\n", "\\\\n")

        nm_fname = model_path(sta, chan, wave['filter_str'], wave_node.srate, wave_node.nm.p, window_stime=wave_node.st, model_type="ar") + "_inferred"
        c = 0
        full_fname = os.path.join(os.getenv('SIGVISA_HOME'), nm_fname)
        ensure_dir_exists(os.path.dirname(full_fname))
        while os.path.exists(full_fname + "_%d" % c):
            c += 1
        nm_fname = nm_fname + "_%d" % c
        full_fname = os.path.join(os.getenv('SIGVISA_HOME'), nm_fname)
        wave_node.nm.dump_to_file(full_fname)
        wave_node.nmid = wave_node.nm.save_to_db(dbconn=s.dbconn, sta=wave_node.sta, 
                                                 chan=wave_node.chan,
                                                 band=wave_node.band, hz=wave_node.srate, 
                                                 env=env_signals, smooth=smooth,
                                                 window_stime=wave_node.st, 
                                                 window_len=wave_node.et-wave_node.st,
                                                 fitting_runid=runid,
                                                 fname=nm_fname, hour=-1)
        print "saving inferred noise model as nmid", wave_node.nmid

        sql_query = "INSERT INTO sigvisa_coda_fit (runid, evid, sta, chan, band, smooth, tmpl_optim_method, wiggle_optim_method, optim_log, iid, stime, etime, hz, acost, dist, azi, timestamp, elapsed, nmid, env) values (%d, %d, '%s', '%s', '%s', '%d', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f, %d, '%s')" % (runid, event.evid, sta, chan, band, smooth, tmpl_optim_param_str, wiggle_optim_param_str, sg.optim_log, 1 , st, et, wave_node.srate, sg.current_log_p(), distance, azimuth, time.time(), elapsed, wave_node.nmid, 't' if env_signals else 'f')

        fitid = execute_and_return_id(s.dbconn, sql_query, "fitid")

        for (eeid, phase) in arrivals:
            if eeid != eid:
                continue

            fit_param_nodes = sg.get_template_nodes(eid=eid, phase=phase, chan=wave_node.chan, band=wave_node.band, sta=wave_node.sta)
            fit_params = dict([(p, n.get_value(k)) for (p,(k, n)) in fit_param_nodes.iteritems()])


            tg = sg.template_generator(phase)

            peak_decay = fit_params['peak_decay'] if 'peak_decay' in fit_params else 0.0
            mult_wiggle_std = fit_params['mult_wiggle_std'] if 'mult_wiggle_std' in fit_params else 0.0

            if eid > 0:
                phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay, amp_transfer, peak_decay, mult_wiggle_std, wiggle_family) values (%d, '%s', '%s', %f, %f, %f, %f, %f, %f, %f, '%s')" % (
                    fitid, phase, tg.model_name(), fit_params['arrival_time'], fit_params['peak_offset'], fit_params['coda_height'], fit_params['coda_decay'], fit_params['amp_transfer'], peak_decay, mult_wiggle_std, sg.wiggle_family)
            else:
                phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay, peak_decay, mult_wiggle_std, wiggle_family) values (%d, '%s', '%s', %f, %f, %f, %f, %f, %f, '%s')" % (
                    fitid, phase, tg.model_name(), fit_params['arrival_time'], fit_params['peak_offset'], fit_params['coda_height'], fit_params['coda_decay'], peak_decay, mult_wiggle_std, sg.wiggle_family)

            fpid = execute_and_return_id(s.dbconn, phase_insert_query, "fpid")
            for (k, n) in fit_param_nodes.values():
               n.fpid = fpid

            if eid < 0:
                # don't bother extracting wiggles for unass templates
                continue

            if messages is not None:
                message_dir = os.path.join(os.getenv("SIGVISA_HOME"), "training_messages")
                run_message_dir = os.path.join(message_dir, "runid_" + str(runid))
                ensure_dir_exists(run_message_dir)
                message_fname = "%d.msg" % (fpid,)
                with open(os.path.join(run_message_dir, message_fname), 'w') as f:
                    f.write(repr(messages[(eid, phase)] ))
                sql_query = "update sigvisa_coda_fit_phase set message_fname='%s' where fpid=%d" % (message_fname, fpid,)
                cursor.execute(sql_query)


        fitids.append(fitid)
        s.dbconn.commit()
    cursor.close()

    return fitids

def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("--tmpl_optim_params", dest="tmpl_optim_params", default="", type="str", help="fitting param string")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run name")
    parser.add_option("-i", "--run_iteration", dest="run_iteration", default=None, type="int",
                      help="run iteration (default is to use the next iteration)")
    parser.add_option("--init_run_name", dest="init_run_name", default=None, type="str", help="run name")
    parser.add_option("--init_run_iteration", dest="init_run_iteration", default=None, type="int",
                      help="run iteration ")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
    parser.add_option("--orid", dest="orid", default=None, type="int", help="origin ID")
    parser.add_option("--template_shape", dest="template_shape", default="lin_polyexp", type="str",
                      help="template model type to fit parameters under (lin_polyexp)")
    parser.add_option("--template_model", dest="template_model", default="dummyPrior", type="str", help="")
    parser.add_option("--dummy_fallback", dest="dummy_fallback", default=False, action="store_true", help="")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=None, type="float", help="if nonzero, allow uatemplate births to explain signal spikes")
    parser.add_option("--raw_signals", dest="raw_signals", default=False, action="store_true", help="fit on raw signals instead of envelopes")
    parser.add_option("--init_templates", dest="init_templates", default=False, action="store_true", help="initialize optimization from previous template fits (False)")
    parser.add_option("--wiggle_family", dest="wiggle_family", default="dummy", type="str", help="")

    parser.add_option("--wiggle_model", dest="wiggle_model", default="dummy", type="str", help="")
    parser.add_option("--phases", dest="phases", default="leb", type="str", help="")
    parser.add_option("--band", dest="band", default="freq_2.0_3.0", type="str", help="")
    parser.add_option("--chan", dest="chan", default="auto", type="str", help="")
    parser.add_option("--smooth", dest="smooth", default=0, type=int, help="perform the given level of smoothing")
    parser.add_option("--steps", dest="steps", default=500, type=int, help="number of MCMC steps to run (500)")
    parser.add_option("--burnin", dest="burnin", default=50, type=int, help="number of initial MCMC steps to disregard (50)")
    parser.add_option("--hz", dest="hz", default=5.0, type="float", help="sampling rate at which to fit the template")
    parser.add_option("--seed", dest="seed", default=0, type="int",
                      help="ranom seed for MCMC (0)")
    parser.add_option("--nocheck", dest="nocheck", default=False, action="store_true", help="don't check to see if we've already fit this arrival in this run")


    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    if options.phases == "leb":
        phases = "leb"
    else:
        phases = options.phases.split(',')

    if '{' in options.template_model:
        # example: "{'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb'}"
        template_model = eval(options.template_model)
    else:
        template_model = options.template_model

    if options.run_name is None or options.run_iteration is None:
        raise ValueError("must specify run name and iteration!")

    if not (options.evid is None and options.orid is None):
        ev = get_event(evid=options.evid, orid=options.orid)
    else:
        raise ValueError("Must specify event id (evid) or origin id (orid) to fit.")

    init_run_name = options.run_name
    init_iteration = options.run_iteration - 1
    if options.init_run_name is not None:
        init_run_name = options.init_run_name
        init_iteration = options.init_run_iteration

    if options.template_model == "hack28":
        init_run_name = "multiphase_wiggle"
        init_iteration = 1
        template_model = {'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb', 'peak_decay': 'param_linear_distmb', 'mult_wiggle_std': 'dummyPrior'}
    if options.template_model == "param":
        template_model = {'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb', 'peak_decay': 'param_linear_distmb', 'mult_wiggle_std': 'dummyPrior'}
        

    uatemplate_rate = options.uatemplate_rate
    enable_uatemplates = uatemplate_rate is not None

    sigvisa_graph = setup_graph(event=ev, sta=options.sta, chan=options.chan, band=options.band,
                                tm_shape=options.template_shape, tm_type=template_model,
                                wm_family=options.wiggle_family, wm_type=options.wiggle_model,
                                phases=phases,
                                fit_hz=options.hz, 
                                uatemplate_rate = uatemplate_rate if enable_uatemplates else 1e-4,
                                init_run_name = init_run_name, init_iteration = init_iteration,
                                smoothing=options.smooth,
                                init_templates=options.init_templates,
                                dummy_fallback=options.dummy_fallback, raw_signals=options.raw_signals)

    runid = get_fitting_runid(cursor, options.run_name, options.run_iteration, create_if_new = True)
    if not options.nocheck:
        wn = sigvisa_graph.station_waves.values()[0][0]
        sql_query = "select * from sigvisa_coda_fit where runid=%d and evid=%d and sta='%s' and chan='%s' and band='%s' and smooth=%d and hz=%d" % (runid, ev.evid, wn.sta, wn.chan, wn.band, options.smooth, options.hz)
        cursor.execute(sql_query)
        r = cursor.fetchall()
        if len(r) > 0:
            print "not fitting because a similar fit already exists in this runid"
            print r
            print "run with --nocheck flag to override"
            return


    if options.seed >= 0:
        np.random.seed(options.seed)

    fitid = run_fit(sigvisa_graph,  fit_hz = options.hz, enable_uatemplates=enable_uatemplates,
                    tmpl_optim_params=construct_optim_params(options.tmpl_optim_params),
                    init_templates=options.init_templates,
                    output_runid = runid, steps=options.steps, burnin=options.burnin)

    print "fit id %d completed successfully." % fitid

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
