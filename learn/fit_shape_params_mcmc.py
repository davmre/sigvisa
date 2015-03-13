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
from sigvisa.infer.autoregressive_mcmc import sample_posterior_armodel_from_signal
from sigvisa.models.signal_model import update_arrivals
from sigvisa.models.noise.noise_util import model_path
import sigvisa.utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import Sigvisa
from sigvisa.signals.io import *

from sigvisa.graph.sigvisa_graph import SigvisaGraph

def setup_graph(event, sta, chan, band,
                tm_shape, tm_type, wm_family, wm_type, phases,
                init_run_name, init_iteration, fit_hz=5, nm_type="ar", absorb_n_phases=False, smoothing=0):

    """
    Set up the graph with the signal for a given training event.
    """

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    try:
        input_runid = get_fitting_runid(cursor, init_run_name, init_iteration, create_if_new = False)
        runids = (input_runid,)
    except RunNotFoundException:
        runids = ()

    sg = SigvisaGraph(template_model_type=tm_type, template_shape=tm_shape,
                      wiggle_model_type=wm_type, wiggle_family=wm_family,
                      phases=phases, nm_type = nm_type,
                      runids = runids,
                      absorb_n_phases=absorb_n_phases)

    wave = load_event_station_chan(event.evid, sta, chan, cursor=cursor, exclude_other_evs=True, phases=None if phases=="leb" else phases).filter("%s;env" % band)
    cursor.close()
    if smoothing > 0:
        wave = wave.filter('smooth_%d' % smoothing)
    if fit_hz != wave['srate']:
        wave = wave.filter('hz_%.2f' % fit_hz)
    sg.add_wave(wave=wave)
    sg.add_event(ev=event)
    #sg.fix_arrival_times()

    sg.uatemplate_rate = 1e-6

    return sg


def optimize_template_params(sigvisa_graph,  wn, tmpl_optim_params):

    nm1 = wn.prior_nm.copy(), wn.prior_nmid
    #means = wn.signal_component_means()
    #noise_mean = means['noise']
    #noise_var = wn.signal_component_means(return_stds_instead=True)['noise']**2
    #nm2 = sample_posterior_armodel_from_signal(noise_mean, noise_var, wn.prior_nm), None

    if sigvisa_graph.template_shape=="lin_polyexp":
        nphases = int(len(sigvisa_graph.template_nodes)/5) # HACK

        v1 = np.array([ 0., 2.0, -0.5, -4, -3] * nphases)
        v2 = np.array([ 2., -1, -1.5, -4, -1.5] * nphases)
        v3 = np.array([ -3., 2.3, 0.5, -4, -3] * nphases)
        init_vs = [(v1, nm1), (v2, nm1), (v3, nm1)]
    else:
        raise Exception("don't know how to initialize params for template shape %s" % sigvisa_graph.template_shape)

    best_vals = None
    best_prob = -np.inf
    best_nm = nm1
    for v, (nm, nmid) in init_vs:
        #wn.set_noise_model(nm, nmid)
        sigvisa_graph.set_all(values=v, node_list=sigvisa_graph.template_nodes)
        sigvisa_graph.optimize_templates(optim_params=tmpl_optim_params)
        v_result = sigvisa_graph.get_all(sigvisa_graph.template_nodes)
        v_p = sigvisa_graph.current_log_p()

        if v_p > best_prob:
            best_prob = v_p
            best_vals = v_result
            best_nm = (nm, nmid)

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

def compute_template_messages(sg, wn, logger):
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

            m = np.mean(vals)
            v = np.var(vals) + 1e-6 # avoid zero-variance problems

            pv = n._parent_values()
            prior_mean = n.model.predict(cond=pv)
            prior_var = n.model.variance(cond=pv)

            print "param", p, "prior", (prior_mean, prior_var), "posterior", (m, v)
            gp_messages[(eid, phase)][p] = multiply_scalar_gaussian(m, v, prior_mean, -prior_var)
            gp_messages[(eid, phase)][p+"_posterior"] = m, v

        best_lp_idx = np.argmax(lps)
        best_vals[(eid, phase)] = (tvals[best_lp_idx,1:], [tnodes[lbl][1] for lbl in labels[1:]])

    return gp_messages, best_vals

def compute_wavelet_messages(sg, wn):

    gp_messages = dict()
    gp_posteriors = dict()

    wn._parent_values() # update the SSM to include current templates
    marginals = wn.tssm.all_filtered_cssm_coef_marginals(wn.get_value().data)

    for i, (eid, phase, scale, sidx, npts, component_type) in enumerate(wn.tssm_components):

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

def run_fit(sigvisa_graph, fit_hz, tmpl_optim_params, output_run_name, output_iteration, steps):

    s = Sigvisa()
    cursor = Sigvisa().dbconn.cursor()
    output_runid = get_fitting_runid(cursor, output_run_name, output_iteration, create_if_new = True)
    cursor.close()

    # initialize the MCMC by finding a good set of template params
    wn = sigvisa_graph.station_waves.values()[0][0]
    optimize_template_params(sigvisa_graph, wn, tmpl_optim_params)


    # run MCMC to sample from the posterior on template params
    st = time.time()

    logger = MCMCLogger(run_dir="scratch/mcmc_fit_%s/" % (str(uuid.uuid4())), write_template_vals=True, dump_interval=10, transient=True)
    run_open_world_MH(sigvisa_graph, steps=steps, enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, logger=logger)
    et = time.time()

    # compute template posterior, and set the graph state to the best template params
    messages, best_tmvals = compute_template_messages(sigvisa_graph, wn, logger)
    for (vals, nodes) in best_tmvals.values():
        for (v, n) in zip(vals, nodes):
            n.set_value(v)

    # compute wavelet posterior
    wavelet_messages, wavelet_posteriors = compute_wavelet_messages(sigvisa_graph, wn)

    for k, v in wavelet_messages.items():
        messages[k][sigvisa_graph.wiggle_family] = v
        messages[k][sigvisa_graph.wiggle_family + "_posterior"] = wavelet_posteriors[k]

    tops=repr(tmpl_optim_params)[1:-1]
    fitids = save_template_params(sigvisa_graph,
                                  tmpl_optim_param_str = "mcmc",
                                  wiggle_optim_param_str = "mcmc",
                                  elapsed=et - st, hz=fit_hz,
                                  runid=output_runid,
                                  messages= messages)
    return fitids[0]

def save_template_params(sg, tmpl_optim_param_str,
                         wiggle_optim_param_str,
                         hz, elapsed,
                         runid, messages):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    fitids = []
    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    run_wiggle_dir = os.path.join(wiggle_dir, "runid_" + str(runid))
    ensure_dir_exists(run_wiggle_dir)

    for wave_node in sg.leaf_nodes:
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
        event = get_event(evid=wave['evid'])

        pv = wave_node._parent_values()
        arrivals = update_arrivals(pv)

        slon, slat, _, _, _, _, _ = s.earthmodel.site_info(sta, st)
        distance = geog.dist_km((event.lon, event.lat), (slon, slat))
        azimuth = geog.azimuth((slon, slat), (event.lon, event.lat))

        tmpl_optim_param_str = tmpl_optim_param_str.replace("'", "''")
        wiggle_optim_param_str = wiggle_optim_param_str.replace("'", "''")
        optim_log = wiggle_optim_param_str.replace("\n", "\\\\n")

        if wave_node.nmid is None:
            nm_fname = model_path(sta, chan, wave['filter_str'], wave_node.srate, wave_node.nm.p, window_stime=wave_node.st, model_type="ar") + "_inferred"
            full_fname = os.path.join(os.getenv('SIGVISA_HOME'), nm_fname)
            ensure_dir_exists(os.path.dirname(full_fname))
            wave_node.nm.dump_to_file(full_fname)
            wave_node.nmid = wave_node.nm.save_to_db(dbconn=s.dbconn, sta=wave_node.sta, chan=wave_node.chan,
                                                     band=wave_node.band, hz=wave_node.srate, env=True, smooth=smooth,
                                                     window_stime=wave_node.st, window_len=wave_node.et-wave_node.st,
                                                     fname=nm_fname, hour=-1)
            print "saving inferred noise model as nmid", wave_node.nmid

        sql_query = "INSERT INTO sigvisa_coda_fit (runid, evid, sta, chan, band, smooth, tmpl_optim_method, wiggle_optim_method, optim_log, iid, stime, etime, hz, acost, dist, azi, timestamp, elapsed, nmid) values (%d, %d, '%s', '%s', '%s', '%d', '%s', '%s', '%s', %d, %f, %f, %f, %f, %f, %f, %f, %f, %d)" % (runid, event.evid, sta, chan, band, smooth, tmpl_optim_param_str, wiggle_optim_param_str, sg.optim_log, 1 if wave_node.nm_type != 'ar' else 0, st, et, hz, sg.current_log_p(), distance, azimuth, time.time(), elapsed, wave_node.nmid)

        fitid = execute_and_return_id(s.dbconn, sql_query, "fitid")

        for (eid, phase) in arrivals:

            fit_param_nodes = sg.get_template_nodes(eid=eid, phase=phase, chan=wave_node.chan, band=wave_node.band, sta=wave_node.sta)
            fit_params = dict([(p, n.get_value(k)) for (p,(k, n)) in fit_param_nodes.iteritems()])


            tg = sg.template_generator(phase)

            peak_decay = fit_params['peak_decay'] if 'peak_decay' in fit_params else 0.0

            if eid > 0:
                phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay, amp_transfer, peak_decay, wiggle_family) values (%d, '%s', '%s', %f, %f, %f, %f, %f, %f, '%s')" % (
                    fitid, phase, tg.model_name(), fit_params['arrival_time'], fit_params['peak_offset'], fit_params['coda_height'], fit_params['coda_decay'], fit_params['amp_transfer'], peak_decay, sg.wiggle_family)
            else:
                phase_insert_query = "insert into sigvisa_coda_fit_phase (fitid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay, peak_decay, wiggle_family) values (%d, '%s', '%s', %f, %f, %f, %f, %f, '%s')" % (
                    fitid, phase, tg.model_name(), fit_params['arrival_time'], fit_params['peak_offset'], fit_params['coda_height'], fit_params['coda_decay'], peak_decay, sg.wiggle_family)

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
    parser.add_option("--wiggle_family", dest="wiggle_family", default="dummy", type="str", help="")
    parser.add_option("--wiggle_model", dest="wiggle_model", default="dummy", type="str", help="")
    parser.add_option("--phases", dest="phases", default="leb", type="str", help="")
    parser.add_option("--band", dest="band", default="freq_2.0_3.0", type="str", help="")
    parser.add_option("--chan", dest="chan", default="auto", type="str", help="")
    parser.add_option("--smooth", dest="smooth", default=0, type=int, help="perform the given level of smoothing")
    parser.add_option("--steps", dest="steps", default=500, type=int, help="number of MCMC steps to run (500)")
    parser.add_option("--hz", dest="hz", default=5.0, type="float", help="sampling rate at which to fit the template")
    parser.add_option("--nm_type", dest="nm_type", default="ar", type="str",
                      help="type of noise model to use (ar)")
    parser.add_option("--seed", dest="seed", default=0, type="int",
                      help="ranom seed for MCMC (0)")
    parser.add_option("--absorb_n_phases", dest="absorb_n_phases", default=False, action="store_true", help="")


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

    if options.template_model == "hack26":
        init_run_name = "multiphase26_linpolyexp"
        init_iteration = 1
        template_model = {'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb', 'peak_decay': 'param_linear_distmb'}

    sigvisa_graph = setup_graph(event=ev, sta=options.sta, chan=options.chan, band=options.band,
                                tm_shape=options.template_shape, tm_type=template_model,
                                wm_family=options.wiggle_family, wm_type=options.wiggle_model,
                                phases=phases,
                                fit_hz=options.hz, nm_type=options.nm_type,
                                init_run_name = init_run_name, init_iteration = init_iteration,
                                absorb_n_phases=options.absorb_n_phases, smoothing=options.smooth)

    if options.seed >= 0:
        np.random.seed(options.seed)

    fitid = run_fit(sigvisa_graph,  fit_hz = options.hz,
                    tmpl_optim_params=construct_optim_params(options.tmpl_optim_params),
                    output_run_name = options.run_name, output_iteration=options.run_iteration, steps=options.steps)


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
