import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph, MAX_TRAVEL_TIME
from sigvisa.graph.region import Region
from sigvisa.source.event import Event
from sigvisa.treegp.gp import GPCov
from sigvisa.utils.geog import dist_km

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser

from sigvisa.infer.initialize_joint_alignments import align_atimes, detect_outlier_fits, prune_empty_wns
from sigvisa.learn.jointgp_iterative_init import jointgp_iterative_align_init


def sort_phases(phases):
    canonical_order=["P", "pP", "Pn", "Pg", "S", "Sn", "Lg"]
    return sorted(phases, key = lambda phase : - canonical_order.index(phase))

def sigvisa_fit_jointgp(stas, evs, runids,  runids_raw, phases, 
                        max_hz=None, resume_from=None, 
                        init_buffer_len=2.0,
                        n_random_inits=200,
                        init_from_predicted_times=False,
                        prune_outlier_fits=False,
                        iterative_align_init=False,
                        coarse_file=None,
                        fine_file=None,
                        infer_evs=True,
                        **kwargs):

    try:
        jgp_runid = runids_raw[0]
    except:
        jgp_runid = None

    if len(stas) > 1:
        print "warning, avoiding distance-based ttr variance hack because we have multiple stas"
        jgp_priors = None
    else:
        jgp_priors = hack_gp_hparam_priors(stas[0], evs)

    obs_stds = None
    if infer_evs:
        obs_stds = {"time": 2.0, "mb": 0.2}

    rs = EventRunSpec(evs=evs, stas=stas, pre_s=50.0, 
                      force_event_wn_matching=True,
                      disable_conflict_checking=True, 
                      observe_ev_stds = obs_stds,
                      seed=0, sample_init_templates=True)

    ms1 = ModelSpec(template_model_type="gp_joint", wiggle_family="iid", 
                    min_mb=1.0,
                    runids=runids, 
                    hack_param_constraint=True,
                    hack_ttr_max=8.0,
                    jointgp_param_run_init=jgp_runid,
                    jointgp_hparam_prior=jgp_priors,
                    phases=phases,
                    skip_levels=0,
                    uatemplate_rate=1e-6,
                    raw_signals=False, max_hz=2.0,  **kwargs)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], enable_phase_openworld=False, steps=500, special_mb_moves=infer_evs, special_time_moves=infer_evs)

    ms4 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", 
                    min_mb=1.0,
                    wiggle_model_type="gp_joint", raw_signals=True, max_hz=max_hz,
                    runids=runids_raw, 
                    hack_param_constraint=True,
                    phases=phases,
                    jointgp_param_run_init=jgp_runid,
                    jointgp_hparam_prior=jgp_priors,
                    hack_ttr_max=8.0,
                    uatemplate_rate=1e-8,
                    skip_levels=0,
                    **kwargs)
    ms4.add_inference_round(enable_event_moves=False, 
                            enable_event_openworld=False, 
                            enable_template_openworld=False, 
                            enable_template_moves=True, 
                            special_mb_moves=False, 
                            enable_phase_openworld=False,
                            fix_atimes=True, steps=10)
    ms4.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, enable_phase_openworld=False, special_mb_moves=infer_evs, special_time_moves=infer_evs, steps=300)


    rundir = None

    if coarse_file is not None:
        with open(coarse_file, 'rb') as f:
            sg1 = pickle.load(f)
        rundir = os.path.dirname(os.path.dirname(coarse_file))
    elif fine_file is None:
        sg1 = rs.build_sg(ms1)
        initialize_sg(sg1, ms1, rs)
        rundir = do_inference(sg1, ms1, rs,
                              model_switch_lp_threshold=None,
                              run_dir=rundir)
    rundir = rundir + ".1.1.1"

    if fine_file is not None:
        with open(fine_file, 'rb') as f:
            sg4 = pickle.load(f)
    else:
        sg4 = rs.build_sg(ms4)
        initialize_from(sg4, ms4, sg1, ms1)

    if prune_outlier_fits:
        outlier_eids = detect_outlier_fits(sg1)
        print "removing outliers", outlier_eids
        for eid in outlier_eids:
            sg4.remove_event(eid)
        prune_empty_wns(sg4)
        

    if n_random_inits > 0:
        for sta in stas:
            for phase in sort_phases(phases):

                try:
                    align_atimes(sg4, sta, phase, buffer_len_s=init_buffer_len, 
                                 patch_len_s=20.0, n_random_inits=n_random_inits,
                                 center_at_current_atime=not init_from_predicted_times)
                except Exception as e:
                    print e
                    continue
    
    if iterative_align_init:
        sg4 = jointgp_iterative_align_init(sg4, base_run_dir=rundir)
    
    sg4.current_log_p()
    do_inference(sg4, ms4, rs, dump_interval_s=10, 
                 print_interval_s=10, 
                 run_dir=rundir,
                 model_switch_lp_threshold=None)


def hack_gp_hparam_priors(sta, evs):
    from sigvisa.models.distributions import LogNormal
    from sigvisa.models.spatial_regression.SparseGP import default_jgp_hparam_priors
    hparam_priors = default_jgp_hparam_priors()

    s = Sigvisa()
    slon, slat = s.earthmodel.site_info(sta, 0)[:2]
    dists = [dist_km((slon, slat), (ev.lon, ev.lat)) for ev in evs]
    print "event-station distances min %.2fkm, mean %.2fkm, max %.2fkm" % (np.min(dists), np.mean(dists), np.max(dists))
    mean_dist = np.mean(dists)

    if mean_dist < 100:
        ttr_signal_var = 1.0
    elif mean_dist < 300:
        ttr_signal_var = 3.0
    elif mean_dist < 600:
        ttr_signal_var = 10.0
    elif mean_dist < 1000:
        ttr_signal_var = 16.0
    elif mean_dist < 1500:
        ttr_signal_var = 24.0
    else:
        ttr_signal_var = 30.0
    print "using traveltime residual variance prior centered at stddev=%.2fs" % (np.sqrt(ttr_signal_var))

    hparam_priors["tt_residual"]["signal_var"] = LogNormal(np.log(ttr_signal_var), 0.5)

    return hparam_priors

def get_evs(min_lon, max_lon, min_lat, max_lat, min_time, max_time, evtype="isc", precision=None):
    if evtype=="isc" and precision is not None:
        prec_cond = "and smaj < %f" % precision
    else:
        prec_cond = ""

    sql = "select evid, lon, lat, depth, time, mb from %s_origin where lon between %f and %f and lat between %f and %f and time between %f and %f %s" % (evtype, min_lon, max_lon, min_lat, max_lat, min_time, max_time, prec_cond)

    print sql

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    cursor.execute(sql)
    r = cursor.fetchall()
    cursor.close()

    evs = []
    for (evid, lon, lat, depth, time, mb ) in r:
        ev = Event(lon=lon, lat=lat, depth=depth, time=time, mb=mb, evid=evid, autoload=False)
        evs.append(ev)

        
    # filter for near-duplicates that screw up fitting
    filtered_evs = []
    for ev1 in evs:

        duplicated = False
        query = "select evid from %s_origin where lon between %f and %f and lat between %f and %f and time between %f and %f" % (ev.lon-1, ev.lon+1, ev.lat-1, ev.lat+1, ev.time-60, ev.time+60)
        r = s.sql(query)
        if len(r) == 1:
            filtered_evs.append(ev1)
        else:
            print "skipping event", ev, "based on duplicates", r

    return filtered_evs

def main():
    parser = OptionParser()

    parser.add_option("--sta", dest="sta", default=None, type=str)
    parser.add_option("--runid", dest="runid", default=None, type=int)
    parser.add_option("--runid_raw", dest="runid_raw", default=None, type=int)
    parser.add_option("--phases", dest="phases", default=None, type=str)
    parser.add_option("--max_hz", dest="max_hz", default=10.0, type=str)
    parser.add_option("--max_lon", dest="max_lon", default=None, type=float)
    parser.add_option("--min_lon", dest="min_lon", default=None, type=float)
    parser.add_option("--max_lat", dest="max_lat", default=None, type=float)
    parser.add_option("--min_lat", dest="min_lat", default=None, type=float)
    parser.add_option("--max_time", dest="max_time", default=None, type=float)
    parser.add_option("--min_time", dest="min_time", default=None, type=float)
    parser.add_option("--evidfile", dest="evidfile", default=None, type=str)
    parser.add_option("--subsample_evs", dest="subsample_evs", default=None, type=int)    
    parser.add_option("--evtype", dest="evtype", default="isc", type=str)
    parser.add_option("--precision", dest="precision", default=None, type=float)
    parser.add_option("--coarse_file", dest="coarse_file", default=None, type=str)
    parser.add_option("--fine_file", dest="fine_file", default=None, type=str)
    parser.add_option("--prune_outlier_fits", dest="prune_outlier_fits", default=False, action="store_true")
    parser.add_option("--iterative_align_init", dest="iterative_align_init", default=False, action="store_true")
    parser.add_option("--init_buffer_len", dest="init_buffer_len", default=2.0, type=float)
    parser.add_option("--init_from_predicted_times", dest="init_from_predicted_times", default=False, action="store_true")
    parser.add_option("--n_random_inits", dest="n_random_inits", default=200, type=int)
    

    (options, args) = parser.parse_args()

    if options.phases is not None:
        phases = options.phases.split(",")
    else:
        phases="P,Pg,pP,PcP,S,ScP,Lg,Rg,PKP,PKPab,PKPbc,PKKPbc".split(",")

    if options.runid is not None:
        runids = (options.runid,)
    else:
        runids = ()
    if options.runid_raw is not None:
        runids_raw = (options.runid_raw,)
    else:
        runids_raw = ()

    if options.evidfile is not None:
        evids = np.loadtxt(options.evidfile)
        evs = [get_event(evid=int(evid)) for evid in evids]
    else:
        evs = get_evs(options.min_lon, options.max_lon, 
                      options.min_lat, options.max_lat, 
                      options.min_time, options.max_time,
                      evtype=options.evtype,
                      precision=options.precision)

    if options.subsample_evs is not None and len(evs) > options.subsample_evs:
        sorted_evs = sorted(evs, key = lambda ev : ev.mb)
        evs = sorted_evs[-options.subsample_evs:]
        
    # HACK for weird llnl signal amplitudes
    from sigvisa.models.distributions import Uniform, Poisson, Gaussian, Exponential, TruncatedGaussian, LogNormal,InvGamma, Beta, Laplacian, Bernoulli
    dummyPrior = dummyPriorModel = {
    "tt_residual": Laplacian(center=0.0, scale=3.0),
    "amp_transfer": Gaussian(mean=8.0, std=10.0),
    "peak_offset": TruncatedGaussian(mean=-0.5, std=1.0, b=4.0),
    "mult_wiggle_std": Beta(4.0, 1.0),
    "coda_decay": Gaussian(mean=0.0, std=1.0),
    "peak_decay": Gaussian(mean=0.0, std=1.0)
    }


    # HACK
    #evs = evs[:30]



    sigvisa_fit_jointgp([options.sta], evs, 
                        runids=runids, 
                        runids_raw=runids_raw, 
                        max_hz=options.max_hz,
                        phases=phases,
                        dummy_prior=dummyPrior,
                        dummy_fallback=True,
                        init_buffer_len=options.init_buffer_len,
                        n_random_inits = options.n_random_inits,
                        init_from_predicted_times=options.init_from_predicted_times,
                        prune_outlier_fits=options.prune_outlier_fits,
                        iterative_align_init=options.iterative_align_init,
                        coarse_file=options.coarse_file,
                        fine_file=options.fine_file)


if __name__ == "__main__":
    main()
