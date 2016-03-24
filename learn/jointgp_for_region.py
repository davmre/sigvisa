import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph, MAX_TRAVEL_TIME
from sigvisa.graph.region import Region
from sigvisa.source.event import Event
from sigvisa.treegp.gp import GPCov

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser

from sigvisa.infer.initialize_joint_alignments import align_atimes

def sort_phases(phases):
    canonical_order=["P", "pP", "Pn", "Pg", "S", "Sn", "Lg"]
    return sorted(phases, key = lambda phase : - canonical_order.index(phase))

def sigvisa_fit_jointgp(stas, evs, runids,  runids_raw, phases, 
                        max_hz=None, resume_from=None, 
                        init_buffer_len=8.0,
                        n_random_inits=200,
                        init_from_predicted_times=False,
                        **kwargs):

    try:
        jgp_runid = runids_raw[0]
    except:
        jgp_runid = None

    rs = EventRunSpec(evs=evs, stas=stas, pre_s=50.0, 
                      force_event_wn_matching=False,
                      disable_conflict_checking=True, 
                      seed=0, sample_init_templates=True)

    ms1 = ModelSpec(template_model_type="param", wiggle_family="iid", 
                    min_mb=1.0,
                    runids=runids, 
                    hack_param_constraint=True,
                    hack_ttr_max=8.0,
                    phases=phases,
                    skip_levels=0,
                    raw_signals=False, max_hz=2.0,  **kwargs)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, special_mb_moves=True, disable_moves=['atime_xc'], enable_phase_openworld=False, steps=500)

    ms4 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", 
                    min_mb=1.0,
                    wiggle_model_type="gp_joint", raw_signals=True, max_hz=max_hz,
                    runids=runids_raw, 
                    hack_param_constraint=True,
                    phases=phases,
                    jointgp_param_run_init=jgp_runid,
                    hack_ttr_max=8.0,
                    skip_levels=0,
                    **kwargs)
    ms4.add_inference_round(enable_event_moves=False, 
                            enable_event_openworld=False, 
                            enable_template_openworld=False, 
                            enable_template_moves=True, 
                            special_mb_moves=False, 
                            enable_phase_openworld=False,
                            fix_atimes=True, steps=50)
    ms4.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, special_mb_moves=False, enable_phase_openworld=False, steps=300)


    rundir = None

    print sort_phases(phases)

    sg1 = rs.build_sg(ms1)
    initialize_sg(sg1, ms1, rs)
    rundir = do_inference(sg1, ms1, rs,
                          model_switch_lp_threshold=None,
                          run_dir=rundir)
    rundir = rundir + ".1.1.1"

    sg4 = rs.build_sg(ms4)
    initialize_from(sg4, ms4, sg1, ms1)

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

    do_inference(sg4, ms4, rs, dump_interval_s=10, 
                 print_interval_s=10, 
                 run_dir=rundir,
                 model_switch_lp_threshold=None)

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

        
    return evs

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
    parser.add_option("--resume_from", dest="resume_from", default=None, type=str)
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
                        resume_from=options.resume_from)


if __name__ == "__main__":
    main()
