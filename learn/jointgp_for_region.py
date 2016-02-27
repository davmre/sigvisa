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

def sigvisa_fit_jointgp(stas, evs, runids,  runids_raw, max_hz=None, **kwargs):

    #isc_evs = [load_isc(evid) for evid in evids]
    #main_ev = isc_evs[0]
    #main_ev = get_event(evid=5335822)
    #isc_evs = [ev for ev in isc_evs if ev is not None and dist_km((main_ev.lon, main_ev.lat), (ev.lon, ev.lat)) < 15]

    jgp_runid = runids_raw[0]

    rs = EventRunSpec(evs=evs, stas=stas, pre_s=50.0, 
                      disable_conflict_checking=True)

    ms1 = ModelSpec(template_model_type="param", wiggle_family="iid", 
                    min_mb=1.0,
                    runids=runids, 
                    raw_signals=False, max_hz=2.0, **kwargs)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'], enable_phase_openworld=True, steps=150)

    ms2 = ModelSpec(template_model_type="gp_joint", wiggle_family="iid", wiggle_model_type="dummy", 
                    min_mb=1.0,
                    runids=runids_raw, 
                    jointgp_param_run_init=jgp_runid, raw_signals=True, max_hz=max_hz, **kwargs)
    ms2.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, disable_moves=['atime_xc'], enable_phase_openworld=False, steps=50)

    ms3 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", 
                    min_mb=1.0,
                    wiggle_model_type="dummy", raw_signals=True, max_hz=max_hz, 
                    runids=runids_raw, 
                    jointgp_param_run_init=jgp_runid, **kwargs)
    ms3.add_inference_round(enable_event_moves=False, enable_event_openworld=False, 
                            enable_template_openworld=False, enable_template_moves=True, 
                            enable_phase_openworld=False, steps=50)

    ms4 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", 
                    min_mb=1.0,
                    wiggle_model_type="gp_joint", raw_signals=True, max_hz=max_hz,
                    runids=runids_raw, 
                    jointgp_param_run_init=jgp_runid,
                    **kwargs)
    ms4.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, enable_phase_openworld=False)

    ms = [ms1, ms2, ms3, ms4]
    do_coarse_to_fine(ms, rs, max_steps_intermediate=None, model_switch_lp_threshold=None, max_steps_final=300)


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
                        dummy_fallback=True)


if __name__ == "__main__":
    main()
