import numpy as np

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.graph.sigvisa_graph import SigvisaGraph, MAX_TRAVEL_TIME
from sigvisa.graph.region import Region
from sigvisa.treegp.gp import GPCov

import os, sys, traceback
import cPickle as pickle
from optparse import OptionParser



default_stas = "AAK,AFI,AKASG,AKTO,ANMO,ARCES,ASAR,ASF,ATAH,ATD,ATTU,BBB,BBTS,BORG,BOSA,BRTR,BVAR,CFAA,CMAR,CMIG,CPUP,CTA,DAV,DBIC,DLBC,DZM,EIL,ELK,ESDC,FINES,FITZ,FRB,GERES,GNI,GUMO,HFS,HNR,ILAR,INK,JCJ,JHJ,JKA,JNU,JOW,JTS,KAPI,KBZ,KDAK,KEST,KMBO,KSRS,KURK,LBTB,LPAZ,LPIG,LSZ,MAW,MBAR,MDT,MJAR,MKAR,MLR,MMAI,NEW,NNA,NOA,NVAR,NWAO,OPO,PDAR,PETK,PFO,PLCA,PMG,PMSA,PPT,PSI,QSPA,RAO,RAR,RCBR,RES,ROSC,RPN,RPZ,SADO,SCHQ,SFJD,SIV,SJG,SNAA,SONM,SPITS,STKA,SUR,TEIG,TGY,TKL,TORD,TSUM,TXAR,ULM,URZ,USHA,USRK,VNDA,VRAC,WRA,YBH,YKA,ZALV".split(",")


region_lon = (-180, 180)
region_lat = (-90, 90)


def main(stas=None, hour=0.0, len_hours=2.0, runid=37, hz=10.0, tmpl_steps=500, ev_steps=1000, resume_from=None, deserialize=None, uatemplate_rate=4e-4, raw_signals=False, bands=["freq_0.8_4.5"], fix_outside=True, phases=("P"), target_evid=-1, stime=None, etime=None, hack_constraint=True, seed=3, use_hough=False):

    np.random.seed(seed)

    if target_evid > 0:
        rs = EventRunSpec(sites=stas, evids=(target_evid,), seed=seed, 
                          force_event_wn_matching=False)
        ev = get_event(target_evid)
        stime = ev.time - 150
        etime = ev.time + 200
    else:
        rs = TimeRangeRunSpec(sites=stas, start_time=stime, end_time=etime, 
                              initialize_events="leb", seed=seed)
        

    min_mb = 2.5
    runids=(runid,)

    region = Region(lons=region_lon, lats=region_lat, 
                    times=(stime, etime),
                    rate_bulletin="leb", 
                    min_mb=min_mb,
                    rate_train_start=1238889600,
                    rate_train_end=1245456000)


    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    wiggle_model_type="dummy",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    bands=bands,
                    runids=runids,
                    inference_region=region,
                    min_mb=min_mb,
                    skip_levels=0,
                    dummy_fallback=True,
                    raw_signals=raw_signals,
                    hack_param_constraint=hack_constraint,
                    vert_only=True)


    if resume_from is None:
        sg = rs.build_sg(ms1)
        ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc', 'ev_lsqr'], steps=30, fix_outside_templates=fix_outside, propose_correlation=False, propose_hough=use_hough)
    else:
        with open(resume_from, 'rb') as f:
            sg = pickle.load(f)
        sg.current_log_p()


    for wns in sg.station_waves.values():
        for wn in wns:
            print "setting iid hack at", wn.sta
            wn.hack_wavelets_as_iid = True

    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, 
                            enable_template_openworld=True, enable_template_moves=True, 
                            disable_moves=['atime_xc', 'ev_lsqr'], steps=ev_steps, 
                            fix_outside_templates=fix_outside, propose_correlation=False, 
                            propose_hough=use_hough)

    do_inference(sg, ms1, rs, dump_interval_s=10, print_interval_s=10, model_switch_lp_threshold=None, dump_proposals=False, max_dumps=1)


if __name__ == "__main__":

    parser = OptionParser()


    parser.add_option("--raw", dest="raw", default=False, action="store_true",
                      help="use raw signals instead of envelopes")
    parser.add_option("--hz", dest="hz", default=10.0, type=float,
                      help="downsample signals to this rate")
    parser.add_option("--bands", dest="bands", default="freq_0.8_4.5", type=str,
                      help="comma-separated frequency bands")
    parser.add_option("--phases", dest="phases", default="P,Pg,pP,PcP,S,ScP,Lg,Rg,PKP,PKPab,PKPbc,PKKPbc", type=str,
                      help="comma-separated phases")
    parser.add_option("--stas", dest="stas", default=None, type=str,
                      help="comma-separated stas")
    parser.add_option("--fix_outside_templates", dest="fix_outside_templates", default=False, action="store_true",
                      help="don't do inference over templates of events outside the region")
    parser.add_option("--no_hack_constraint", dest="no_hack_constraint", default=False, action="store_true",
                      help="don't force template params to reasonable values")
    parser.add_option("--target_evid", dest="target_evid", default=-1, type=int,
                      help="evid in training set to attempt to infer. overrides --hour.")
    parser.add_option("--hour", dest="hour", default=0.0, type=float,
                      help="start time, relative to training dataset")
    parser.add_option("--len_hours", dest="len_hours", default=2.0, type=float,
                      help="length of signal to use")
    parser.add_option("--stime", dest="stime", default=None, type=float,
                      help="")
    parser.add_option("--etime", dest="etime", default=None, type=float,
                      help="")
    parser.add_option("--tmpl_steps", dest="tmpl_steps", default=500, type=int,
                      help="steps to take in tmpl-only inference")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=4e-4, type=float,
                      help="")
    parser.add_option("--ev_steps", dest="ev_steps", default=1000, type=int,
                      help="steps to take in full inference ")
    parser.add_option("--resume_from", dest="resume_from", default=None, type=str,
                      help="saved sg state to initialize inference")
    parser.add_option("--deserialize", dest="deserialize", default=None, type=str,
                      help="serialized (tgz) sg state to initialize inference")
    parser.add_option("--runid", dest="runid", default=1, type=int,
                      help="runid for models to load")
    parser.add_option("--seed", dest="seed", default=1, type=int,
                      help="")
    parser.add_option("--use_hough", dest="use_hough", default=False, action="store_true",
                      help="enable hough transform proposals for de novo events")

    (options, args) = parser.parse_args()

    bands = options.bands.split(",")
    phases = options.phases.split(",")

    stas = options.stas
    if stas is None:
        stas = default_stas
    else:
        stas = stas.split(",")
        

    main(stas=stas, hour=options.hour, len_hours=options.len_hours, resume_from=options.resume_from, runid=options.runid, tmpl_steps=options.tmpl_steps, ev_steps=options.ev_steps, deserialize=options.deserialize, uatemplate_rate=options.uatemplate_rate, raw_signals=options.raw, hz=options.hz, bands=bands, fix_outside=options.fix_outside_templates, phases=phases, target_evid=options.target_evid, stime=options.stime, etime=options.etime, hack_constraint=not options.no_hack_constraint, seed=options.seed, use_hough=options.use_hough)
