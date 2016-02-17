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

#stas = "NEW,PDAR,ELK,NVAR,ANMO,TXAR,PFO,YKA,ULM,ILAR".split(",")
stas = "NEW,PDAR,NVAR,ANMO,TXAR,PFO,YKA,ULM,ILAR".split(",")

stas = "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(",")

region_lon = (-126, -100)
region_lat = (32, 49)

#stimes = {5411098: 1244065600,
#          5381051: 1242617900,
#          5356788: 1241373200,
#          5336072: 1240250500}

#training_stime = 1238889600
training_stime =  1167634400
#region_stime = stimes[target_evid]
#region_etime = region_stime + 7200

def reference_evids(region, bulletin="isc"):
    relevant_stime = region.stime - MAX_TRAVEL_TIME
    relevant_etime = region.etime

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    sql_query = "select evid from %s_origin where (time between %.1f and %.1f) and ((lat between %.1f and %.1f) and (lon between %.1f and %.1f) ) " % (bulletin, relevant_stime, relevant_etime, region.bottom_lat, region.top_lat, region.left_lon, region.right_lon)
    print sql_query
    cursor.execute(sql_query)
    evids = [evid[0] for evid in cursor.fetchall()]
    cursor.close()

    return evids
    

def main(hour=0.0, len_hours=2.0, runid=37, hz=2.0, tmpl_steps=500, ev_steps=1000, resume_from=None, deserialize=None, uatemplate_rate=4e-4, raw_signals=False, bands=["freq_0.8_4.5"], fix_outside=True, phases=("P"), target_evid=-1, stime=None, etime=None, reference_bulletin="isc"):

    region = Region(lons=region_lon, lats=region_lat, times=(stime, etime))

    evids = reference_evids(region, reference_bulletin)

    rs = EventRunSpec(sites=stas, runids=(runid,), evids=evids, pre_s=40.0, 
                      post_s=60, disable_conflict_checking=False)

    ms1 = ModelSpec(template_model_type="param",
                    wiggle_family="iid",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    bands=bands,
                    inference_region=region,
                    dummy_fallback=True,
                    raw_signals=raw_signals,
                    hack_param_constraint=True,
                    vert_only=True)
    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=tmpl_steps)


    sg = rs.build_sg(ms1)
    initialize_sg(sg, ms1, rs)
    do_inference(sg, ms1, rs, dump_interval_s=10, print_interval_s=10, model_switch_lp_threshold=None)


if __name__ == "__main__":

    parser = OptionParser()


    parser.add_option("--raw", dest="raw", default=False, action="store_true",
                      help="use raw signals instead of envelopes")
    parser.add_option("--hz", dest="hz", default=2.0, type=float,
                      help="downsample signals to this rate")
    parser.add_option("--reference_bulletin", dest="reference_bulletin", default="isc", type=str,
                      help="bulletin to load reference events")
    parser.add_option("--bands", dest="bands", default="freq_0.8_4.5", type=str,
                      help="comma-separated frequency bands")
    parser.add_option("--phases", dest="phases", default="P,Pg,pP,PcP,S,ScP,Lg,Rg,PKP,PKPab,PKPbc,PKKPbc", type=str,
                      help="comma-separated phases")
    parser.add_option("--fix_outside_templates", dest="fix_outside_templates", default=False, action="store_true",
                      help="don't do inference over templates of events outside the region")
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

    (options, args) = parser.parse_args()

    bands = options.bands.split(",")
    phases = options.phases.split(",")

    main(hour=options.hour, len_hours=options.len_hours, resume_from=options.resume_from, runid=options.runid, tmpl_steps=options.tmpl_steps, ev_steps=options.ev_steps, deserialize=options.deserialize, uatemplate_rate=options.uatemplate_rate, raw_signals=options.raw, hz=options.hz, bands=bands, fix_outside=options.fix_outside_templates, phases=phases, target_evid=options.target_evid, stime=options.stime, etime=options.etime, reference_bulletin=options.reference_bulletin)
