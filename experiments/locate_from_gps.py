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

stas = ["PDAR",]

region_lon = (-126, -100)
region_lat = (32, 49)

ev_stimes = {
651342: 1168024000.4,
}

def relevant_events(region):
    
    relevant_stime = region.stime - MAX_TRAVEL_TIME
    relevant_etime = region.etime

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    sql_query = "select evid from leb_origin where (time between %.1f and %.1f) and not ((lat between %.1f and %.1f) and (lon between %.1f and %.1f) and (time between %.1f and %.1f)) " % (relevant_stime, relevant_etime, region.bottom_lat, region.top_lat, region.left_lon, region.right_lon, region.stime, region.etime)
    print sql_query
    cursor.execute(sql_query)
    evids = [evid[0] for evid in cursor.fetchall()]
    evs = [get_event(evid=evid) for evid in evids]
    evs = [ev for ev in evs if ev.mb > 0]
    print "got relevant evs", evs
    return evs


def main(runid=6, hz=10.0, tmpl_steps=500, ev_steps=1000, uatemplate_rate=4e-4, raw_signals=True, bands=["freq_0.8_4.5"], fix_outside=True, phases=("P", "Pg", "S", "Lg"), ):
    
    len_hours = 0.5
    region_stime = ev_stimes[651342]
    region_etime = region_stime + len_hours * 3600

    rs = TimeRangeRunSpec(sites=stas, runids=(runid,), start_time=region_stime, end_time=region_etime)

    region_stime = rs.start_time
    region_etime = rs.end_time
    region = Region(lons=region_lon, lats=region_lat, times=(region_stime, region_etime))

    ms1 = ModelSpec(template_model_type="gp_lld",
                    wiggle_family="db4_2.0_3_20.0",
                    uatemplate_rate=uatemplate_rate,
                    max_hz=hz,
                    phases=phases,
                    bands=bands,
                    inference_region=region,
                    dummy_fallback=True,
                    raw_signals=raw_signals,
                    hack_param_constraint=True,
                    vert_only=True)


    sg = rs.build_sg(ms1)
    revs = relevant_events(region)
    for ev in revs:
        evnodes = sg.add_event(ev)
        eid = evnodes["lon"].eid
        sg.fix_event(eid, fix_templates=False)

    ms1.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=tmpl_steps)

    ms1.add_inference_round(enable_event_moves=True, enable_event_openworld=True, enable_template_openworld=True, enable_template_moves=True, disable_moves=['atime_xc'], steps=ev_steps, fix_outside_templates=fix_outside)

    do_inference(sg, ms1, rs, dump_interval_s=10, print_interval_s=10, model_switch_lp_threshold=None)

main()
