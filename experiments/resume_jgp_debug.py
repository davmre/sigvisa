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

def sigvisa_fit_jointgp(resume_from=None):

    with open(resume_from, 'rb') as f:
            sg = pickle.load(f)

    stas = sg.station_waves.keys()
    evs = [sg.get_event(eid) for eid in sg.evnodes.keys()]
    runids = sg.runids
    max_hz = sg.station_waves.values()[0][0].srate

    rs = EventRunSpec(evs=[], stas=stas, runids=runids, disable_conflict_checking=True)

    ms4 = ModelSpec(template_model_type="gp_joint", wiggle_family="db4_2.0_3_20.0", wiggle_model_type="gp_joint", raw_signals=True, max_hz=max_hz)
    ms4.add_inference_round(enable_event_moves=False, enable_event_openworld=False, enable_template_openworld=False, enable_template_moves=True, enable_phase_openworld=False, steps=10000)

    do_inference(sg, ms4, rs, dump_interval_s=10, print_interval_s=10, model_switch_lp_threshold=None)


def main():
    parser = OptionParser()

    parser.add_option("--sgfile", dest="sgfile", default=None, type=str)

    (options, args) = parser.parse_args()

    sigvisa_fit_jointgp(resume_from=options.sgfile)


if __name__ == "__main__":
    main()
