import numpy as np
from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, do_coarse_to_fine, initialize_from, do_inference
from sigvisa.infer.correlations.event_proposal import correlation_location_proposal #, generate_historical_db
from sigvisa.infer.correlations.ar_correlation_model import estimate_ar, ar_advantage, iid_advantage

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.mcmc_logger import MCMCLogger
import os, sys, traceback
import cPickle as pickle

stas = ['ASAR', 'KURK', 'MKAR', 'SONM', 'BVAR', 'FITZ', 'CTA', 'CMAR', 'WRA', 'ZALV', 'MJAR', 'AKTO', 'INK']

doublet = 5334939


doublet_ev = get_event(evid=doublet)
rs = EventRunSpec(evids=[doublet,], stas=["MKAR", "ASAR", "FITZ"], runids=(1,), disable_conflict_checking=False)

ms1 = ModelSpec(template_model_type="gp_lld", wiggle_family="db4_2.0_3_20.0", wiggle_model_type="gp_lld", max_hz=10.0, raw_signals=True)

sg = rs.build_sg(ms1)
sg.event_end_time = doublet_ev.time + 200
sg.event_start_time = doublet_ev.time - 200
sg.correlation_proposal_stas=["MK31", ]

np.random.seed(1)

logger = MCMCLogger( write_template_vals=False, dump_interval=10, print_interval=10, write_gp_hparams=False)

run_open_world_MH(sg, logger=logger, steps=1000,
                      enable_event_openworld=True,
                      enable_event_moves=True,
                      enable_template_openworld=False,
                      enable_template_moves=True)
