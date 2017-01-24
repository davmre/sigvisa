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
from sigvisa.models.distributions import Uniform, Poisson, Gaussian, Exponential, TruncatedGaussian, LogNormal, InvGamma, Beta, Laplacian, Bernoulli
from sigvisa.models.noise.noise_util import model_path
import sigvisa.utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import Sigvisa
from sigvisa.signals.io import *

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.load_sigvisa_graph import load_sg_from_db_fit

def load_blank_sg(fitid, uatemplate_rate=None): 


    sg = load_sg_from_db_fit(fitid)

    eids = set(sg.evnodes.keys())
    for eid in eids:
        sg.remove_event(eid)

    if uatemplate_rate is not None:
        sg.uatemplate_rate = uatemplate_rate

    sg.template_shape="lin_polyexp"

    return sg

def fit_uatemplates(sg, steps):

    logger = MCMCLogger(run_dir="scratch/uatemplate_fit_%s/" % (str(uuid.uuid4())), write_template_vals=False, dump_interval_s=1000, transient=True, serialize_interval_s=1000, print_interval_s=5)

    run_open_world_MH(sg, steps=steps, enable_event_moves=False, enable_event_openworld=False, enable_phase_openworld=False, enable_template_openworld=True, logger=logger, disable_moves=['atime_xc', 'constpeak_atime_xc'], tmpl_birth_rate=0.1)
    #logger.dump(sigvisa_graph)

def save_uatemplates(sg, fitid, uatemplate_rate, seed):

    s = Sigvisa()

    for tmid, tmnodes in sg.uatemplates.items():
        tmvals = dict([(p, n.get_value()) for (p, n) in tmnodes.items()])
        sql_query = "insert into sigvisa_uatemplate_tuning (fitid, arrival_time, peak_offset, coda_height, coda_decay, peak_decay, mult_wiggle_std, uatemplate_rate, seed) values (%d, %f, %f, %f, %f, %f, %f, %f, %d)" % (fitid, tmvals["arrival_time"], tmvals["peak_offset"], tmvals["coda_height"], tmvals["coda_decay"], tmvals["peak_decay"], tmvals["mult_wiggle_std"], uatemplate_rate, seed)

        uaid = execute_and_return_id(s.dbconn, sql_query, "uaid")

    if len(sg.uatemplates) == 0:
        sql_query = "insert into sigvisa_uatemplate_tuning (fitid, uatemplate_rate, seed) values (%d, %f, %d)" % (fitid, uatemplate_rate, seed)
        uaid = execute_and_return_id(s.dbconn, sql_query, "uaid")
        


def main():
    parser = OptionParser()

    parser.add_option("--fitid", dest="fitid", default=None, type="int", help="fitid for which to fit uatemplates")
    parser.add_option("--uatemplate_rate", dest="uatemplate_rate", default=None, type="float", help="if nonzero, allow uatemplate births to explain signal spikes")
    parser.add_option("--steps", dest="steps", default=500, type=int, help="number of MCMC steps to run (500)")
    parser.add_option("--seed", dest="seed", default=0, type="int",
                      help="random seed for MCMC (0)")

    parser.add_option("--nocheck", dest="nocheck", default=False, action="store_true", help="don't check to see if we've already fit this arrival in this run")


    (options, args) = parser.parse_args()

    s = Sigvisa()

    if not options.nocheck:
        sql_query = "select * from sigvisa_uatemplate_tuning where fitid=%d and seed=%d" % (options.fitid, options.seed)
        r = s.sql(sql_query)
        if len(r) > 0:
            print "not fitting because we already have fits for this fitid and seed"
            print r
            print "run with --nocheck flag to override"
            return

    if options.seed >= 0:
        np.random.seed(options.seed)

    sg = load_blank_sg(options.fitid, options.uatemplate_rate)
    fit_uatemplates(sg, options.steps)
    save_uatemplates(sg, options.fitid, options.uatemplate_rate, options.seed)

    print "fit id %d completed successfully." % options.fitid



if __name__ == "__main__":
    main()
