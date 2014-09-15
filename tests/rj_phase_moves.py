


import numpy as np
import sys
import os
import traceback
import pickle
import copy
import time

from collections import defaultdict
from optparse import OptionParser


from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.infer.mcmc_logger import MCMCLogger
from sigvisa.source.event import Event
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.utils.fileutils import clear_directory
from sigvisa.signals.io import fetch_waveform

def rj_phase_test():

    tm_type_str = "tt_residual:constant_laplacian,peak_offset:param_linear_mb,amp_transfer:param_sin1,coda_decay:param_linear_distmb,peak_decay:param_linear_distmb"
    tm_types = {}
    for p in tm_type_str.split(','):
        (param, model_type) = p.strip().split(':')
        tm_types[param] = model_type


    sg = SigvisaGraph(template_model_type="dummyPrior", template_shape="lin_polyexp",
                      wiggle_model_type="dummy", wiggle_family="dummy",
                      phases=["Pn",], nm_type = "ar",
                      runid=26)
    sg.uatemplate_rate=10

    stas = ["MK31",]
    stime = 1238889600
    etime = stime + 2600

    s = Sigvisa()
    wns = []
    for sta in stas:
        chan = s.default_vertical_channel[sta]
        wave = fetch_waveform(sta, chan, stime, etime)
        wave = wave.filter("freq_2.0_3.0;env;smooth_15;hz_1.0")
        wn = sg.add_wave(wave)
        wns.append(wn)


    ev = Event(mb=4.5, depth=34.99, lon=82.29, lat=25.8, time=stime, autoload=False)
    sg.add_event(ev)

    for wn in wns:
        wn.unfix_value()
        wn.parent_sample()
        wn.fix_value()

    run_dir=os.path.join(s.homedir, "logs", "mcmc", "rj_phase")
    clear_directory(run_dir)
    logger = MCMCLogger(run_dir=run_dir, write_template_vals=True, dump_interval=50)
    np.random.seed(4)

    run_open_world_MH(sg, steps=200,
                      logger=logger,
                      enable_template_openworld=False,
                      enable_template_moves=True,
                      enable_event_moves=True,
                      enable_event_openworld=False)




if __name__ == "__main__":
    rj_phase_test()
