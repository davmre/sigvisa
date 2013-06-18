import os
import errno
import sys
import time
import traceback
import numpy as np
import numpy.ma as ma
import scipy

from sigvisa.database.dataset import *
from sigvisa.database.signal_data import *
from sigvisa.database import db
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.models.wiggles.wiggle import extract_phase_wiggle

import sigvisa.utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import Sigvisa
from sigvisa.signals.io import *

from sigvisa.graph.sigvisa_graph import SigvisaGraph

def setup_graph(event, sta, chan, band,
                tm_shape, tm_type, wm_family, wm_type, phases,
                output_run_name, output_iteration, fit_hz=5, nm_type="ar",
                init_run_name=None, init_iteration=None):
    sg = SigvisaGraph(template_model_type=tm_type, template_shape=tm_shape,
                      wiggle_model_type=wm_type, wiggle_family=wm_family,
                      phases=phases, nm_type = nm_type,
                      run_name = output_run_name, iteration = output_iteration)
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    wave = load_event_station_chan(event.evid, sta, chan, cursor=cursor).filter("%s;env" % band)
    cursor.close()
    if fit_hz != wave['srate']:
        wave = wave.filter('hz_%.2f' % fit_hz)
    sg.add_wave(wave=wave)
    sg.add_event(ev=event)
    #sg.fix_arrival_times()
    return sg


def e_step(sigvisa_graph,  fit_hz, tmpl_optim_params, wiggle_optim_params, fit_wiggles):

    s = Sigvisa()

    st = time.time()
    sigvisa_graph.optimize_templates(optim_params=tmpl_optim_params)

    if fit_wiggles:
        sigvisa_graph.init_wiggles_from_template()

        for i in range(4):
            sigvisa_graph.optimize_templates(optim_params=tmpl_optim_params)
            sigvisa_graph.init_wiggles_from_template()

        sigvisa_graph.optimize_wiggles(optim_params=wiggle_optim_params)
        sigvisa_graph.optimize_templates(optim_params=tmpl_optim_params)
        sigvisa_graph.optimize_wiggles(optim_params=wiggle_optim_params)

    et = time.time()

    tops=repr(tmpl_optim_params)[1:-1]
    wops=repr(wiggle_optim_params)[1:-1] if fit_wiggles else ""
    fitids = sigvisa_graph.save_template_params(tmpl_optim_param_str = tops,
                                                wiggle_optim_param_str = wops,
                                                elapsed=et - st, hz=fit_hz)
    if fit_wiggles:
        sigvisa_graph.save_wiggle_params()

    return fitids[0]


def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("--tmpl_optim_params", dest="tmpl_optim_params", default="", type="str", help="fitting param string")
    parser.add_option("--wiggle_optim_params", dest="wiggle_optim_params", default="'normalize': False, 'bfgs_factr': 1000, 'disp': True", type="str", help="fitting param string")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run name")
    parser.add_option("-i", "--run_iteration", dest="run_iteration", default=None, type="int",
                      help="run iteration (default is to use the next iteration)")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
    parser.add_option("--orid", dest="orid", default=None, type="int", help="origin ID")
    parser.add_option("--init_run_name", dest="init_run_name", default=None, type="str",
                      help="initialize template fitting with results from this run name")
    parser.add_option("--init_run_iteration", dest="init_run_iteration", default=None, type="int",
                      help="initialize template fitting with results from this run iteration (default: most recent)")
    parser.add_option("--template_shape", dest="template_shape", default="paired_exp", type="str",
                      help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest="template_model", default="dummy", type="str", help="")
    parser.add_option("--fit_wiggles", dest="fit_wiggles", default=False, action="store_true", help="")
    parser.add_option("--wiggle_family", dest="wiggle_family", default="fourier_0.8", type="str", help="")
    parser.add_option("--wiggle_model", dest="wiggle_model", default="dummy", type="str", help="")
    parser.add_option("--band", dest="band", default="freq_2.0_3.0", type="str", help="")
    parser.add_option("--chan", dest="chan", default="BHZ", type="str", help="")
    parser.add_option("--hz", dest="hz", default=5.0, type="float", help="sampling rate at which to fit the template")
    parser.add_option("--nm_type", dest="nm_type", default="ar", type="str",
                      help="type of noise model to use (ar)")


    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    if options.run_name is None or options.run_iteration is None:
        raise ValueError("must specify run name and iteration!")

    if not (options.evid is None and options.orid is None):
        ev = get_event(evid=options.evid, orid=options.orid)
    else:
        raise ValueError("Must specify event id (evid) or origin id (orid) to fit.")

    sigvisa_graph = setup_graph(event=ev, sta=options.sta, chan=options.chan, band=options.band,
                                tm_shape=options.template_shape, tm_type=options.template_model,
                                wm_family=options.wiggle_family, wm_type=options.wiggle_model, phases="leb",
                                fit_hz=options.hz, nm_type=options.nm_type,
                                output_run_name = options.run_name, output_iteration = options.run_iteration)

    fitid = e_step(sigvisa_graph,  fit_hz = options.hz,
                   tmpl_optim_params=construct_optim_params(options.tmpl_optim_params),
                   wiggle_optim_params=construct_optim_params(options.wiggle_optim_params),
                   fit_wiggles = options.fit_wiggles)


    print "fit id %d completed successfully." % fitid

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
