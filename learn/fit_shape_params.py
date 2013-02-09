import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database.signal_data import *
from database import db
from models.templates.load_by_name import load_template_model

import utils.geog
import obspy.signal.util

from optparse import OptionParser

from sigvisa import Sigvisa
from signals.io import *
from infer.optimize.optim_utils import minimize_matrix
from models.wiggles.wiggle_models import PlainWiggleModel, StupidL1WiggleModel
from models.envelope_model import EnvelopeModel




def fit_event_wave(event, sta, chan, band, tm, output_run_name, output_iteration, init_run_name=None, init_iteration=None, optim_params=None, fit_hz=5):
    """
    Find the best-fitting template parameters for each band/channel of
    a particular event at a particular station. Store the template
    parameters in the database.
    """

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    wave = load_event_station_chan(event.evid, sta, chan, cursor=cursor).filter("%s;env" % band)

    wm = PlainWiggleModel(tm)
    em = EnvelopeModel(template_model=tm, wiggle_model=wm, phases=None)

    # DO THE FITTING
    method = optim_params['method']
    st = time.time()

    if fit_hz != wave['srate']:
        wave = wave.filter('hz_%.2f' % fit_hz)

    ll, fit_params = em.wave_log_likelihood_optimize(wave, event, use_leb_phases=True, optim_params=optim_params)

    et = time.time()
    fitid = store_template_params(wave, fit_params, optim_param_str=repr(optim_params)[1:-1], iid=False, acost=ll, run_name=output_run_name, iteration=output_iteration, elapsed=et-st, hz=fit_hz)
    s.dbconn.commit()
    return fitid

def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("--optim_params", dest="optim_params", default="", type="str", help="fitting param string")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run name")
    parser.add_option("-i", "--run_iteration", dest="run_iteration", default=None, type="int", help="run iteration (default is to use the next iteration)")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
    parser.add_option("--orid", dest="orid", default=None, type="int", help="origin ID")
    parser.add_option("--init_run_name", dest="init_run_name", default=None, type="str", help="initialize template fitting with results from this run name")
    parser.add_option("--init_run_iteration", dest="init_run_iteration", default=None, type="int", help="initialize template fitting with results from this run iteration (default: most recent)")
    parser.add_option("--template_shape", dest = "template_shape", default="paired_exp", type="str", help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest = "template_model", default="gp_dad", type="str", help="")
    parser.add_option("--band", dest = "band", default="freq_2.0_3.0", type="str", help="")
    parser.add_option("--chan", dest = "chan", default="BHZ", type="str", help="")


    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()


    if options.run_name is None or options.run_iteration is None:
        raise Exception("must specify run name and iteration!")

    if options.run_iteration == 1:
        iid=True
        fix_first_cols = 2
    elif options.run_iteration == 2:
        iid=False
        fix_first_cols = 1
    else:
        iid=False
        fix_first_cols = 0
    if not iid:
        raise Exception("non-iid fits not currently implemented!")
#        raise Exception("need to specify wiggle model for non-iid fits!")

    if options.init_run_name is None:
        tm = load_template_model(template_shape = options.template_shape, model_type="dummy")
    else:
        tm = load_template_model(template_shape = options.template_shape, run_name=options.init_run_name, run_iter=options.init_run_iteration, model_type=options.template_model)


    if not (options.evid is None and options.orid is None):
        ev = get_event(evid=options.evid, orid=options.orid)
    else:
        raise Exception("Must specify event id (evid) or origin id (orid) to fit.")

    try:
        fitid = fit_event_wave(event=ev, sta=options.sta, band=options.band, chan=options.chan, tm = tm, output_run_name=options.run_name, output_iteration=options.run_iteration, init_run_name=options.init_run_name, init_iteration=options.init_run_iteration,  optim_params=construct_optim_params(options.optim_params))
    except KeyboardInterrupt:
        s.dbconn.commit()
        raise
    except:
        s.dbconn.commit()
        print traceback.format_exc()
        raise
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
        import pdb; pdb.post_mortem(tb)
