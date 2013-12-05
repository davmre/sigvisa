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
from sigvisa.utils.fileutils import clear_directory, remove_directory, mkdir_p

from optparse import OptionParser

from sigvisa import Sigvisa
from sigvisa.signals.io import *

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.models.distributions import Uniform, Exponential
from sigvisa.models.ttime import tt_predict

def fetch_ev_P_signal(ev, sta, chan):
    atime = tt_predict(ev, sta, phase='P') + ev.time
    return fetch_waveform(sta, chan, atime - 10, atime + 100)

def setup_graphs(ev, sta, chan, band,
                tm_shape, tm_type, wm_family, wm_type, phases,
                init_run_name, init_iteration, fit_hz=5, nm_type="ar", absorb_n_phases=False):
    sg = SigvisaGraph(template_model_type=tm_type, template_shape=tm_shape,
                      wiggle_model_type=wm_type, wiggle_family=wm_family,
                      phases=phases, nm_type = nm_type,
                      run_name=init_run_name, iteration=init_iteration,
                      absorb_n_phases=absorb_n_phases)
    sg_noise = SigvisaGraph(nm_type = nm_type,
                            absorb_n_phases=absorb_n_phases)

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    wave = fetch_ev_P_signal(ev, sta, chan).filter("%s;env" % band)
    cursor.close()
    if fit_hz != wave['srate']:
        wave = wave.filter('hz_%.2f' % fit_hz)
    sg.add_wave(wave=wave)
    sg_noise.add_wave(wave=wave)

    evnodes = sg.add_event(ev=ev, fixed=True)
    evnode_lp = np.sum([n.log_p() for n in set(evnodes.values())])
    evnode_lp += sg.nevents_log_p()
    return sg, sg_noise, evnode_lp


def random_event(seed):

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    stime, etime = read_timerange(cursor, "test", hours=None, skip=0)
    cursor.close()

    np.random.seed(seed)
    s.sigmodel.srand(seed)

    event_time_dist = Uniform(stime, etime)
    event_mag_dist = Exponential(rate=10.0, min_value=3.5)

    origin_time = event_time_dist.sample()
    lon, lat, depth = s.sigmodel.event_location_prior_sample()
    mb = event_mag_dist.sample()
    natural_source = True # TODO : sample from source prior

    ev = get_event(lon=lon, lat=lat, depth=depth, time=origin_time, mb=mb, natural_source=natural_source)

    return ev

def ev_lp(sg, evnode_lp, run_dir):

    s = Sigvisa()

    clear_directory(run_dir)

    sg.parent_predict_all()
    run_open_world_MH(sg, enable_event_openworld=False, enable_event_moves=False, enable_template_openworld=False, enable_template_moves=True, run_dir=run_dir, steps=2000, skip=20000)

    lps = np.loadtxt(os.path.join(run_dir, 'lp.txt'))[500:]
    lps = lps - evnode_lp # p(signal, params | event) = p(signal, params, event) / p(event)

    # factor out a constant to avoid overflow
    max_lp = np.max(lps)
    lps -= max_lp
    ps = np.exp(lps)
    mean_p = np.mean(ps)
    log_mean_p = np.exp(mean_p) + max_lp

    remove_directory(run_dir)

    return log_mean_p


def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station for which to fit templates")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run name")
    parser.add_option("-i", "--run_iteration", dest="run_iteration", default=None, type="int",
                      help="run iteration (default is to use the next iteration)")
    parser.add_option("-e", "--evid", dest="evid", default=None, type="str", help="event ID")
    parser.add_option("--orid", dest="orid", default=None, type="int", help="origin ID")
    parser.add_option("--template_shape", dest="template_shape", default="paired_exp", type="str",
                      help="template model type to fit parameters under (paired_exp)")
    parser.add_option("--template_model", dest="template_model", default="dummy", type="str", help="")
    parser.add_option("--wiggle_family", dest="wiggle_family", default="fourier_0.8", type="str", help="")
    parser.add_option("--wiggle_model", dest="wiggle_model", default="dummy", type="str", help="")
    parser.add_option("--phases", dest="phases", default="leb", type="str", help="")
    parser.add_option("--band", dest="band", default="freq_2.0_3.0", type="str", help="")
    parser.add_option("--chan", dest="chan", default="auto", type="str", help="")
    parser.add_option("--hz", dest="hz", default=5.0, type="float", help="sampling rate at which to fit the template")
    parser.add_option("--nm_type", dest="nm_type", default="ar", type="str",
                      help="type of noise model to use (ar)")
    parser.add_option("--absorb_n_phases", dest="absorb_n_phases", default=False, action="store_true", help="")

    (options, args) = parser.parse_args()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    if options.phases == "leb":
        phases = "leb"
    else:
        phases = options.phases.split(',')

    if '{' in options.template_model:
        # example: "{'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb'}"
        template_model = eval(options.template_model)
    elif options.template_model=='param':
        template_model = {'tt_residual': 'constant_laplacian' , 'peak_offset': 'param_linear_mb', 'amp_transfer': 'param_sin1', 'coda_decay': 'param_linear_distmb'}

    else:
        template_model = options.template_model

    if options.run_name is None or options.run_iteration is None:
        raise ValueError("must specify run name and iteration!")

    if options.evid.startswith('fake'):
        seed = int(options.evid[4:])
        ev = random_event(seed)
        print "got random ev", ev
    else:
            ev = get_event(evid=int(options.evid))

    sg, sg_noise, evnode_lp = setup_graphs(ev=ev, sta=options.sta, chan=options.chan, band=options.band,
                                tm_shape=options.template_shape, tm_type=template_model,
                                wm_family=options.wiggle_family, wm_type=options.wiggle_model,
                                phases=phases,
                                fit_hz=options.hz, nm_type=options.nm_type,
                                init_run_name = options.run_name, init_iteration = options.run_iteration-1,
                                absorb_n_phases=options.absorb_n_phases)

    run_dir = 'experiments/logodds/' + options.sta + "_" + options.evid
    lp = ev_lp(sg, evnode_lp, run_dir=run_dir)

    noise_lp = sg_noise.current_log_p() - sg_noise.nevents_log_p()

    print "ev lp", lp
    print "noise lp", noise_lp

    results_dir = 'experiments/logodds/results/%s/' % options.sta
    if options.evid is not None and (not options.evid.startswith('fake')):
        results_dir += "events/"
        mkdir_p(results_dir)
        with open(results_dir + str(options.evid), 'w') as f:
            f.write("%f %f %f\n" % (lp, noise_lp, lp - noise_lp))
    else:
        results_dir += "calibration/"
        mkdir_p(results_dir)
        with open(results_dir + str(options.evid), 'w') as f:
            f.write("%f %f %f\n" % (lp, noise_lp, lp - noise_lp))


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
