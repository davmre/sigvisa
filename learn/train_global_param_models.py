import time
import numpy as np
import os
import shutil
import errno
import re
import pickle
from collections import defaultdict
from functools32 import lru_cache

from sigvisa import Sigvisa

from sigvisa.database.signal_data import get_fitting_runid, insert_wiggle, ensure_dir_exists

from sigvisa.source.event import get_event
from sigvisa.learn.train_param_common import load_modelid
import sigvisa.utils.geog as geog
from sigvisa.models import DummyModel
from sigvisa.models.distributions import Uniform, Poisson, Gaussian, Exponential
from sigvisa.models.ev_prior import setup_event
from sigvisa.models.ttime import tt_predict, tt_log_p, ArrivalTimeNode
from sigvisa.graph.nodes import Node
from sigvisa.graph.dag import DirectedGraphModel
from sigvisa.graph.graph_utils import extract_sta_node, predict_phases, create_key, get_parent_value, parse_key
from sigvisa.models.signal_model import ObservedSignalNode, update_arrivals
from sigvisa.graph.array_node import ArrayNode
from sigvisa.models.templates.load_by_name import load_template_generator
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle import extract_phase_wiggle
from sigvisa.models.wiggles import load_wiggle_generator, load_wiggle_generator_by_family
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform

from sigvisa.utils.fileutils import clear_directory, mkdir_p

class ModelNotFoundError(Exception):
    pass




"""


given a list of params, a fitting runid, and a local model type (including parametric component).

a) get the list of ALL stations for which fits exist in this runid
b) train "preliminary" param models for each station. go ahead and save to DB
c) go through and pull out their parametric components. compute a posterior (mean and covariance) on global parameters. save this to a file somewhere
d) go through and retrain each model with the new prior. remove the old DB entry, and replace it with the new trained model. also don't re-optimize hyperparams from scratch, but start with the learned hyperparams from the last round
e) keep doing this until station param posteriors stop changing (measured in KL divergence? or just in euclidean distance on mean and covar matrices)

QUESTION: can we update global params on EVERY step? answer: yes, this is still coordinate ascent but we're breaking it down to the level of individual stations as coordinates.


TODO:
- figure out update to global param posterior from each station update (also: formalize what we're doing: we're really updating the *posterior*, not just finding a good point estimate of global parameters. and then we should be able to integrate out the global params themselves when optimizing the per-station params). so:
  - figure out the resulting per-station prior, given a global posterior

- figure out a convention for logging the global param posterior

- write code to re-train and replace a param model in the database (starting from its previous hyperparams)

- do some testing and experimentation to see to what extent we actually get reasonable param estimates
- write code to plot params at multiple stations

- maybe think about other features to include. how to keep it from blowing up at 0? note there's nothing *intrinsic* in the polynomial which

- figure out the connection between global and station models. we definitely want each station to have its own mean / constant term, so that should have wider variance. maybe we also don't include a constant in the global model? so each station has nonconstant terms conditioned on the global model, but a constant term just coming from its own very wide prior?


p(sta, global | data) = p(sta1, sta2, sta3, global | data)
= p(data | sta1, sta2, sta3, global)p(sta1, sta2, sta3, global)/ p(data)
= p(data1 | sta1)p(data2|sta2)p(data3|sta3) * p(sta1|global)p(sta2|global)p(sta3|global)p(global)/ p(data)
=

now the last few terms here are Gaussians, so multiplying them should be another Gaussian

the first terms are multivariate Gaussian for fixed GP hyperparams. so for fixed hyperparams, I ought to be able to multiply everything together and just get a joint Gaussian posterior on the station and global params.

but if we optimize over hyperparams,then p(data1|sta1) is actually the max over a bunch of Gaussian distributions. which is non-Gaussian.

but wat?

if I have a Gaussian prior on station params, then I max over hyperparams, I still get some fixed set of hyperparams which then give me a Gaussian posterior on the station params.

so really the situation is, I have all the station/global distance params which are jointly gaussian. then I have the hyperparams. so we have

p(hparams, dparams | data)

and we'd like to integrate over hyperparams to just get the posterior on params. but we can't do that (easily). so we take a point estimate of the hyperparams (given the current dparam distribution), then find the distribution of the dparams under the new hyperparams, and repeat. This is basically EM, where we're maximizing the hyperparameters in an M step, then getting the distribution over the distance parameters in the E step.

so how do I think of this? it's a little complicated because I'm not actually holding distance params fixed as I try different hyperparameters: instead I'm finding a new set of (per-station) distance params for each hyperparam set I try. but that's fine: I just retreat to the global params.
now this is back to what I was doing before, except that I have a justification for it.

there's a separate issue, which is how to learn the obs model. for now I'll just do something arbitrary.
"""


def update_global_params(global_mean, global_posterior, station_mean, station_covar, obs_covar):

    # we're not updating global params from an observation of station params. instead we have a distribution on the station params.

    # we have global + eps = station
    # <station, h> + eps2 = datapoints

    # now we have p(station | datapoints) which we are saying is Gaussian (technically it's also given the other global params, but hopefully we can ignore this in the interest of message passing)
    # and we want p(global | datapoints)
    # =int_station p(global | station)p(station|datapoints)
    # =int_station p(station|global)p(global)p(station|datapoints)
    # which should also be Gaussian.

    #

    """
    g = global params
    s = station params
    d = data

    LET:
    prior(g) ~ N(\mu_g, sigma_g)
    s = g + eps, eps ~ N(0, \sigma_e)


    p(g | d) = int_s p(g|s)p(s|d) ds
    where
    p(s|d) ~ N(s; mu_s, sigma_s)

    p(g|s) ~ N(g; mu, sigma)
             mu = sigma * (sigma_e^-1 * s + sigma_g^-1 * mu_g)
             sigma = (sigma_e^-1 + sigma_g^-1)^-1
    (see "observations" section of my gaussian identity notes)


    now let A = sigma * sigma_e^-1, b = sigma * sigma_g^-1 * mu_g, then
    p(g|d) = int_s N(g; As+b, sigma) N(s; mu_s, sigma_s) ds
    by the "linear gaussian marginalization" section of my notes, this gives

    p(g|d) ~ N(g; A*mu_s +b, sigma + A * sigma_s * A^T )

    so

    mu_g <= A*mu_s + b
          = sigma * (sigma_e^-1 * mu_s + sigma_g^-1 * mu_g)
          = sigma * (p_e * mu_s + p_g * mu_g)

    sigma_g <= sigma + A * sigma_s * A^T
             = sigma + sigma * sigma_e^-1 * sigma_s * sigma_e^-1 * sigma
             = sigma * (I + sigma_e^-1 * sigma_s * sigma_e^-1)


    where sigma = (p_e + p_g)^-1: if we had a specific value of s to update with, we'd get a new precision for the global params by summing our current precision, with the precision on our observed value of s.


    TODO: implement the equations above, hope for numerical stability

    """





def do_em(options, site_elements, target, chan, band, phase, min_amp):
    model_type = "gplocal+lld+%s" % options.basisfn_str

    # choose a global param prior
    # compute the corresponding prior on station params (integrating out the global params)
    global_mean = ??
    global_covar = ??

    basisfns, _, _ = basisfns_from_str(options.basisfn_str)
    k = len(basisfns)

    # prior distribution on global parameters
    global_mean = np.zeros((k,))
    global_covar = np.eye(k) * 10000

    obs_covar = np.eye(k) * 100

    for (site, elems) in site_elements.items():
        chan = chan_for_site(site, options)

        try:
            X, y, evids = load_site_data(elems, wiggles=False, target=target, param_num=None, runid=options.runid, chan=chan, band=band, phases=[phase, ], require_human_approved=options.require_human_approved, max_acost=options.max_acost, min_amp=min_amp, array = options.array_joint)
        except NoDataException:
            print "no data for %s %s %s, skipping..." % (site, target, phase)
            continue

        model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, model_type, evids, unique=True)
        evid_fname = os.path.splitext(os.path.splitext(model_fname)[0])[0] + '.evids'
        np.savetxt(evid_fname, evids, fmt='%d')

        b = np.array(global_mean, copy=True)
        B = global_covar + obs_covar
        st = time.time()
        model = learn_gp(X=X, y=y, sta=site,
                         kernel_str='lld',
                         target=target, build_tree=False,
                         optim_params=optim_params, basisfns=basisfns, b=b, B=B)
        et = time.time()

        if np.isnan(model.log_likelihood()):
            print "error training model for %s %s %s, likelihood is nan! skipping.." % (site, target, phase)
            continue

        model.save_trained_model(model_fname)
        template_options = {'template_shape': options.template_shape, }
        modelid = insert_model(s.dbconn, fitting_runid=runid, param=target, site=site, chan=chan, band=band, phase=phase, model_type=model_type, model_fname=model_fname, training_set_fname=evid_fname, training_ll=model.log_likelihood(), require_human_approved=options.require_human_approved, max_acost=options.max_acost, n_evids=len(evids), min_amp=min_amp, elapsed=(et-st), hyperparams = model_params(model, model_type), optim_method = repr(optim_params) if model_type.startswith('gp') else None, **template_options)
        print "inserted as", modelid, "ll", model.log_likelihood()


def main():
    parser = OptionParser()

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser.add_option("-r", "--runid", dest="runid", default=None, type="str", help="runid")
    parser.add_option("-c", "--channel", dest="chan", default="vertical", type="str", help="name of channel to examine (vertical)")
    parser.add_option(
        "-n", "--band", dest="band", default="freq_2.0_3.0", type="str", help="name of band to examine (freq_2.0_3.0)")
    parser.add_option("-p", "--phase", dest="phase", default=",".join(s.phases), type="str",
                      help="phase for which to train models)")
    parser.add_option("-t", "--targets", dest="targets", default="coda_decay,amp_transfer,peak_offset", type="str",
                      help="comma-separated list of target parameter names (coda_decay,amp_transfer,peak_offset)")
    parser.add_option("--template_shape", dest="template_shape", default="paired_exp", type="str", help="")
    parser.add_option(
        "--basisfns", dest="basisfn_str", default="dist5", type="str", help="set of basis functions to use (dist5)")
    parser.add_option("--require_human_approved", dest="require_human_approved", default=False, action="store_true",
                      help="only train on human-approved good fits")
    parser.add_option(
        "--max_acost", dest="max_acost", default=np.float('inf'), type=float, help="maximum fitting cost of fits in training set (inf)")
    parser.add_option("--min_amp", dest="min_amp", default=-3, type=float,
                      help="only consider fits above the given amplitude (does not apply to amp_transfer fits)")
    parser.add_option("--min_amp_for_at", dest="min_amp_for_at", default=-5, type=float,
                      help="only consider fits above the given amplitude (for amp_transfer fits)")
    #parser.add_option("--optim_params", dest="optim_params", default="'method': 'bfgs_fastcoord', 'normalize': False, 'disp': True, 'bfgs_factr': 1e10, 'random_inits': 3", type="str", help="fitting param string")


    (options, args) = parser.parse_args()

    sql_query = "select sta from sigvisa_coda_fit where runid=%d" % options.runid
    cursor.execute(sql_query)
    stas = set(np.array(cursor.fetchall()).flatten())

    site_elements = defaultdict(set)

    for sta in stas:
        _, _, _, isarr, _, _, ref_site_id = s.earthmodel.site_info(sta, 0.0)
        ref_site_name = s.siteid_minus1_to_name[ref_site_id-1]
        site_elements[ref_site_name].add(sta)

    runid = options.runid
    run_name, run_iter = read_fitting_run(runid)



    while not STOPPING_CONDITION:

        for (param_num, target) in enumerate(targets):
            if target == "amp_transfer":
                min_amp = options.min_amp_for_at
            else:
                min_amp = options.min_amp



            # compute NEW global priors

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import sys, traceback, pdb
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
