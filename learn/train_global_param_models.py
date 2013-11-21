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
             = sigma * (I + sigma_e^-1 * sigma_s * sigma_e^-1 * sigma)


    where sigma = (p_e + p_g)^-1: if we had a specific value of s to update with, we'd get a new precision for the global params by summing our current precision, with the precision on our observed value of s.


now imagine we have
sigma = (sigma_g^-1 - sigma_e^-1)^-1
A = - sigma * sigma_e^-1
b = sigma * sigma_g^-1 * mu_g

int_s N(g; As + b, sigma) N(s; mu_s, sigma_s)


    TODO: implement the equations above, hope for numerical stability

    """


def message_from_model(model, mean_prior, cov_prior, station_slack):

    """

    Given a model containing p(s|d) (trained under some prior p(s)), compute the message

    m_{s->g} (g) = int_ds p(s|g) * m_{d->s} (s)

    where p(s|g) = N(s; g, station_slack)

    and m_{d->s} (s) = p(d|s) = p(s|d)/p(s)
                              = p(s|d) / N(s; mean_prior, cov_prior)
                              = N(c, C)
                                with c and C defined as computed below.

    Most of the work is in computing the message m_{d->s}. The only additional effect of passing
    that message through the node s, yielding m_{s->g}(g), is the addition of the station_slack
    term to the covariance matrix.

    Note that the message we return, taken as a function of g, is proportional to p(d|g).

    """

    mu_s = model.mean
    try:
        sigma_s = np.dot(model.sqrt_covar.T, model.sqrt_covar)
    except:
        sigma_s = model.c

    sigma_s_inv = np.linalg.inv(sigma_s)
    cov_prior_inv = np.linalg.inv(cov_prior)

    C = np.linalg.inv(sigma_s_inv - cov_prior_inv)
    c = np.dot(C, np.dot(sigma_s_inv, mu_s) - np.dot(cov_prior_inv, mean_prior))
    return c, C + station_slack

def receive_message(mu_g, cov_g, mu_message, cov_message):

    """
    Given current global param distribution N(mu_g, cov_g),
    update to the new distribution N(mu_g, cov_g)N(mu_message, cov_message).

    Note we can simulate the effect of "removing" a message by passing in a
    negated version of cov_message.
    """

    inv_cov_g = np.linalg.inv(cov_g)
    inv_cov_message = np.linalg.inv(cov_message)

    C = np.linalg.inv(inv_cov_g + inv_cov_message)
    c = np.dot(C, np.dot(inv_cov_g, mu_g) + np.dot(inv_cov_message, mu_message))
    return c,C

def receive_data_message(mu_g, cov_g, H, y, station_slack, noise_var, negate=False):
    """
    Given current global param distribution N(mu_g, cov_g), update to the new
    distribution after observing data (H,y) at a station, where H is the feature matrix
    and we assume a linear model with i.i.d. Gaussian noise of variance noise_var.

    This function is not actually used in practice, since it is limited to i.i.d. noise
    and can't handle Gaussian processes*. But it's a useful sanity check for the other
    methods above; we should get identical results from

    mu2_g, cov2_g = receive_data_message(mu_g, cov_g, H, y, station_slack, noise_var)
    where H = features(X) with respect to some basis

    as from

    lbm = LinearBasisModel(X=X, y=y, basis=basis, sta=None, param_mean = mu_g, param_cov = (cov_g + station_slack))
    m, c = message_from_model(lbm, mu_g, (cov_g + station_slack), station_slack)
    mu2_g, cov2_g = receive_message(mu_g, cov_g, m,c).


    * note this could actually handle GPs if we replaced eye*noise_var with
      the actual GP kernel matrix K_y...
    """

    invA = np.linalg.inv(cov_g)
    B = np.eye(len(y))*noise_var + np.dot(H, np.dot(station_slack, H.T))
    if negate:
        B = -B
    invB = np.linalg.inv(B)

    C = np.linalg.inv(invA + np.dot(H.T, np.dot(invB, H)))
    c = np.dot(C, np.dot(invA, mu_g) + np.dot(H.T, np.dot(invB, y)))
    return c, C


def retrain_model(modelid, prior_mean, prior_cov, bounds=None):

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sql_query = "select model_fname, model_type, param, site, optim_method from sigvisa_param_model where modelid=%d" % modelid
    cursor.execute(sql_query)
    model_fname, model_type, target, site, optim_params = cursor.fetchone()

    xy_fname = os.path.splitext(os.path.splitext(model_fname)[0])[0] + '.Xy'
    d = np.load(xy_fname)
    X, y = d['X'], d['y']

    model = learn_model(X, y, model_type, target=target, sta=site, optim_params=optim_params, gp_build_tree=False, k=options.subsample, bounds=bounds, param_mean=prior_mean, param_cov=prior_cov)

    hyperparams = model_params(model, model_type)
    model.save_trained_model(model_fname)
    sql_query = "update sigvisa_param_model set shrinkage='%s', hyperparams='%s', training_ll=%f, timestamp=%f where modelid=%d" % (str(prior_mean) + str(prior_cov), hyperparams, model.log_likelihood(), time.time(), modelid)
    cursor.execute(sql_query)

    cursor.close()

def retrain_models(modelids, global_var=1.0, station_slack_var=1.0):

    mu_g = np.zeros((n,))
    sigma_g = np.eye(n) * global_var
    station_slack = np.eye(n) * station_slack_var

    messages = dict()
    for modelid in modelids:
        model = load_model(modelid=modelid)
        m,c = message_from_model(model)
        messages[modelid] = (m,c)
        mu_g, cov_g = receive_message(mu_g, cov_g, m,c)

    for modelid in modelids:
        m,c = messages[modelid]
        mu_partial, cov_partial = receive_message(mu_g, cov_g, m,c)

        retrain_model(modelid, mu_partial, cov_partial)



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
    parser.add_option(
        "-m", "--model_type", dest="model_type", default="gp_lld_dist5", type="str", help="type of model to train (gp_lld_dist5)")


    (options, args) = parser.parse_args()


    runid = options.runid
    run_name, run_iter = read_fitting_run(runid)

    targets = options.targets.split(',')
    for target in targets:
        sql_query = "select modelid from sigvisa_param_model where target='%s' and model_type='%s' and band='%s' and chan='%s' and phase='%s'" % (target_cond, options.model_type, options.band. options.chan, options.phase)
        cursor.execute(sql_query)
        modelids = np.array(cursor.fetchall())
        retrain_models(modelids)


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