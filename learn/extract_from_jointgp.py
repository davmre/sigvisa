import numpy as np

import time

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, do_coarse_to_fine, initialize_sg, initialize_from, do_inference
from sigvisa.models.distributions import Gaussian
from sigvisa.models.spatial_regression.baseline_models import ConstGaussianModel
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.treegp.gp import GPCov
from sigvisa.signals.io import load_event_station_chan
from sigvisa.models.joint_gp import multiply_scalar_gaussian
from sigvisa.utils.geog import dist_km_ev, dist_km

import os, sys, traceback
import cPickle as pickle

dbl1 = 9 #6
dbl2 = 17 #32

def hparams_to_cov(hparam_nodes):
     nv = hparam_nodes["noise_var"].get_value()
     try:
          sv = hparam_nodes["signal_var"].get_value()
     except:
          sv = 1 - nv
     hls, dls = hparam_nodes["horiz_lscale"].get_value(), hparam_nodes["depth_lscale"].get_value()

     #nv = 0.3
     #sv = 0.7
     #hls, dls = 20.0, 10.0

     cov_main = GPCov(wfn_str="matern32", wfn_params=[sv,], dfn_str="lld", dfn_params=[hls, dls])
     return cov_main, nv


from sigvisa.learn.fit_shape_params_mcmc import compute_wavelet_messages
from sigvisa.learn.train_param_common import learn_gp, insert_model, load_model, get_model_fname, model_params
from sigvisa.database.signal_data import *
from sigvisa.plotting.event_heatmap import EventHeatmap

#with open("/home/dmoore/python/sigvisa/logs/mcmc/01962/step_000019/pickle.sg", 'rb') as f:
#    sg_joint = pickle.load(f)
with open("logs/mcmc/00198/step_000999/pickle.sg", 'rb') as f:
     sg_joint = pickle.load(f)

s = Sigvisa()
holdout_eid = None
target_phase="P"
band="freq_0.8_4.5"
wiggle_family = "db4_2.0_3_15.0"

run_name = "kampos_init_phases" # "aftershock_joint_P"
run_iter = 1
cursor = s.dbconn.cursor()
runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=True)
print "runid", runid

for sta in sg_joint.station_waves.keys():
    wn = list(sg_joint.station_waves[sta])[0]
    chan = wn.chan
    X = []
    eids = []
    message_means = []
    message_vars = []
    posterior_means = []
    posterior_vars = []
    sta_loc = s.earthmodel.site_info(sta, 0)[:2]

    for wn in sg_joint.station_waves[sta]:
         if wn.chan != chan or wn.band != band: continue
         gp_messages, gp_posteriors = compute_wavelet_messages(sg_joint, wn)
         for (eid, phase) in gp_messages.keys():
             if eid == holdout_eid: continue
             if phase != target_phase: continue
             eids.append(eid)
             gpm = gp_messages[(eid, phase)]
             gpp = gp_posteriors[(eid, phase)]
             message_means.append(gpm[0])
             message_vars.append(gpm[1])
             posterior_means.append(gpp[0])
             posterior_vars.append(gpp[1])
             ev = sg_joint.get_event(eid)
             X.append((ev.lon, ev.lat, ev.depth, ev.mb, dist_km((ev.lon, ev.lat), sta_loc)))



    X = np.array(X)

    evids = eids

    model_type = "gp_lld"
    # TODO: properly track the number of coefs in the repeatable levels
    n_coefs = len(message_means[0])
    n_coefs_skipped = np.sum(wn.wavelet_basis[4][-sg_joint.skip_levels:])
    n_coefs_modeled = n_coefs-n_coefs_skipped
    print n_coefs, n_coefs_skipped, n_coefs_modeled

    targets = [wiggle_family + "_%d" % i for i in range(n_coefs_modeled)]

    def save_model(model, st, et, target, y=[], yvars=[], model_type="gp_lld"):
        model_fname = get_model_fname(run_name, run_iter, sta, chan, band, target_phase,
                                      target, model_type, evids,
                                      model_name=sg_joint.template_shape, unique=True)
        evid_fname = os.path.splitext(os.path.splitext(os.path.splitext(model_fname)[0])[0])[0] + '.evids'
        Xy_fname = os.path.splitext(os.path.splitext(os.path.splitext(model_fname)[0])[0])[0] + '.Xy.npz'
        np.savetxt(evid_fname, evids, fmt='%d')
        np.savez(Xy_fname, X=X, y=y, yvars=yvars)

        model.save_trained_model(model_fname)
        template_options = {'template_shape': sg_joint.template_shape, }
        insert_options = template_options

        try:
             hparams = model_params(model, model_type)
        except:
             hparams = ""


        shrinkage = {}
        modelid = insert_model(s.dbconn, fitting_runid=runid, param=target,
                               site=sta, chan=chan, band=band, phase=target_phase,
                               model_type=model_type, model_fname=model_fname,
                               training_set_fname=evid_fname,
                               training_ll=model.log_likelihood(),
                               require_human_approved=False, max_acost=9999999,
                               n_evids=len(evids), min_amp=-999, elapsed=(et-st),
                               hyperparams = hparams,
                               optim_method = None,
                               shrinkage=repr(shrinkage), **insert_options)
        print "inserted as", modelid, sta, chan, "ll", model.log_likelihood()


    for level in range(sg_joint.skip_levels):
        model = ConstGaussianModel(sta=sta, mean=0.0, std=1.0)
        save_model(model, 0.0, 0.01, wiggle_family + "_level%d" % (level+1), model_type="constant_gaussian")

    gps = []

    for i in range(n_coefs_modeled):
        y = np.array([mm[i] for mm in message_means])
        yvars = np.array([mv[i] for mv in message_vars])
        target = targets[i]

        jgp, hparam_nodes = sg_joint._joint_gpmodels[sta][(target, band, chan, target_phase)]
        cov_main, nv = hparams_to_cov(hparam_nodes)

        # if the jointgp includes a parametric component,
        # the trained GP should too. 
        gpinit_params = {}
        try:
             gpinit_params = jgp._gpinit_params
        except:
             pass

        st = time.time()
        gp = learn_gp(X=X, y=y, y_obs_variances=yvars, sta=sta,
                      kernel_str="lld",
                      target=target, build_tree=False,
                      optimize=False, noise_var=nv,
                      cov_main=cov_main, **gpinit_params)
        et = time.time()
        gps.append(gp)

        save_model(gp, st, et, target, y=y, yvars=yvars)


    # how to learn level params?
    # what's the prior variance? the model is that we sample each *true* coef from this prior, then observe through the Kalman model. One option: take empirical variance of posterior means. But this would fail if we had uninformative observations: the posterior means wouold all be zero and the variance is zero.
"""
Suppose we have a invGamme prior on the variance. We want p(var|signals) = int_coefs p(var|coefs)p(coefs|signals)
where p_coefs|signals) is Gaussian and p(var|coefs) is an invGamma posteiror that comes from p(coefs|var)p(var).
Can we just write out the model?
var ~ InvGamma
coefs|var ~ Gaussian
signals|coefs ~ LinGaussian

p(var, coefs, signals) = p(var)p(coefs|var)p(signals|coefs)
p(var | signals) \propto int_coefs p(var)p(cofs|var)p(signals|coefs)
= p(var) int_coefs p(signals|coefs) p(coefs|var)
"""
