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
from sigvisa.models.joint_gp import multiply_scalar_gaussian, JointIndepGaussian
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
with open("logs/mcmc/00347/step_000009/pickle.sg", 'rb') as f:
     sg_joint = pickle.load(f)

s = Sigvisa()
holdout_eid = None
target_phase="P"
band="freq_0.8_4.5"


run_name = "kampos_init_phases" # "aftershock_joint_P"
run_iter = 1
cursor = s.dbconn.cursor()
runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=True)
print "runid", runid

for sta in sg_joint.station_waves.keys():
    wn = list(sg_joint.station_waves[sta])[0]
    chan = wn.chan

    eids, phases = zip(*wn.arrivals())
    evids = eids

    def save_model(model, st, et, target, y=[], yvars=[], model_type="gp_lld"):
        model_fname = get_model_fname(run_name, run_iter, sta, chan, band, target_phase,
                                      target, model_type, evids,
                                      model_name=sg_joint.template_shape, unique=True)
        evid_fname = os.path.splitext(os.path.splitext(os.path.splitext(model_fname)[0])[0])[0] + '.evids'
        Xy_fname = os.path.splitext(os.path.splitext(os.path.splitext(model_fname)[0])[0])[0] + '.Xy.npz'
        np.savetxt(evid_fname, evids, fmt='%d')
        np.savez(Xy_fname, X=model.X, y=y, yvars=yvars)

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



    if True: # has GP template models
         for (param, b, c, phase), (jgp, hnodes) in sg_joint._joint_gpmodels[sta].items():
              #if param.startswith("db"): continue 
              if isinstance(jgp, JointIndepGaussian): continue
              st = time.time()
              gp = jgp.train_gp(holdout_eid=holdout_eid, force_ll=True)
              et = time.time()
              model_type = "gp_lld"
              if gp.featurizer_recovery is not None:
                   if gp.basis.startswith("poly"):
                        modelstr = gp.basis
                   elif len(gp.featurizer_recovery["extract_dim"]) == 1:
                        modelstr = "linear_mb"
                   elif len(gp.featurizer_recovery["extract_dim"]) == 2:
                        modelstr = "linear_distmb"
                   elif len(gp.featurizer_recovery["extract_dim"]) == 0:
                        modelstr = "bias"
                   else:
                        raise Exception("unrecognized parametric model for param %s: %s" % (param, gp.featurizer_recovery))
                   model_type = "gplocal+lld+%s" % modelstr
              save_model(gp, st, et, param, y=gp.y, yvars=gp.y_obs_variances, model_type=model_type)
              print "saved", param
