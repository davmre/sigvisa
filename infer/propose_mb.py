import numpy as np
import copy
import sys
import traceback
import pdb
import pickle
import os

import scipy.optimize
import scipy.stats

from sigvisa import Sigvisa

import sigvisa.source.brune_source as brune
from sigvisa.graph.sigvisa_graph import SigvisaGraph, get_param_model_id
from sigvisa.learn.train_param_common import load_modelid
import numdifftools as nd

def ev_mb_posterior_laplace_functional(sg, ev, targets, amps):

    s = Sigvisa()
    models = []
    for (sta, phase, chan, band) in targets:
        site = s.get_array_site(sta)
        ampt_model_type = sg._tm_type(param="amp_transfer", site=site, wiggle_param=False)
        modelid = get_param_model_id(runid=sg.runid, sta=sta,
                                     phase=phase, model_type=ampt_model_type,
                                     param="amp_transfer", template_shape=sg.template_shape,
                                     chan=chan, band=band)
        model = load_modelid(modelid)
        models.append(model)

    def amp_transfer_nlp(x):
        mb = x[0]
        lp = 0
        for amp, model, (sta, phase, chan, band) in zip(amps, models, targets):
            source_amp = brune.source_logamp(mb, phase=phase, band=band)
            lp += model.log_p(amp - source_amp, ev)
        return -lp

    x0 = [4.0,]
    r = scipy.optimize.minimize(amp_transfer_nlp, x0, bounds=[(0.0, 10.0)])
    prec = nd.Hessian(amp_transfer_nlp).hessian(r.x)

    var = 1.0/prec[0,0]

    return r.x[0], var


def ev_mb_posterior_laplace(sg, eid):

    mb_node = sg.evnodes[eid]["mb"]
    amp_nodes = [n for n in sg.extended_evnodes[eid] if "amp_transfer" in n.label]
    ch_nodes = [nn for n in amp_nodes for nn in n.children if "coda_height" in nn.label]
    coda_heights = [nn.get_value() for nn in ch_nodes]
    ch_node_labels = [nn.label for nn in ch_nodes]

    assert(len(amp_nodes) > 0)

    def reset_coda_heights():
        for ch, label in zip(coda_heights, ch_node_labels):
            nn = sg.all_nodes[label]
            nn.set_value(ch)

    orig_mb = mb_node.get_value()

    def amp_transfer_nlp(x):
        mb = x[0]
        mb_node.set_value(mb)
        reset_coda_heights()
        return -np.sum([n.log_p() for n in amp_nodes])


    x0 = [4.0,]
    r = scipy.optimize.minimize(amp_transfer_nlp, x0, bounds=[(0.0, 10.0),])
    mb_node.set_value(orig_mb)
    reset_coda_heights()

    prec = nd.Hessian(amp_transfer_nlp).hessian(r.x)
    var = 1.0/prec[0,0]
    #return r.x[0], r.hess_inv[0,0], reset_coda_heights
    return r.x[0], var, reset_coda_heights

def propose_mb(sg, eid, fix_result=None):
    z, v, reset_coda_heights = ev_mb_posterior_laplace(sg, eid)
    rv = scipy.stats.norm(z, scale=np.sqrt(v))

    if fix_result is not None:
        print "reverse dist", z, v, "old mb", fix_result, "lp", rv.logpdf(fix_result)

        return rv.logpdf(fix_result), reset_coda_heights
    else:
        new_mb=rv.rvs(1)[0]
        print "eid", eid, "proposing mb", new_mb, "from dist", z, np.sqrt(v)
        sg.evnodes[eid]["mb"].set_value(new_mb)
        reset_coda_heights()

        return rv.logpdf(new_mb)
