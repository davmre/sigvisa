import numpy as np
import os
import cPickle as pickle

from sigvisa import Sigvisa
from sigvisa.utils.fileutils import mkdir_p
from sigvisa.ssms_c import CompactSupportSSM
from sigvisa.models.spatial_regression.SparseGP import SparseGP


def get_historical_signals(sg, phase="P"):
    history = []
    try:
        stas = sg.correlation_proposal_stas
    except AttributeError:
        stas = sg.station_waves.keys()

    for sta in stas:
        wns = sg.station_waves[sta]
        # TODO: figure out multiple bands/chans, or multiple time periods
        assert(len(wns) == 1)
        wn = wns[0]
        wn_history = get_historical_signals_for_wn(wn, phase)
        history = merge_station_histories(history, wn_history)
    history = filter_history_to_region(sg, history)
    return history

def merge_station_histories(h1, h2):
    key = lambda x : "%.3f %.3f %.1f" % (x[0,0], x[0,1], x[0,2])
    h1_dict = dict([(key(x), (x, s)) for (x, s) in h1])
    for (x, s) in h2:
        k = key(x)
        if k in h1_dict:
            x1, s1 = h1_dict[k]
            s1.update(s)
        else:
            h1_dict[k] = (x, s)
    return h1_dict.values()

def filter_history_to_region(sg, history):
    if sg.inference_region is None:
        return history
    else:
        return [(x,s) for (x, s) in history if sg.inference_region.contains_event(lon=x[0,0], lat=x[0,1])]

def get_historical_signals_for_wn(wn, phase):
    s = Sigvisa()
    if (phase not in wn.wavelet_param_models) or \
       (not isinstance(wn.wavelet_param_models[phase][0], SparseGP)): 
        return []
    modelid = wn.wavelet_param_models[phase][0].modelid
    fname = os.path.join(s.homedir, "db_cache", "history_%d.pkl" % modelid)
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            r = pickle.load(f)
    else:
        r = build_signal_library(wn, phase)
        mkdir_p(os.path.join(s.homedir, "db_cache"))
        with open(fname, 'wb') as f:
            pickle.dump(r, f)
    return r

def build_signal_library(wn, phase):

    (start_idxs, end_idxs, identities, basis_prototypes, levels, N) = wn.wavelet_basis

    n_basis = len(start_idxs)
    prior_means = np.zeros((n_basis,))
    prior_vars = np.ones((n_basis,))
    cssm = CompactSupportSSM(start_idxs, end_idxs, identities, basis_prototypes, prior_means, prior_vars, 0.0, 0.0)

    library = []
    Xs = wn.wavelet_param_models[phase][0].X
    wn_key = (wn.sta, wn.chan, wn.band)
    for x in Xs:
        x = x.reshape(1, -1)
        prior_means = np.array([gp.predict(cond=x) for gp in wn.wavelet_param_models[phase]], dtype=np.float)
        prior_vars = np.array([gp.variance(cond=x, include_obs=True) for gp in wn.wavelet_param_models[phase]], dtype=np.float)
        cssm.set_coef_prior(prior_means, prior_vars)
        s = cssm.mean_obs(N)
        s /= np.linalg.norm(s)
        library.append((x, {wn_key: s}))
    return library

