import numpy as np
import os
import cPickle as pickle

vals = {'noise_var': 0.01,
        'signal_var': 1.0,
        'horiz_lscale': 100,
        'depth_lscale': 5.0}

for fn in os.listdir('.'):
    if fn.startswith("step"):
        with open(os.path.join(fn, 'pickle.sg'), 'rb') as f:
            sg = pickle.load(f)
        break

true_params = dict()
for sta in sg.station_waves.keys():
    true_params[sta] = dict()
    nparams = len(sg.wavelet_basis(5.0)[0][0])
    for hparam in sg.jointgp_hparam_prior.keys():
        true_params[sta][hparam] = [vals[hparam],]*nparams

with open(os.path.join('gp_hparams', 'true.pkl'), 'wb') as f:
    pickle.dump(true_params, f)
