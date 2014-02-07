import os
import errno
import sys
import time
import traceback
import numpy as np
import numpy.ma as ma
import scipy
import re

from sigvisa import Sigvisa
from sigvisa.models.latent_arrival import LatentArrivalNode



def ar_smoothing_conditional(model, data, i):
    # compute the smoothed distribution on data[i], given the rest of the data

    # see p. 247 of "TIME SERIES ANALYSIS VIA THE GIBBS SAMPLER", MCCULLOCH and TSAY, JOURNAL OF TIME SERIES ANALYSIS, 1994.
    # http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9892.1994.tb00188.x/pdf

    pred = _ # precompute compute filtered mean for each variable
    residuals = data - pred

    # data gives us the y's
    # pred gives us linear combinations of ys
    # residuals gives us y's, minus their preceding linear combinations.

    # so in each residual following i, there is exactly one term that involves data[i]

    for t in range(i+1, i + p+1):
        z[t] = residuals[t] - data[i] * model.params[t-i-1]

    # okay, so now we have z
    smoothed_mean = ( np.dot(z[i+1:i+p+1], params) + pred[i] ) / np.linalg.norm(params)**2
    smoothed_var = model.emd.std**2 / np.linalg.norm(params)**2

    # so this works: it's naively written to compute predictions and residuals over the whole signal instead of just the p elements that we need, but that's fixable.

    # when we do a gibbs sweep, we'll be updating everything online. so once we change an x_i, that will change the filtered predictions and residuals for the p steps past that. one approach is still to precompute all the predictions and residuals, then just update them by adding/subtracting the appropriate weight whenever we make a change. this way we do p floating point operations whenever we change a value, as opposed to doing p^2 operations every time we need to compute updated predictions and residuals for a new index (since we have to compute predictions p steps in the future, and each of those depends on its p predecessors).

    # pseudocode for the full algorithm:
    # - compute our current latent representation of y (the observed signal minus predicted arrivals)
    # - precompute the template shape that will intermediate between x and y
    # - compute predicted values for x's and y's (no need for residuals since these are constant time to compute at a given index so we don't save by precomputing)
    # loop over indices. for each i, do:
    #    - compute smoothed distribution on x_i (as above)
    #    - compute smoothed distribution on y_i (also as above)
    #    - combine these for a posterior conditional on x_i (as in the google doc)
    #    - sample from that posterior conditional to get a new value for x_i (automatically accepted)
    #    - update the predicted values for x, based on the new x values.
    #    - update our latent y representation (only change will be y_i)
    #    - update predicted values for y
    # we do this for a full forward sweep over y, then a backward sweep.


def resample_gibbs():
