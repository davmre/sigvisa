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
    # - precompute predicted values for x's and y's (no need for residuals since these are constant time to compute at a given index so we don't save by precomputing)
    # loop over indices. for each i, do:
    #    - compute smoothed distribution on x_i (as above)
    #    - compute smoothed distribution on y_i (also as above)
    #    - combine these for a posterior conditional on x_i (as in the google doc)
    #    - sample from that posterior conditional to get a new value for x_i (automatically accepted)
    #    - update the predicted values for x, based on the new x values.
    #    - update our latent y representation (only change will be y_i)
    #    - update predicted values for y
    # we do this for a full forward sweep over y, then a backward sweep.

def gibbs_sweep(latent, start_idx, end_idx):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    (x, shape, repeatable) = latent._empirical_wiggle(return_components=True)
    padded_repeatable = np.pad_with_zeros(repeatable, len(x)) # TODO: find the function that actually implements this

    child_wn = list(latent.children)[0]
    predicted_signal = child_wn.assem_signal()
    observed_signal = child_wn.get_value()
    y = observed_signal - predicted_signal
    # TODO: don't touch the entire observed signal: we only need to compute these quantities for the region we're reasmpling

    model_x = latent.arwm
    model_y = child_wn.nm

    # offset between the latent and observed signals
    latent_offset = child_wn.arrival_start_idx(self.eid, self.phase)

    # we also precompute the following quantity, which determines the linear relationship between y and x. 
    # (this is the quantity C_i from the googledoc derivation)
    observed_nonrepeatable = observed_signal[latent_offset: latent_offset + len(x)] - shape * padded_repeatable 

    # TO START: let's assume the latent signal is fully contained in the observed signal
    # also we'll assume the observed signal is not masked

    # precompute predicted values for x's and y's (no need for residuals since these are constant time to compute at a given index so we don't save by precomputing)
    resample_len = end_idx - start_idx
    ar_predictions_x = np.zeros((resample_len,))
    ar_predictions_y = np.zeros((resample_len,))
    model_x.filtered_predictions(x, start_idx, ar_predictions_x)
    model_y.filtered_predictions(y, start_idx+latent_offset, ar_predictions_y)
    
    param_sqnorm_x = np.linalg.norm(model_x.p, 2)**2
    param_sqnorm_y = np.linalg.norm(model_y.p, 2)**2

    # precision (1/variance) of the smoothing distribution doesn't
    # depend on the actual data, so we can precompute it.
    smoothed_prec_x = param_sqnorm_x / model_x.em.std**2 
    smoothed_prec_y = param_sqnorm_y / model_y.em.std**2

    ##########
    # BEGIN C CODE HERE
    ##########

    # TODO: right now indexing of y is broken. I'm assuming y has the
    # same indices as x, but really, it will be offset.

    # loop over to resample each index in succession
    for i in range(start_idx, end_idx, step):
        smoothed_mean_xi = smoothed_mean(x, ar_predictions_x, start_idx, model_x.p, param_sqnorm_x, i)
        smoothed_mean_yi = smoothed_mean(y, ar_predictions_y, start_idx+latent_offset, model_y.p, param_sqnorm_y, i+latent_offset)

        combined_posterior_precision = smoothed_prec_x - smoothed_prec_y * shape[i] * shape[i]
        combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_x - shape[i] * smoothed_prec_y * (smoothed_mean_yi - observed_nonrepeatable[i]) ) / combined_posterior_precision
        new_xi = np.random.randn() / sqrt(combined_posterior_precision) + combined_posterior_mean

        xi_diff = new_xi - x[i]
        x[i] = new_xi
        # our predicted signal at time i has increased by xi_diff * shape[i], so the station noise will decrease.
        yi_diff = -xi_diff * shape[i]
        y[i+latent_offset] += yi_diff

        # we also need to updated our cached AR predictions to account for the new x and y values
        for j in range(i+1, i+1+model_x.n_p):
            ar_predictions_x[j-start_idx] += xi_diff * model_x.p[j-i-1]
            ar_predictions_y[j-start_idx+latent_offset] += yi_diff * model_y.p[j-i-1]


    code = """
    for(int i=start_idx; i < end_idx, i += step) {
       double smoothed_mean_xi = smoothed_mean(x, ar_predictions_x, start_idx, x_params, param_sqnorm_x, i);
       double smoothed_mean_yi = smoothed_mean(y, ar_predictions_y, start_idx+latent_offset, y_params, param_sqnorm_y, i+latent_offset);

       double combined_posterior_precision = smoothed_prec_x - smoothed_prec_y * shape(i) * shape(i);
       double combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_x - shape(i) * smoothed_prec_y * (smoothed_mean_yi - observed_nonrepeatable(i)) ) / combined_posterior_precision;
       double new_xi = randn() / sqrt(combined_posterior_precision) + combined_posterior_mean;

       double xi_diff = new_xi = x(i);
       x(i) = new_xi;
       // our predicted signal at time i has increased by xi_diff * shape[i], so the station noise will decrease.
       yi_diff = -xi_diff * shape(i);
       y(i+latent_offset) += yi_diff;

        # we also need to updated our cached AR predictions to account for the new x and y values
        for (int j=i+1; j <  i+1+x_nparams; ++j) {
            ar_predictions_x(j-start_idx) += xi_diff * x_params(j-i-1);
        }
        for (int j=i+1; j <  i+1+y_nparams; ++j) {
            ar_predictions_y(j-start_idx+latent_offset) += yi_diff * y_params(j-i-1);
        }
    }

    """
    # (start_idx, end_idx, step, x, y, ar_predictions_x, ar_predictions_y, latent_offset, x_params, y_params, x_nparams, y_nparams, param_sqnorm_x, param_sqnorm_y, observed_nonrepeatable )


# given:
#   d: observed data
#   pred: expected values from the AR forward model (not necessarily aligned with the beginning of d)
#   pred_start: index into d of the first prediction
#   p: AR params
#   p_sqnorm: precomputed squared-L2-norm of p
#   i: index into d for which to compute the smoothing distribution
#
# returns:
#   smoothed_mean: mean of the Gaussian smoothing distribution (note the variance is constant and can be precomputed)
def smoothed_mean(d, pred, pred_start, p, p_sqnorm, i):

    n_p = len(p)

    # first compute the z-value for each relevant index.
    # this is the residual (d[t] - pred[t]), 
    # but removing the contribution of d[i] to the predicted value.
    accum_mean = 0
    for t in range(i+1, i+1 + n_p):
        z = (d[t] - pred[t-pred_start]) - d[i] * p[t-i-1]
        accum_mean += z * p[t-i-1]
    
    smoothed_mean = ( accum_mean + pred[i-pred_start] ) / param_sqnorm
    return smoothed_mean


    support = """

    double smoothed_mean(blitz::Array<double, 1> d, blitz::Array<double, 1> pred, int pred_start, blitz::Array<double, 1> p, double p_sqnorm, int i) {
           n_p = len(p); // TODO: can we do this?
           double accum_mean = 0;
           for (int i=i+1; t < i+1+n_p; ++t) {
               double z = (d(t) - pred(t-pred_start)) - d(i) * p(t-i-1);
               accum_mean += z * p(t-i-1);
           }
           return ( accum_mean + pred(i-pred_start) ) / param_sqnorm;
    }

    """

def resample_gibbs():
