import os
import errno
import sys
import time
import traceback
import numpy as np
import numpy.ma as ma
import scipy
import re

import scipy.weave as weave
from scipy.weave import converters

from sigvisa import Sigvisa
from sigvisa.models.latent_arrival import LatentArrivalNode


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

def prepare_gibbs_sweep(latent, start_idx=None, end_idx=None, step=1):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    (x, shape, repeatable) = latent._empirical_wiggle(return_components=True)

    padded_repeatable = np.zeros(x.shape)
    padded_repeatable[:len(repeatable)] = repeatable
    # this should use the np.pad function, but not available in numpy 1.6

    start_idx = start_idx if start_idx else 0
    end_idx = end_idx if end_idx else len(x)

    if step < 0:
        step = np.abs(step)
        start_idx, end_idx = start_idx, end_idx

    child_wn = list(latent.children)[0]
    predicted_signal = child_wn.assem_signal()
    observed_signal = child_wn.get_value()
    y = observed_signal - predicted_signal
    # TODO: don't touch the entire observed signal: we only need to compute these quantities for the region we're reasmpling
    y = y.data # TODO: support masking as missing data


    # offset between the latent and observed signals
    latent_offset = child_wn.arrival_start_idx(latent.eid, latent.phase)

    model_x = latent.arwm
    model_y = child_wn.nm
    x -= model_x.c
    y -= model_y.c

    # we also precompute the following quantity, which determines the linear relationship between y and x.
    # (this is the quantity C_i from the googledoc derivation)
    # this is really [ (unexplained obs signal) - shape*(repeatable wiggle + model_x.c) ],
    # so it contains the remaining part of the observed signal that will need to be
    # explained as [shape*(zero-mean unrepeatable wiggle) + station noise]
    observed_nonrepeatable = y[latent_offset: latent_offset + len(x)] + shape * x


    # precompute predicted values for x's and y's (no need for residuals since these are constant time to compute at a given index so we don't save by precomputing)
    resample_len = end_idx - start_idx
    ar_predictions_x = np.zeros((resample_len,))
    ar_predictions_y = np.zeros((resample_len,))
    model_x.filtered_predictions(x, start_idx, ar_predictions_x)
    model_y.filtered_predictions(y, start_idx+latent_offset, ar_predictions_y)

    # give some explicit variable names to some quantities for passing
    # into weave (below)
    x_error_var = model_x.em.std**2;
    y_error_var = model_y.em.std**2;
    x_params = np.asarray(model_x.params)
    y_params = np.asarray(model_y.params)
    x_mean = model_x.c

    return start_idx, end_idx, step, x, y, ar_predictions_x, ar_predictions_y, shape, repeatable, observed_nonrepeatable, latent_offset, x_params, y_params, x_error_var, y_error_var, x_mean, resample_len


def gibbs_sweep_python(latent, start_idx=None, end_idx=None, step=1):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    start_idx, end_idx, step, x, y, ar_predictions_x, ar_predictions_y, shape, repeatable, observed_nonrepeatable, latent_offset, x_params, y_params, x_error_var, y_error_var, x_mean, resample_len = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, step=step)

    # loop over to resample each index in succession
    for i in range(start_idx, end_idx, step):
        smoothed_mean_xi, smoothed_prec_xi = smoothed_mean_python(x, ar_predictions_x, start_idx, x_params, x_error_var, i)
        smoothed_mean_yi, smoothed_prec_yi = smoothed_mean_python(y, ar_predictions_y, start_idx+latent_offset, y_params, y_error_var, i+latent_offset)

        combined_posterior_precision = smoothed_prec_xi + smoothed_prec_yi * shape[i] * shape[i]
        combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_xi - shape[i] * smoothed_prec_yi * (smoothed_mean_yi - observed_nonrepeatable[i]) ) / combined_posterior_precision
        new_xi = np.random.randn() / np.sqrt(combined_posterior_precision) + combined_posterior_mean


        xi_diff = new_xi - x[i]
        x[i] = new_xi
        # our predicted signal at time i has increased by xi_diff * shape[i], so the station noise will decrease.
        yi_diff = -xi_diff * shape[i]
        y[i+latent_offset] += yi_diff

        # we also need to updated our cached AR predictions to account for the new x and y values
        for j in range(i+1, i+1+len(x_params)):
            if (j - start_idx) >= len(ar_predictions_x): break
            ar_predictions_x[j-start_idx] += xi_diff * x_params[j-i-1]
        for j in range(i+1, i+1+len(y_params)):
            if (j - start_idx) >= len(ar_predictions_y): break
            ar_predictions_y[j-start_idx] += yi_diff * y_params[j-i-1]

    x += x_mean
    latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable)

# given:
#   d: observed data
#   pred: expected values from the AR forward model (not necessarily aligned with the beginning of d)
#   pred_start: index into d of the first prediction
#   p: AR params
#   p_var: step variance of AR model
#   i: index into d for which to compute the smoothing distribution
#
# returns:
#   smoothed_mean: mean of the Gaussian smoothing distribution (note the variance is constant and can be precomputed)
def smoothed_mean_python(d, pred, pred_start, p, p_var, i):

    n_p = len(p)

    # first compute the z-value for each relevant index.
    # this is the residual (d[t] - pred[t]),
    # but removing the contribution of d[i] to the predicted value.
    accum_mean = pred[i-pred_start]
    accum_param_sqnorm = 1
    for t in range(i+1, i+1 + n_p):

        if (t - pred_start) >= len(pred):
            break

        z = (d[t] - pred[t-pred_start]) + d[i] * p[t-i-1]
        accum_mean += z * p[t-i-1]
        accum_param_sqnorm += p[t-i-1]**2

    smoothed_mean = ( accum_mean ) / accum_param_sqnorm
    smoothed_prec = accum_param_sqnorm / p_var
    return smoothed_mean, smoothed_prec


def gibbs_sweep(latent, start_idx=None, end_idx=None, step=1):


    start_idx, end_idx, step, x, y, ar_predictions_x, ar_predictions_y, shape, repeatable, observed_nonrepeatable, latent_offset, x_params, y_params, x_error_var, y_error_var, x_mean, resample_len = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, step=step)

    # seed the C RNG from the numpy RNG, so that we can reproduce results given only a Numpy seed
    seed = np.random.randint(0, sys.maxint)

    code = """
    srand(seed);

    int x_nparams = x_params.size();
    int y_nparams = y_params.size();

    double smoothed_mean_xi, smoothed_prec_xi;
    double smoothed_mean_yi, smoothed_prec_yi;
    for(int i=start_idx; i < end_idx; i += step) {
       smoothed_mean(x, ar_predictions_x, start_idx, x_params, x_error_var, i, &smoothed_mean_xi, &smoothed_prec_xi);
       smoothed_mean(y, ar_predictions_y, start_idx+latent_offset, y_params, y_error_var, i+latent_offset, &smoothed_mean_yi, &smoothed_prec_yi);

       double combined_posterior_precision = smoothed_prec_xi + smoothed_prec_yi * shape(i) * shape(i);
       double combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_xi - shape(i) * smoothed_prec_yi * (smoothed_mean_yi - observed_nonrepeatable(i)) ) / combined_posterior_precision;
       double r = randn();
       double new_xi = r / sqrt(combined_posterior_precision) + combined_posterior_mean;

       double xi_diff = new_xi - x(i);
       x(i) = new_xi;
       // our predicted signal at time i has increased by xi_diff * shape[i], so the station noise will decrease.
       double yi_diff = -xi_diff * shape(i);
       y(i+latent_offset) += yi_diff;

        // we also need to updated our cached AR predictions to account for the new x and y values
        for (int j=i+1; j <  i+1+x_nparams; ++j) {
            if ( (j - start_idx) >= resample_len ) {
               break;
            }
            ar_predictions_x(j-start_idx) += xi_diff * x_params(j-i-1);
        }
        for (int j=i+1; j <  i+1+y_nparams; ++j) {
            if ( (j - start_idx) >= resample_len ) {
               break;
            }
            ar_predictions_y(j-start_idx) += yi_diff * y_params(j-i-1);
        }
    }

    """

    support = """
/*
*randn.c
*Use the polar form of the Box-Muller transform to generate random numbers
*from a 0-mean standard deviation 1 normal distribution
*/
#include <stdio.h>
#include <time.h>
#include <cmath>
#include <stdlib.h>

double randn()
{
        float x1, x2, w, y1;
        do
        {
                x1 = 2.0 * (double)rand()/RAND_MAX - 1.0;
                x2 = 2.0 * (double)rand()/RAND_MAX - 1.0;
                w = x1 * x1 + x2 * x2;
        } while ( w >= 1.0 );

        w = sqrt( (-2.0 * log( w ) ) / w );
        y1 = x1 * w;
        return y1;
}



    void smoothed_mean(blitz::Array<double, 1> d, blitz::Array<double, 1> pred, int pred_start, blitz::Array<double, 1> p, double p_var, int i, double * smoothed_mean, double *smoothed_prec) {
           int n_p = p.size();
           int len_pred = pred.size();
           double accum_mean = pred(i-pred_start);
           double accum_param_sqnorm = 1;
           for (int t=i+1; t < i+1+n_p; ++t) {
               if ( (t - pred_start) >= len_pred ) {
                   break;
               }

               double z = (d(t) - pred(t-pred_start)) + d(i) * p(t-i-1);
               accum_mean += z * p(t-i-1);
               accum_param_sqnorm += p(t-i-1) * p(t-i-1);

           }
           *smoothed_mean =  ( accum_mean ) / accum_param_sqnorm;
           *smoothed_prec = accum_param_sqnorm / p_var;
    }

    """

    weave.inline(code, ['start_idx', 'end_idx', 'step', 'x', 'y',
                        'ar_predictions_x', 'ar_predictions_y',
                        'shape', 'observed_nonrepeatable',
                        'latent_offset', 'x_params', 'y_params',
                        'x_error_var', 'y_error_var',
                        'resample_len', 'seed',
                        ],
                 type_converters=converters.blitz,
                 support_code=support, compiler='gcc')
    x += x_mean
    latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable)
