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

def prepare_gibbs_sweep(latent, start_idx=None, end_idx=None, step=1, target_signal=None):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    child_wn = list(latent.children)[0]
    model_x = latent.arwm
    model_y = child_wn.nm

    (x, shape, repeatable) = latent.get_signal_components()

    gibbs_start_idx = start_idx if start_idx else 0
    gibbs_end_idx = end_idx if end_idx else len(x)

    # load extra signal so that we have the necessary data for AR
    # filtering and smoothing
    padded_x_start_idx = gibbs_start_idx - model_x.p
    padded_x_end_idx = gibbs_end_idx + model_x.p
    #padded_x_len = padded_x_end_idx - padded_x_start_idx
    clipped_x_start_idx = max(padded_x_start_idx, 0)
    clipped_x_end_idx = min(padded_x_end_idx, len(x))


    x = x[clipped_x_start_idx:clipped_x_end_idx]
    shape = shape[clipped_x_start_idx:clipped_x_end_idx]
    repeatable = repeatable[clipped_x_start_idx:clipped_x_end_idx]

    # make sure we don't modify the cached latent wiggle
    x = np.array(x, copy=True)

    # offset between the latent and observed signals.
    # if we have an latent-based index, we add this quantity to get an obs-based index
    latent_offset = child_wn.arrival_start_idx(latent.eid, latent.phase)


    padded_y_start_idx = gibbs_start_idx - model_y.p
    padded_y_end_idx = gibbs_end_idx + model_y.p
    clipped_y_start_idx = max(0, latent_offset + padded_y_start_idx)
    clipped_y_end_idx = min(child_wn.npts, latent_offset + padded_y_end_idx)
    clipped_y_len = clipped_y_end_idx - clipped_y_start_idx

    # if we have an index into an clipped_x vector, this is the
    # correction to get an index into a clipped_y vector
    yx_offset =  clipped_x_start_idx - (clipped_y_start_idx - latent_offset)

    predicted_signal = child_wn.assem_signal(start_idx=clipped_y_start_idx, end_idx=clipped_y_end_idx)
    observed_signal = child_wn.get_value()[clipped_y_start_idx:clipped_y_end_idx]
    y = observed_signal - predicted_signal
    y = y.data # TODO: support masking as missing data

    x -= model_x.c
    y -= model_y.c

    have_target = int(target_signal is not None)
    if have_target:
        assert(len(target_signal) == gibbs_end_idx - gibbs_start_idx)
        target_wiggle = latent.compute_empirical_wiggle(env=target_signal, start_idx=gibbs_start_idx)
        target_wiggle -= model_x.c
    if not have_target:
        target_wiggle = np.zeros((0,));

    # the region of indices where x and y intersect, in latent_node-based indices
    clipped_y_start_LNidx = clipped_y_start_idx - latent_offset
    clipped_y_end_LNidx = clipped_y_end_idx - latent_offset

    isect_start = max(clipped_x_start_idx, clipped_y_start_LNidx)
    isect_end = min(clipped_x_end_idx, clipped_y_end_LNidx)
    # to convert from latent-node indexing to padded-x indexing, we subtract clipped_x_start_idx
    # to convert from latent_node indexing to padded-y indexing, we subtract padded_y_start_idx

    # we also precompute the following quantity, which determines the linear relationship between y and x.
    # (this is the quantity C_i from the googledoc derivation)
    # this is really [ (unexplained obs signal) - shape*(repeatable wiggle + model_x.c) ],
    # so it contains the remaining part of the observed signal that will need to be
    # explained as [shape*(zero-mean unrepeatable wiggle) + station noise].
    observed_nonrepeatable = shape * x
    observed_nonrepeatable[isect_start - clipped_x_start_idx : isect_end - clipped_x_start_idx] += y[isect_start - clipped_y_start_LNidx:isect_end - clipped_y_start_LNidx]

    # precompute predicted values for x's and y's
    ar_predictions_x = np.zeros((len(x),))
    ar_predictions_y = np.zeros((len(y),))
    model_x.filtered_predictions(x, 0, ar_predictions_x)
    model_y.filtered_predictions(y, 0, ar_predictions_y)

    # give some explicit variable names to some quantities for passing
    # into weave (below)
    x_error_var = model_x.em.std**2;
    y_error_var = model_y.em.std**2;
    x_params = np.asarray(model_x.params)
    y_params = np.asarray(model_y.params)
    x_mean = model_x.c

    i_start = gibbs_start_idx - clipped_x_start_idx
    i_end = gibbs_end_idx - clipped_x_start_idx

    return x, y, ar_predictions_x, ar_predictions_y, shape, repeatable, observed_nonrepeatable, latent_offset, x_params, y_params, x_error_var, y_error_var, x_mean, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end


def gibbs_sweep_python(latent, start_idx=None, end_idx=None, step=1, target_signal=None):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    x, y, ar_predictions_x, ar_predictions_y, shape, repeatable, observed_nonrepeatable, latent_offset, x_params, y_params, x_error_var, y_error_var, x_mean, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, step=step, target_signal=target_signal)

    # alignments:
    #
    # indices latent_offset, clipped_y_start_idx, clipped_y_end_idx
    #         are all relative to the beginning of the observed signal
    #         (child_wn.get_value()).
    #
    # indices start_idx, end_idx, gibbs_start_idx, gibbs_end_idx,
    #         padded_x_start_idx, padded_x_end_idx,
    #         padded_y_start_idx, padded_y_end_idx, are all relative
    #         to the beginning of latent_arrival.get_value().
    #
    # indices i_start and i_end are relative to padded_x_start_idx.
    # (they specify the bits of the padded signal that we actually
    # care about resampling / computing log-prob on)
    #
    #
    # index yx_offset gives
    #
    # index latent_offset gives padded_x_start_idx, relative to the
    #       beginning of the observed signal (wn.get_value())
    #
    #

    # objects x, shape, repeatable, observed_nonrepeatable, ar_predictions_x,
    #         all run from padded_x_start_idx to padded_x_end_idx.
    # objects y, ar_predictions_y all run from clipped_y_start_idx to clipped_y_end_idx
    # objects target_wiggle run from i_start to i_end.


    # loop over to resample each index in succession
    sample_lp = .5 * np.log(2*np.pi) * (i_end - i_start) # precompute the constant components of the gaussian densities

    # the index i is relative to padded_x_start_idx
    for i in range(i_start, i_end, step):
        smoothed_mean_xi, smoothed_prec_xi = smoothed_mean_python(x, ar_predictions_x, 0, x_params, x_error_var, i)

        # to get the correction from i to y-coords
        smoothed_mean_yi, smoothed_prec_yi = smoothed_mean_python(y, ar_predictions_y, 0, y_params, y_error_var, i+yx_offset)

        combined_posterior_precision = smoothed_prec_xi + smoothed_prec_yi * shape[i] * shape[i]
        combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_xi - shape[i] * smoothed_prec_yi * (smoothed_mean_yi - observed_nonrepeatable[i]) ) / combined_posterior_precision

        if not have_target:
            r  = np.random.randn()
            new_xi = r / np.sqrt(combined_posterior_precision) + combined_posterior_mean
        else:
            new_xi = target_wiggle[i-i_start]
            r = (new_xi - combined_posterior_mean) * np.sqrt(combined_posterior_precision)

        #if (i < 10):
        #    print "i %d r %.3f xi %.3f mean %.3f prec %.3f xm %.3f xprec %.3f ym %.3f yprec %.3f nxi %.3f" % (i, r, x[i], combined_posterior_mean, combined_posterior_precision, smoothed_mean_xi, smoothed_prec_xi, smoothed_mean_yi, smoothed_prec_yi, new_xi)

        sample_lp -= (r*r)/2.0

        xi_diff = new_xi - x[i]
        x[i] = new_xi
        # our predicted signal at time i has increased by xi_diff * shape[i], so the station noise will decrease.
        yi_diff = -xi_diff * shape[i]
        y[i+yx_offset] += yi_diff

        # we also need to updated our cached AR predictions to account for the new x and y values
        for j in range(i+1, i+1+len(x_params)):
            if (j) >= len(ar_predictions_x): break
            ar_predictions_x[j] += xi_diff * x_params[j-i-1]
        for j in range(i+1, i+1+len(y_params)):
            if (j+yx_offset) >= len(ar_predictions_y): break
            ar_predictions_y[j+yx_offset] += yi_diff * y_params[j-i-1]

    x += x_mean
    if not have_target:
        latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=clipped_x_start_idx)

    return sample_lp

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


def gibbs_sweep(latent, start_idx=None, end_idx=None, step=1, target_signal = None):


    x, y, ar_predictions_x, ar_predictions_y, shape, repeatable, observed_nonrepeatable, latent_offset, x_params, y_params, x_error_var, y_error_var, x_mean, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, step=step, target_signal=target_signal)

    # seed the C RNG from the numpy RNG, so that we can reproduce results given only a Numpy seed
    seed = np.random.randint(0, sys.maxint)

    code = """
    srand(seed);

    int x_nparams = x_params.size();
    int y_nparams = y_params.size();

    int x_len = x.size();
    int y_len = y.size();

    double smoothed_mean_xi, smoothed_prec_xi;
    double smoothed_mean_yi, smoothed_prec_yi;

    // precompute the n/2 * log(2pi) constant factor for the gaussian densities
    double sample_lp = 0.9189385332 * (i_end-i_start);
    for(int i=i_start; i < i_end; i += step) {
       smoothed_mean(x, ar_predictions_x, 0, x_params, x_error_var, i, &smoothed_mean_xi, &smoothed_prec_xi);
       smoothed_mean(y, ar_predictions_y, 0, y_params, y_error_var, i+yx_offset, &smoothed_mean_yi, &smoothed_prec_yi);

       double combined_posterior_precision = smoothed_prec_xi + smoothed_prec_yi * shape(i) * shape(i);
       double combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_xi - shape(i) * smoothed_prec_yi * (smoothed_mean_yi - observed_nonrepeatable(i)) ) / combined_posterior_precision;

       double r, new_xi;
       r = 1;
       if (have_target) {
          new_xi = target_wiggle(i-i_start);
          r = (new_xi - combined_posterior_mean) * sqrt(combined_posterior_precision);
       } else {
          r = randn();
          new_xi = r / sqrt(combined_posterior_precision) + combined_posterior_mean;
       }

        // if (i < 10) {
        //     printf("i %d r %.3f xi %.3f mean %.3f prec %.3f xm %.3f xprec %.3f ym %.3f yprec %.3f nxi %.3f\\n", i, r, x(i), combined_posterior_mean, combined_posterior_precision, smoothed_mean_xi, smoothed_prec_xi, smoothed_mean_yi, smoothed_prec_yi, new_xi);
       // }

       sample_lp -= (r*r)/2.0;

       double xi_diff = new_xi - x(i);
       x(i) = new_xi;
       // our predicted signal at time i has increased by xi_diff * shape[i], so the station noise will decrease.
       double yi_diff = -xi_diff * shape(i);
       y(i+yx_offset) += yi_diff;

        // we also need to updated our cached AR predictions to account for the new x and y values
        for (int j=i+1; j <  i+1+x_nparams; ++j) {
            if ( j >= x_len ) {
               break;
            }
            ar_predictions_x(j) += xi_diff * x_params(j-i-1);
        }
        for (int j=i+1; j <  i+1+y_nparams; ++j) {
            if ( (j + yx_offset) >= y_len ) {
               break;
            }
            ar_predictions_y(j+yx_offset) += yi_diff * y_params(j-i-1);
        }
    }

    return_val = sample_lp;
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

    sample_lp = weave.inline(code, ['i_start', 'i_end', 'step', 'x', 'y',
                                    'ar_predictions_x', 'ar_predictions_y',
                                    'shape', 'observed_nonrepeatable',
                                    'yx_offset', 'x_params', 'y_params',
                                    'x_error_var', 'y_error_var',
                                    'seed', 'have_target',
                                    'target_wiggle',
                                ],
                             type_converters=converters.blitz,
                             support_code=support, compiler='gcc')
    x += x_mean

    if not have_target:
        latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=clipped_x_start_idx)
    return sample_lp
