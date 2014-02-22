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


def prepare_gibbs_sweep(latent, start_idx=None, end_idx=None, mask_unsampled=False, target_signal=None):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    child_wn = list(latent.children)[0]
    model_x = latent.arwm
    model_y = child_wn.nm

    (empirical_wiggle, shape, repeatable) = latent.get_signal_components()

    # either ew is longer than shape, or shorter
    # if it's shorter

    gibbs_start_idx = start_idx if start_idx else 0
    gibbs_end_idx = end_idx if end_idx else len(shape)

    # load extra signal so that we have the necessary data for AR
    # filtering and smoothing
    padded_x_start_idx = gibbs_start_idx - model_x.p
    padded_x_end_idx = gibbs_end_idx + model_x.p
    #padded_x_len = padded_x_end_idx - padded_x_start_idx
    clipped_x_start_idx = max(padded_x_start_idx, 0)
    clipped_x_end_idx = min(padded_x_end_idx, len(shape))

    i_start = gibbs_start_idx - clipped_x_start_idx
    i_end = gibbs_end_idx - clipped_x_start_idx


    x = np.ones(shape.shape)
    end_copy_idx = min(clipped_x_end_idx, len(empirical_wiggle))
    x[clipped_x_start_idx:end_copy_idx] = empirical_wiggle[clipped_x_start_idx:end_copy_idx]
    shape = shape[clipped_x_start_idx:clipped_x_end_idx]
    repeatable = repeatable[clipped_x_start_idx:clipped_x_end_idx]

    # make sure we don't modify the cached latent wiggle

    x_mask = np.zeros(x.shape, dtype=bool)
    if mask_unsampled:
        x_mask[i_start:i_end] = True

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
    obs_mask = ma.getmaskarray(y)
    y_mask = np.copy(obs_mask)
    if mask_unsampled:
        y_mask[i_start+yx_offset:i_end+yx_offset] = True

    y = y.data

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

    """
    # precompute predicted values for x's and y's
    x_filtered_means = np.zeros((len(x),))
    x_filtered_vars = np.zeros((len(x),))
    y_filtered_means = np.zeros((len(y),))
    y_filtered_vars = np.zeros((len(y),))
    model_x.filtered_masked_predictions(x, x_mask, 0, x_filtered_means, x_filtered_vars)
    model_y.filtered_masked_predictions(y, y_mask, 0, y_filtered_means, y_filtered_vars)
    """

    return x, x_mask, y, y_mask, obs_mask, shape, repeatable, observed_nonrepeatable, latent_offset, model_x, model_y, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end

def gibbs_sweep_python(latent, start_idx=None, end_idx=None, reverse=False, target_signal=None, mask_unsampled=False):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    x, x_mask, y, y_mask, obs_mask, shape, repeatable, observed_nonrepeatable, latent_offset, x_model, y_model, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, mask_unsampled=mask_unsampled, target_signal=target_signal)

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

    # objects x, shape, repeatable, observed_nonrepeatable, x_filtered_means,
    #         all run from padded_x_start_idx to padded_x_end_idx.
    # objects y, y_filtered_means all run from clipped_y_start_idx to clipped_y_end_idx
    # objects target_wiggle run from i_start to i_end.

    x_params = np.asarray(x_model.params)
    y_params = np.asarray(y_model.params)




    x_filtered_means = np.zeros(x.shape)
    x_filtered_vars = np.zeros(x.shape)
    y_filtered_means = np.zeros(y.shape)
    y_filtered_vars = np.zeros(y.shape)


    if reverse:
        # ordinarily we compute the filtering distribution on-the-fly for each sample.
        # when sampling at step i, we need current filtering distributions for
        #       a) step i
        #       b) steps i+1, ..., i+n_p, since these are used in the
        #       smoothing calculation for step i.
        #       c) steps i-1, ..., i-n_p, since if any of these
        #       timesteps have masked observations, we need to use
        #       their filtered distributions instead to compute
        #       filtering distributions for the above. But if any of
        #       *these* depend on masked values, we'll need the
        #       filtering distributions of those values, and so. So
        #       really we need all filtering distributions from steps
        #       1 to i..
        #
        # if we're sampling from left-to-right, then we'll build up
        # these distributions as we go, so at each sample we just need
        # to compute the filtering distribution on that sample and on
        # its n_p successors.
        #
        # but if we're sampling from right-to-left, we need to have
        # available the precomputed filtering distributions of the
        # entire chain, to satisfy (c) above.
        # so we do that, here.
        filtered_masked_predictions(x, x_mask, x_params, x_model.em.std**2, i_start, len(x), x_filtered_means, x_filtered_vars)
        filtered_masked_predictions(y, y_mask, y_params, y_model.em.std**2, i_start+yx_offset, len(y), y_filtered_means, y_filtered_vars)


    # loop over to resample each index in succession
    sample_lp = .5 * np.log(2*np.pi) * (i_end - i_start) # precompute the constant components of the gaussian densities
    # the index i is relative to padded_x_start_idx
    for idx in range(i_start, i_end):
        if reverse:
           i = i_end - (idx - i_start) - 1
        else:
           i = idx

        filtered_masked_predictions(x, x_mask, x_params, x_model.em.std**2, i, i +1 + len(x_params), x_filtered_means, x_filtered_vars)
        if not np.all(np.isfinite(x_filtered_means)):
            print "NAN in x_filtered_means", i
            print x_params, x_model.em.std
            print x_filtered_means[i-10:i+10]
            print x[i-10:i+10]
            print x_mask[i-10:i+10]
            print x_filtered_vars[i-10:i+10]
            print y_filtered_means[i+yx_offset-10:i+yx_offset+10]

            raise Exception('nanananana')



        smoothed_mean_xi, smoothed_prec_xi = smoothed_mean_python(x, x_mask, x_filtered_means, x_filtered_vars, 0, x_params, i)
        if i+yx_offset >= len(obs_mask) or obs_mask[i+yx_offset]:
            combined_posterior_mean, combined_posterior_precision = smoothed_mean_xi, smoothed_prec_xi
        else:
            filtered_masked_predictions(y, y_mask, y_params, y_model.em.std**2, i+yx_offset, i + yx_offset + 1 + len(y_params),  y_filtered_means, y_filtered_vars)
            if not np.all(np.isfinite(y_filtered_means)):
                print "NAN in y_filtered_means", i+yx_offset
                print y_params, y_model.em.std
                print y_filtered_means[i+yx_offset-10:i+yx_offset+10]
                print x[i+yx_offset-10:i+yx_offset+10]
                print y_mask[i+yx_offset-10:i+yx_offset+10]
                print y_fi+yx_offsetltered_vars[i+yx_offset-10:i+10]
                print x_filtered_means[i-10:i+10]


            # to get the correction from i to y-coords
            smoothed_mean_yi, smoothed_prec_yi = smoothed_mean_python(y, y_mask, y_filtered_means, y_filtered_vars, 0, y_params, i+yx_offset)

            combined_posterior_precision = smoothed_prec_xi + smoothed_prec_yi * shape[i] * shape[i]
            combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_xi - shape[i] * smoothed_prec_yi * (smoothed_mean_yi - observed_nonrepeatable[i]) ) / combined_posterior_precision


        if not have_target:
            r  = np.random.randn()
            new_xi = r / np.sqrt(combined_posterior_precision) + combined_posterior_mean
        else:
            new_xi = target_wiggle[i-i_start]
            r = (new_xi - combined_posterior_mean) * np.sqrt(combined_posterior_precision)

        #if np.abs(0.87542302 - x_params[1]) < .0001:
            #print "i %d r %.3f xi %.3f mean %.3f prec %.3f xm %.3f xprec %.3f ym %.3f yprec %.3f fym %.3f fyp %.3f nxi %.3f" % (i, r, x[i], combined_posterior_mean, combined_posterior_precision, smoothed_mean_xi, smoothed_prec_xi, smoothed_mean_yi, smoothed_prec_yi, y_filtered_means[i+yx_offset], 1.0/y_filtered_vars[i+yx_offset], new_xi)

        sample_lp -= (r*r)/2.0

        # IF we had masked_unsampled, then our current predictive means and vars were using a filtered value for this timestep, and they had higher vars than necessary.
        # this means that to update the mean, we need the difference between the filtered value and the sampled value (NOT between the true value and sampled value)
        # and to update vars, we subtract out the relevant filtering variance term.

        # also, we need to UNMASK x and y at this timestep
        # and we need to CALCULATE the new y
        x_mask[i] = False
        xi_diff = new_xi - x[i]
        x[i] = new_xi

        if i+yx_offset < len(obs_mask) and not obs_mask[i+yx_offset]:
            y_mask[i+yx_offset] = False

            # we want y = obs - rw*shape - x*shape - y_model.c
            # and we have obs_nonrepeatable = (obs - y_model.c) - shape*(repeatable wiggle + model_x.c)
            new_yi = observed_nonrepeatable[i] - shape[i]*new_xi;
            y[i+yx_offset] = new_yi


    x += x_model.c

    final_x = np.copy(x)

    if not have_target:
        if start_idx is None and end_idx is None:
            latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=0, set_signal_length=True)
        else:
            latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=clipped_x_start_idx)
    return sample_lp

# given:
#   d: observed data
#   m: boolean mask array, aligned with d
#   pred: expected values from the AR forward model (not necessarily aligned with the beginning of d)
#   pred_var: filtering variances from AR forward model (these should be constant except for missing data)
#   pred_start: index into d of the first prediction
#   p: AR params
#   i: index into d for which to compute the smoothing distribution
#
# returns:
#   smoothed_mean: mean of the Gaussian smoothing distribution (note the variance is constant and can be precomputed)
def smoothed_mean_python(d, m, pred, pred_var, pred_start, p, i):
    n_p = len(p)

    # first compute the z-value for each relevant index.
    # this is the residual (d[t] - pred[t]),
    # but removing the contribution of d[i] to the predicted value.


    accum_mean = pred[i-pred_start]
    accum_param_sqnorm = 1
    accum_var = pred_var[i-pred_start]

    for t in range(i+1, i+1 + n_p):
        if (t - pred_start) >= len(pred) or t >= len(d):
            break

        if m[t]: continue

        z = (d[t] - pred[t-pred_start]) + d[i] * p[t-i-1]
        accum_mean += z * p[t-i-1]
        accum_param_sqnorm += p[t-i-1]**2
        accum_var += p[t-i-1]**2 * pred_var[t-pred_start]

        if np.isnan(z):
            import pdb; pdb.set_trace()


    if accum_param_sqnorm == 0:
        import pdb; pdb.set_trace()


    smoothed_mean = ( accum_mean ) / accum_param_sqnorm
    smoothed_prec = (accum_param_sqnorm  / accum_var) * accum_param_sqnorm

    return smoothed_mean, smoothed_prec


def gibbs_sweep(latent, start_idx=None, end_idx=None, reverse=False, mask_unsampled=False, target_signal = None):

    x, x_mask, y, y_mask, obs_mask, shape, repeatable, observed_nonrepeatable, latent_offset, x_model, y_model, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, mask_unsampled=mask_unsampled, target_signal=target_signal)

    x_filtered_means = np.zeros(x.shape)
    x_filtered_vars = np.zeros(x.shape)
    y_filtered_means = np.zeros(y.shape)
    y_filtered_vars = np.zeros(y.shape)


    x_params = np.asarray(x_model.params)
    y_params = np.asarray(y_model.params)
    x_mean = x_model.c
    x_var = x_model.em.std**2
    y_var = y_model.em.std**2

    reverse = int(reverse)

    if reverse:
        # ordinarily we compute the filtering distribution on-the-fly for each sample.
        # when sampling at step i, we need current filtering distributions for
        #       a) step i
        #       b) steps i+1, ..., i+n_p, since these are used in the
        #       smoothing calculation for step i.
        #       c) steps i-1, ..., i-n_p, since if any of these
        #       timesteps have masked observations, we need to use
        #       their filtered distributions instead to compute
        #       filtering distributions for the above. But if any of
        #       *these* depend on masked values, we'll need the
        #       filtering distributions of those values, and so. So
        #       really we need all filtering distributions from steps
        #       1 to i..
        #
        # if we're sampling from left-to-right, then we'll build up
        # these distributions as we go, so at each sample we just need
        # to compute the filtering distribution on that sample and on
        # its n_p successors.
        #
        # but if we're sampling from right-to-left, we need to have
        # available the precomputed filtering distributions of the
        # entire chain, to satisfy (c) above.
        # so we do that, here.
        filtered_masked_predictions(x, x_mask, x_params, x_model.em.std**2, i_start, len(x), x_filtered_means, x_filtered_vars)
        filtered_masked_predictions(y, y_mask, y_params, y_model.em.std**2, i_start+yx_offset, len(y), y_filtered_means, y_filtered_vars)


    # seed the C RNG from the numpy RNG, so that we can reproduce results given only a Numpy seed
    seed = np.random.randint(0, sys.maxint)

    code = """
    srand(seed);

    int x_nparams = x_params.size();
    int y_nparams = y_params.size();

    int a = 5;

    int x_len = x.size();
    int y_len = y.size();

    double smoothed_mean_xi, smoothed_prec_xi;
    double smoothed_mean_yi, smoothed_prec_yi;

    // precompute the n/2 * log(2pi) constant factor for the gaussian densities
    double sample_lp = 0.9189385332 * (i_end-i_start);
    int i = 0;
    for(int idx=i_start; idx < i_end; idx += 1) {
        if (reverse) {
           i = i_end - (idx - i_start) - 1;
        } else {
           i = idx;
        }

     filter_AR_masked(x, x_mask, x_params, x_var, i, i+1+x_nparams,   x_filtered_means, x_filtered_vars);
     smoothed_mean(x, x_mask, x_filtered_means, x_filtered_vars, 0, x_params, i, &smoothed_mean_xi, &smoothed_prec_xi);

     double combined_posterior_precision, combined_posterior_mean;
     if (i+yx_offset < y_len || obs_mask(i+yx_offset)) {
         // if we have no observation at this timestep, then the station
         // noise y provides no constraint on the latent wiggle x.
         combined_posterior_mean = smoothed_mean_xi;
         combined_posterior_precision = smoothed_prec_xi;
    } else {
         filter_AR_masked(y, y_mask, y_params, y_var, i+yx_offset, i+yx_offset+1+y_nparams,  y_filtered_means, y_filtered_vars);
         smoothed_mean(y, y_mask, y_filtered_means, y_filtered_vars, 0, y_params, i+yx_offset, &smoothed_mean_yi, &smoothed_prec_yi);
         combined_posterior_precision = smoothed_prec_xi + smoothed_prec_yi * shape(i) * shape(i);
         combined_posterior_mean = ( smoothed_mean_xi * smoothed_prec_xi - shape(i) * smoothed_prec_yi * (smoothed_mean_yi - observed_nonrepeatable(i)) ) / combined_posterior_precision;
    }

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
        // printf("i %d r %.3f xi %.3f mean %.3f prec %.3f xm %.3f xprec %.3f ym %.3f yprec %.3f nxi %.3f\\n", i, r, x(i), combined_posterior_mean, combined_posterior_precision, smoothed_mean_xi, smoothed_prec_xi, smoothed_mean_yi, smoothed_prec_yi, new_xi);
       // }

       sample_lp -= (r*r)/2.0;

       x_mask(i) = false;
       x(i) = new_xi;

       // our predicted signal at time i has increased by xi_diff * shape[i], so the station noise will decrease.

       if (i+yx_offset < x_len && !obs_mask(i+yx_offset)) {
           y_mask(i+yx_offset) = false;
           double new_yi = observed_nonrepeatable(i) - shape(i)*new_xi;
           y(i+yx_offset) = new_yi;
       }
    }

    return_val = sample_lp;
    """




    sample_lp = weave.inline(code, ['i_start', 'i_end', 'x', 'y',
                                    'x_filtered_means', 'y_filtered_means',
                                    'x_filtered_vars', 'y_filtered_vars',
                                    'shape', 'observed_nonrepeatable',
                                    'yx_offset', 'x_params', 'y_params',
                                    'x_var', 'y_var',
                                    'seed', 'have_target',
                                    'target_wiggle', 'reverse',
                                'x_mask', 'y_mask', 'obs_mask',
                                ],
                             type_converters=converters.blitz,
                             support_code=support, compiler='gcc')
    x += x_mean

    if not have_target:
        if start_idx is None and end_idx is None:
            latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=0, set_signal_length=True)
        else:
            latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=clipped_x_start_idx)

    return sample_lp


def filtered_masked_predictions(d, m, params, var, start_idx, end_idx, filtered_means, filtered_vars):

    code = """
    filter_AR_masked(d, m, params, var, start_idx, end_idx, filtered_means, filtered_vars);
    """

    weave.inline(code, ['d', 'm', 'params', 'var', 'start_idx', 'end_idx', 'filtered_means', 'filtered_vars',
                                ],
                             type_converters=converters.blitz,
                             support_code=support, compiler='gcc')



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



    void smoothed_mean(blitz::Array<double, 1> d, blitz::Array<bool, 1> m, blitz::Array<double, 1> pred, blitz::Array<double, 1> pred_var, int pred_start, blitz::Array<double, 1> p, int i, double * smoothed_mean, double *smoothed_prec) {
           int n_p = p.size();
           int len_pred = pred.size();
           int len_d = d.size();

           double accum_mean = pred(i-pred_start);
           double accum_param_sqnorm = 1;
           double accum_var = pred_var(i-pred_start);
           for (int t=i+1; t < i+1+n_p; ++t) {

               if ( (t - pred_start) >= len_pred  || t >= len_d) {
                   break;
               }

               if (m(t)) {
                   continue;
               }

               double z = (d(t) - pred(t-pred_start)) + d(i) * p(t-i-1);
               accum_mean += z * p(t-i-1);
               accum_param_sqnorm += p(t-i-1) * p(t-i-1);
               accum_var += p(t-i-1) * p(t-i-1) * pred_var(t-pred_start);
           }

           *smoothed_mean =  ( accum_mean ) / accum_param_sqnorm;
           *smoothed_prec = (accum_param_sqnorm / accum_var) * accum_param_sqnorm;
    }


// compute filtered mean for a fully observed AR process. (note that
// the filtering variance will just be constant, equal to the step
// variance of the process).
void filter_AR(blitz::Array<double, 1> d, blitz::Array<double, 1> p, int start_idx, blitz::Array<double, 1> destination) {

    int n = destination.size();
    int n_p = p.size();

    // t indexes into the destination array, where we assume
    // that destination(0) = d(start_idx).
    for (int t=0; t < n; ++t) {
        destination(t) = 0;
        for (int i=0; i < n_p; ++i) {
            if (t + start_idx > i ) {
               double ne = p(i) * d(t-i-1 + start_idx);
               destination(t) += ne;
            }
        }
    }
}

// compute filtered mean and variance for an AR process, with unobserved entries specified by a mask m
void filter_AR_masked(blitz::Array<double, 1> d, blitz::Array<bool, 1> m, blitz::Array<double, 1> p, double v, int start_idx, int end_idx, blitz::Array<double, 1> mean_destination, blitz::Array<double, 1> var_destination) {

    int n = mean_destination.size();
    int n_p = p.size();

    if (end_idx > n) {
       end_idx = n;
    }


    for (int t=start_idx; t < end_idx; ++t) {
        mean_destination(t) = 0;
        var_destination(t) = v;

        // i indexes into the parameters p.
        // When computing the filtered mean at destination(t),
        // p(i) will be the coefficient for d(t- i-1).
        for (int i=0; i < n_p; ++i) {

            // if this observation is prior to the beginning of available signal
            // (so we don't have a filtering distribution for it),
            // then just assume it's 0 with high variance
            if (t <= i) {
                var_destination(t) += 10 * p(i)*p(i) * v;
            }

            // if this observation is masked:
            // replace with its filtered value
            // and increase variance accordingly.
            else if (m(t-i-1)) {

                mean_destination(t) += p(i) * mean_destination(t-i-1);
                var_destination(t) += p(i)*p(i) * var_destination(t-i-1);

            // otherwise, do the standard filtering calculation and
            // leave variance untouched
            } else {
                  mean_destination(t) += p(i) * d(t-i-1);
            }
        }
    }
}
"""
