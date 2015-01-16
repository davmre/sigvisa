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

import scipy.weave as weave
from scipy.weave import converters



from sigvisa.infer.ar_smoothing_stupid import *

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


def prepare_gibbs_sweep(latent, start_idx=None, end_idx=None, target_signal=None):
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
    clipped_x_end_idx = max(min(padded_x_end_idx, len(shape)), 0)

    i_start = gibbs_start_idx - clipped_x_start_idx # + 1
    i_end = gibbs_end_idx - clipped_x_start_idx

    if i_end > clipped_x_end_idx:
        raise Exception("trying to resample latent wiggle %s from idx %d to %d, but latent shape stops at idx %d" % (latent.label, gibbs_start_idx, gibbs_end_idx, clipped_x_end_idx))

    x = np.ones((clipped_x_end_idx - clipped_x_start_idx,))
    end_copy_idx = min(clipped_x_end_idx, len(empirical_wiggle))
    x[:end_copy_idx-clipped_x_start_idx] = empirical_wiggle[clipped_x_start_idx:end_copy_idx]
    shape = shape[clipped_x_start_idx:clipped_x_end_idx]
    repeatable = repeatable[clipped_x_start_idx:clipped_x_end_idx]

    # offset between the latent and observed signals.
    # if we have an latent-based index, we add this quantity to get an obs-based index
    latent_offset = child_wn.arrival_start_idx(latent.eid, latent.phase)

    padded_y_start_idx = gibbs_start_idx - model_y.p
    padded_y_end_idx = gibbs_end_idx + model_y.p
    clipped_y_start_idx = max(0, latent_offset + padded_y_start_idx)
    clipped_y_end_idx = max(min(child_wn.npts, latent_offset + padded_y_end_idx), 0)
    clipped_y_len = clipped_y_end_idx - clipped_y_start_idx

    # if we have an index into an clipped_x vector, this is the
    # correction to get an index into a clipped_y vector
    yx_offset =  clipped_x_start_idx - (clipped_y_start_idx - latent_offset)

    predicted_signal = child_wn.assem_signal(start_idx=clipped_y_start_idx, end_idx=clipped_y_end_idx)
    observed_signal = child_wn.get_value()[clipped_y_start_idx:clipped_y_end_idx]
    y = observed_signal - predicted_signal
    obs_mask = ma.getmaskarray(y)

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

    return x, y, obs_mask, shape, repeatable, observed_nonrepeatable, latent_offset, model_x, model_y, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end


def get_lp_obs(x_means, x_covs, y_means, y_covs, ix_offset, shape, observed_nonrepeatable):
    n = x_means.shape[0]

    for i in range(n):
        filtered_mean_obs = shape[i] * x_means[i, 0] + y_means[i+ix_offset, 0]
        filtered_var_obs = shape[i]*shape[i]*x_covs[i, 0,0] + y_covs[i+yx_offset, 0, 0]

        log_gaussian = lambda v, mu, var : -.5 * (np.log(2 * np.pi * var) + (v - mu)**2/var)
        local_obs_lp = log_gaussian(observed_nonrepeatable[i], filtered_mean_obs, filtered_var_obs)
        all_obs_lp += local_obs_lp
        print "step %d local_lp %f all_lp %f " % (i, local_obs_lp, all_obs_lp)
    return all_obs_lp

def gibbs_sweep_python(latent, start_idx=None, end_idx=None, target_signal=None, debug=False, stationary=False, adjust_latent_length=False):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    x, y, obs_mask, shape, repeatable, observed_nonrepeatable, latent_offset, x_model, y_model, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, target_signal=target_signal)

    if not adjust_latent_length:
        assert((i_end-i_start) == len(latent.get_value()))

    if len(y) == 0:
        # don't bother resampling if there's no signal
        print "WARNING: not resampling %s because no obs signal is available" % latent.label
        return 0


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


    x_mask = np.ones((len(x),), dtype=bool)
    x_mask[:i_start] = False
    x_mask[i_end:] = False
    x_mask[np.isnan(x)] = True

    y_mask = np.copy(obs_mask)

    y_mask_start = max(i_start+yx_offset, 0)
    y_mask_end = max(i_end+yx_offset, 0)
    y_mask[y_mask_start:y_mask_end] = True

    seed = np.random.randint(0, sys.maxint)

    if stationary:
        x_filtered_means, x_filtered_covs = filter_AR_stationary(x, x_mask, x_model)
        y_filtered_means, y_filtered_covs = filter_AR_stationary(y, y_mask, y_model)
    else:
        x_filtered_means, x_filtered_covs = filter_AR_stupid(x, x_mask, x_model)
        y_filtered_means, y_filtered_covs = filter_AR_stupid(y, y_mask, y_model)


    x_lambda_hat, x_lambda_squiggle, x_Lambda_hat, x_Lambda_squiggle = smooth_AR_stupid(x, x_mask, x_model, x_filtered_means, x_filtered_covs, i_end)
    y_lambda_hat, y_lambda_squiggle, y_Lambda_hat, y_Lambda_squiggle = smooth_AR_stupid(y, y_mask, y_model, y_filtered_means, y_filtered_covs, i_end+yx_offset)

    x_np = len(x_params)
    y_np = len(y_params)

    # loop over to resample each index in succession
    #sample_lp = -.5 * np.log(2*np.pi) * (i_end - i_start) # precompute the constant components of the gaussian densities
    cum_lp = 0
    obs_lp_cum = 0
    smooth_lp_cum = 0

    # the index i is relative to padded_x_start_idx
    for i in range(i_end-1, i_start-1, -1):

        smoothed_mean_x = smooth_mean_functional(x_filtered_means[i,:], x_filtered_covs[i,:,:], x_lambda_hat)
        smoothed_cov_x = smooth_cov_functional(x_filtered_covs[i,:,:], x_Lambda_hat)

        observed_y = i+yx_offset < len(obs_mask) and i+yx_offset >= 0 and not obs_mask[i+yx_offset] and shape[i] > 0

        if observed_y:

            smoothed_mean_y = smooth_mean_functional(y_filtered_means[i+yx_offset,:], y_filtered_covs[i+yx_offset,:,:], y_lambda_hat)
            smoothed_cov_y = smooth_cov_functional(y_filtered_covs[i+yx_offset,:,:], y_Lambda_hat)

            #smoothed_mean_y = np.zeros((1,))
            #smoothed_cov_y = np.ones((1,1)) * 1e100



            combined_posterior_precision = 1.0/smoothed_cov_x[0,0] + shape[i] * shape[i] / smoothed_cov_y[0,0]
            combined_posterior_mean = ( smoothed_mean_x[0] / smoothed_cov_x[0,0] - shape[i] * (smoothed_mean_y[0] - observed_nonrepeatable[i]) / smoothed_cov_y[0,0] ) / combined_posterior_precision
        else:
            smoothed_cov_y = np.zeros((1,1))
            smoothed_mean_y = np.zeros((1,))
            combined_posterior_precision = 1.0/smoothed_cov_x[0,0]
            combined_posterior_mean = smoothed_mean_x[0]


        if not have_target:
            r  = np.random.randn() #c_randn()
            new_xi = r / np.sqrt(combined_posterior_precision) + combined_posterior_mean
        else:
            new_xi = target_wiggle[i-i_start]
            r = (new_xi - combined_posterior_mean) * np.sqrt(combined_posterior_precision)

        if np.isnan(new_xi):
            new_xi = 0
            entropy = -.5 * np.log(2*np.pi * np.e / combined_posterior_precision)
            lp = entropy
            cum_lp += entropy
        else:
            lp = -.5 * (np.log(2 * np.pi * 1.0/combined_posterior_precision) + r * r)
            cum_lp += lp


        log_gaussian = lambda v, mu, var : -.5 * (np.log(2 * np.pi * var) + (v - mu)**2/var)
        smooth_lp_x = log_gaussian(new_xi, smoothed_mean_x[0], smoothed_cov_x[0,0])
        if observed_y:
            smooth_lp_y = log_gaussian(observed_nonrepeatable[i] - shape[i] * new_xi, smoothed_mean_y[0], smoothed_cov_y[0,0])
            #if smooth_lp_y < -20:
            #    import pdb; pdb.set_trace()
            smooth_lp_cum += smooth_lp_x + smooth_lp_y
            obs_lp = log_gaussian(observed_nonrepeatable[i] - smoothed_mean_y[0], shape[i] * smoothed_mean_x[0], shape[i] * shape[i] * smoothed_cov_x[0,0] + smoothed_cov_y[0,0])
            obs_lp_cum += obs_lp

            if debug:
                print "lp %f cum_lp %f smooth_lp_x %f smooth_lp_y %f smooth_lp_cum %f obs_lp %f obs_lp_cum %f" % (lp, cum_lp, smooth_lp_x, smooth_lp_y, smooth_lp_cum, obs_lp, obs_lp_cum)

        # IF we had masked_unsampled, then our current predictive means and vars were using a filtered value for this timestep, and they had higher vars than necessary.
        # this means that to update the mean, we need the difference between the filtered value and the sampled value (NOT between the true value and sampled value)
        # and to update vars, we subtract out the relevant filtering variance term.

        # also, we need to UNMASK x and y at this timestep
        # and we need to CALCULATE the new y
        xi_diff = new_xi - x[i]
        x[i] = new_xi
        update_Lambda_squiggle(x_Lambda_hat, x_Lambda_squiggle, x_filtered_covs[i,:,:])
        update_lambda_squiggle(x_lambda_hat, x_lambda_squiggle, x_filtered_covs[i,:,:], new_xi - x_filtered_means[i,0])

        if observed_y:
            # we want y = obs - rw*shape - x*shape - y_model.c
            # and we have obs_nonrepeatable = (obs - y_model.c) - shape*(repeatable wiggle + model_x.c)
            new_yi = observed_nonrepeatable[i] - shape[i] * new_xi
            y[i+yx_offset] = new_yi

            update_Lambda_squiggle(y_Lambda_hat, y_Lambda_squiggle, y_filtered_covs[i+yx_offset,:,:])
            update_lambda_squiggle(y_lambda_hat, y_lambda_squiggle, y_filtered_covs[i+yx_offset,:,:], new_yi - y_filtered_means[i+yx_offset,0])
        else:
            new_yi = 0
            y_lambda_squiggle[:] = y_lambda_hat
            y_Lambda_squiggle[:,:] = y_Lambda_hat

        if debug:
            try:
                print "time %d filtered_mean_x %f filtered_cov_x %f filtered_mean_y %f filtered_cov_y %f mean_x %f cov_x %f mean_y %f cov_y %f mean_combined %f cov_combined %f sampled_x %f sampled_y %f" % (i, x_filtered_means[i,0], x_filtered_covs[i,0,0], y_filtered_means[i+yx_offset, 0], y_filtered_covs[i+yx_offset,0,0], smoothed_mean_x[0], smoothed_cov_x[0,0], smoothed_mean_y[0], smoothed_cov_y[0,0], combined_posterior_mean, 1.0/combined_posterior_precision, new_xi, new_yi)
                print ""
            except:
                print "time %d filtered_mean_x %f filtered_cov_x %f filtered_mean_y OOR filtered_cov_y OOR mean_x %f cov_x %f mean_y OOR cov_y OOR mean_combined %f cov_combined %f sampled_x %f sampled_y %f y_idx %d yshape %s" % (i, x_filtered_means[i,0], x_filtered_covs[i,0,0], smoothed_mean_x[0], smoothed_cov_x[0,0], combined_posterior_mean, 1.0/combined_posterior_precision, new_xi, new_yi, i+yx_offset, y_filtered_means.shape)
        if np.isnan(combined_posterior_mean) or smoothed_cov_x[0,0] < 0 or smoothed_cov_y[0,0] < 0 or 1.0/combined_posterior_precision < 0 or np.abs(new_xi) > 1e80 or np.abs(new_yi) > 1e80:
            raise Exception('something fucked up')


        update_Lambda_hat(x_Lambda_hat, x_Lambda_squiggle, x_params)
        update_lambda_hat(x_lambda_hat, x_lambda_squiggle, x_params)
        update_Lambda_hat(y_Lambda_hat, y_Lambda_squiggle, y_params)
        update_lambda_hat(y_lambda_hat, y_lambda_squiggle, y_params)


    x += x_model.c

    final_x = np.copy(x)

    if not have_target:
        if start_idx is None and end_idx is None:
               latent.set_nonrepeatable_wiggle(x, start_idx=0, set_signal_length=adjust_latent_length)
        else:
            latent.set_nonrepeatable_wiggle(x, start_idx=clipped_x_start_idx)

    print "cum_lp", cum_lp, "obs_lp_cum", obs_lp_cum
    return cum_lp + obs_lp_cum


def c_randn():
    code = "return_val = randn();"

    support = """
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
"""

    return weave.inline(code, ['seed'],
                        type_converters=converters.blitz,
                        support_code=support, compiler='gcc')

def gibbs_sweep_forward(latent, start_idx=None, end_idx=None, target_signal=None, debug=False, stationary=False, adjust_latent_length=False):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    x, y, obs_mask, shape, repeatable, observed_nonrepeatable, latent_offset, x_model, y_model, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, target_signal=target_signal)

    if not adjust_latent_length:
        assert((i_end-i_start) == len(latent.get_value()))

    if len(y) == 0:
        # don't bother resampling if there's no signal
        print "WARNING: not resampling %s because no obs signal is available" % latent.label
        return 0


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


    x_mask = np.ones((len(x),), dtype=bool)
    x_mask[:i_start] = False
    x_mask[i_end:] = False

    y_mask = np.copy(obs_mask)

    y_mask_start = max(i_start+yx_offset, 0)
    y_mask_end = max(i_end+yx_offset, 0)
    y_mask[y_mask_start:y_mask_end] = True

    x_np = len(x_params)
    y_np = len(y_params)

    # loop over to resample each index in succession
    sample_lp = -.5 * np.log(2*np.pi) * (i_end - i_start) # precompute the constant components of the gaussian densities

    cum_lp = 0
    cum_lp_y = 0
    cum_diff = 0

    unnorm_lp = 0
    filter_x_cum = 0
    filter_y_cum = 0

    obs_lp_cum = 0


    all_obs_lp = 0


    # the index i is relative to padded_x_start_idx
    for i in range(i_start, i_end):

        filtered_mean_x, filtered_var_x  = ar_filter(i, x, x_params, x_model.em.std)
        filtered_mean_y, filtered_var_y  = ar_filter(i+yx_offset, y, y_params, y_model.em.std)

        #filtered_mean_y = 0.0
        #filtered_var_y = 1.0e100
        #filtered_mean_x = 0.0
        #filtered_var_x = 1.0e100

        combined_posterior_precision = 1.0/filtered_var_x + shape[i] * shape[i] / filtered_var_y
        combined_posterior_mean = ( filtered_mean_x / filtered_var_x - shape[i] * (filtered_mean_y - observed_nonrepeatable[i]) / filtered_var_y ) / combined_posterior_precision

        x_posterior_precision_given_y = shape[i] * shape[i] / filtered_var_y
        x_posterior_mean_given_y = - (filtered_mean_y - observed_nonrepeatable[i]) / shape[i]

        combined_posterior_mean_y = observed_nonrepeatable[i] - combined_posterior_mean * shape[i]
        combined_posterior_var_y =  (shape[i] * shape[i]) / combined_posterior_precision

        #if debug:
        #    print "x %f (y %f - obsnr %f) / shape %f = %f" % (x[i], y[i+yx_offset], observed_nonrepeatable[i], shape[i], -(y[i+yx_offset] - observed_nonrepeatable[i]) / shape[i])
        #    print "y %f obsnr %f - x %f * shape %f = %f" % (y[i+yx_offset], observed_nonrepeatable[i], x[i], shape[i], observed_nonrepeatable[i] - x[i] * shape[i])

        if not have_target:
            r  = np.random.randn()
            new_xi = r / np.sqrt(combined_posterior_precision) + combined_posterior_mean
        else:
            new_xi = target_wiggle[i-i_start]
            r = (new_xi - combined_posterior_mean) * np.sqrt(combined_posterior_precision)

        sample_lp += (np.log(combined_posterior_precision) - r*r) /2.0
        lp = -.5 * (np.log(2 * np.pi * 1.0/combined_posterior_precision) + r * r)


        cum_lp += lp



        # also, we need to UNMASK x and y at this timestep
        # and we need to CALCULATE the new y
        xi_diff = new_xi - x[i]
        x[i] = new_xi

        new_yi = observed_nonrepeatable[i] - shape[i]*new_xi;
        y[i+yx_offset] = new_yi

        r_y = (new_yi - combined_posterior_mean_y) / np.sqrt(combined_posterior_var_y)
        lp_y = -.5 * (np.log(2 * np.pi * combined_posterior_var_y) + r_y * r_y)
        cum_lp_y += lp_y
        cum_diff += .5 * (np.log(2 * np.pi * combined_posterior_var_y) - np.log(2 * np.pi * 1.0 / combined_posterior_precision) )


        log_gaussian = lambda v, mu, var : -.5 * (np.log(2 * np.pi * var) + (v - mu)**2/var)
        filter_lp_x = log_gaussian(x[i], filtered_mean_x, filtered_var_x)
        filter_lp_y = log_gaussian(y[i+yx_offset], filtered_mean_y, filtered_var_y)
        filter_x_cum += filter_lp_x
        filter_y_cum += filter_lp_y
        unnorm_lp += filter_lp_x + filter_lp_y

        obs_lp  = log_gaussian(observed_nonrepeatable[i] - filtered_mean_y, shape[i] * filtered_mean_x, shape[i] * shape[i] * filtered_var_x + filtered_var_y)
        obs_lp_cum += obs_lp
        # N(z; 0, sqrt(s^2 sigma1^2 + sigma2^2))

        if debug:
            print "t %d obs %f shape %f filtered x %f std %f filtered_y %f std %f filtered_x_given_y mean %f std %f combined_x_mean %f combined_x_std %f sampled_x %f sampled_y %f lp %f cum_lp %f cpmy %f cpstd_y %f lp_y %f r %f r_y %f cum_lp_y %f cum_diff %f filter_x_lp %f filter_y_lp %f filter_lp %f filter_x_cum %f filter_y_cum %f unnorm_lp %f obs_lp %f obs_lp_cum %f bayes_lp %f" % (i, observed_nonrepeatable[i], shape[i], filtered_mean_x, np.sqrt(filtered_var_x), filtered_mean_y, np.sqrt(filtered_var_y), x_posterior_mean_given_y, np.sqrt(1.0/x_posterior_precision_given_y), combined_posterior_mean, np.sqrt(1.0/combined_posterior_precision), new_xi, new_yi, lp, cum_lp, combined_posterior_mean_y, np.sqrt(combined_posterior_var_y), lp_y, r, r_y, cum_lp_y, cum_diff, filter_lp_x, filter_lp_y, filter_lp_x + filter_lp_y, filter_x_cum, filter_y_cum, unnorm_lp, obs_lp, obs_lp_cum, cum_lp+obs_lp_cum)



    x += x_model.c

    final_x = np.copy(x)

    if not have_target:
        if start_idx is None and end_idx is None:
               latent.set_nonrepeatable_wiggle(x, start_idx=0, set_signal_length=adjust_latent_length)
        else:
            latent.set_nonrepeatable_wiggle(x, start_idx=clipped_x_start_idx)
    return cum_lp + obs_lp_cum

def ar_filter(idx, x, params, std):
    n_p = len(params)
    filtered_mean = 0
    filtered_var = std**2
    for i in range(n_p):
        if idx-i-1 < 0:
            filtered_var += params[i]
            filtered_var = 1.0e4
            continue
        filtered_mean += x[idx-i-1] * params[i]
    return filtered_mean, filtered_var
