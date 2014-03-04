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
    clipped_x_end_idx = min(padded_x_end_idx, len(shape))

    i_start = gibbs_start_idx - clipped_x_start_idx + 1
    i_end = gibbs_end_idx - clipped_x_start_idx

    if i_end > clipped_x_end_idx:
        raise Exception("trying to resample latent wiggle %s from idx %d to %d, but latent shape stops at idx %d" % (latent.label, gibbs_start_idx, gibbs_end_idx, clipped_x_end_idx))

    x = np.ones((clipped_x_end_idx - clipped_x_start_idx,))
    end_copy_idx = min(clipped_x_end_idx, len(empirical_wiggle))
    x[:end_copy_idx-clipped_x_start_idx] = empirical_wiggle[clipped_x_start_idx:end_copy_idx]
    x[0] = 1.0
    shape = shape[clipped_x_start_idx:clipped_x_end_idx]
    repeatable = repeatable[clipped_x_start_idx:clipped_x_end_idx]

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

def gibbs_sweep_c(latent, start_idx=None, end_idx=None, target_signal=None, debug=False, stationary=False):
    # see 'sigvisa scratch' from feb 5, 2014 for a derivation of some of this.

    x, y, obs_mask, shape, repeatable, observed_nonrepeatable, latent_offset, x_model, y_model, have_target, target_wiggle, clipped_x_start_idx, yx_offset, i_start, i_end = prepare_gibbs_sweep(latent, start_idx=start_idx, end_idx=end_idx, target_signal=target_signal)

    # seed the C RNG from the numpy RNG, so that we can reproduce results given only a Numpy seed
    seed = np.random.randint(0, sys.maxint)

    debug = int(debug)

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
    x_np = len(x_params)
    y_np = len(y_params)

    x_var = float(x_model.em.std**2)
    y_var = float(y_model.em.std**2)

    x_mask = np.ones((len(x),), dtype=bool)
    x_mask[:i_start] = False
    x_mask[i_end:] = False

    y_mask = np.copy(obs_mask)
    y_mask[i_start+yx_offset:i_end+yx_offset] = True

    x_n = len(x)
    x_filtered_means = np.zeros((x_n, x_np))
    x_filtered_covs = np.zeros((x_n, x_np, x_np))
    x_filtered_covs[0,:,:] = np.eye(x_np)

    y_n = len(y)
    y_filtered_means = np.zeros((y_n, y_np))
    y_filtered_covs = np.zeros((y_n, y_np, y_np))
    y_filtered_covs[0,:,:] = np.eye(y_np)

    x_Lambda_hat = np.zeros((x_np, x_np))
    x_Lambda_squiggle = np.zeros((x_np, x_np))
    x_lambda_hat = np.zeros((x_np),)
    x_lambda_squiggle = np.zeros((x_np),)
    smoothed_mean_x = np.zeros((x_np))
    smoothed_cov_x = np.zeros((x_np, x_np))

    y_Lambda_hat = np.zeros((y_np, y_np))
    y_Lambda_squiggle = np.zeros((y_np, y_np))
    y_lambda_hat = np.zeros((y_np),)
    y_lambda_squiggle = np.zeros((y_np),)
    smoothed_mean_y = np.zeros((y_np))
    smoothed_cov_y = np.zeros((y_np, y_np))

    code = """
    srand(seed);

    int len_obs_mask = obs_mask.size();

    filter_AR(x, x_mask, x_params, x_var, x_filtered_means, x_filtered_covs, smoothed_cov_x);
    filter_AR(y, y_mask, y_params, y_var, y_filtered_means, y_filtered_covs, smoothed_cov_y);

    smooth_AR(x, x_mask, x_params, x_filtered_means, x_filtered_covs, i_end, x_lambda_hat, x_Lambda_hat, x_lambda_squiggle, x_Lambda_squiggle, smoothed_cov_x);
    smooth_AR(x, y_mask, y_params, y_filtered_means, y_filtered_covs, i_end+yx_offset, y_lambda_hat, y_Lambda_hat, y_lambda_squiggle, y_Lambda_squiggle, smoothed_cov_y);

    double sample_lp = 0.9189385332 * (i_end-i_start);
    double combined_posterior_mean, combined_posterior_precision;
    // the index i is relative to padded_x_start_idx
    for (int i=i_end-1; i >= i_start; --i) {

        smooth_mean(x_filtered_means, x_filtered_covs, i, x_lambda_hat, smoothed_mean_x);

        // passing Lambda_squiggle as a scratch variable
        smooth_cov(x_filtered_covs, i, x_Lambda_hat, x_Lambda_squiggle, smoothed_cov_x);



        if (i+yx_offset < len_obs_mask && !obs_mask(i+yx_offset)) {

            smooth_mean(y_filtered_means, y_filtered_covs, i+yx_offset, y_lambda_hat, smoothed_mean_y);
            smooth_cov(y_filtered_covs, i+yx_offset, y_Lambda_hat, y_Lambda_squiggle, smoothed_cov_y);

            combined_posterior_precision = 1.0/smoothed_cov_x(0,0) + shape(i) * shape(i) / smoothed_cov_y(0,0);
            combined_posterior_mean = ( smoothed_mean_x(0) / smoothed_cov_x(0,0) - shape(i) * (smoothed_mean_y(0) - observed_nonrepeatable(i)) / smoothed_cov_y(0,0) ) / combined_posterior_precision;
        } else {
            combined_posterior_precision = 1.0/smoothed_cov_x(0,0);
            combined_posterior_mean = smoothed_mean_x(0);
        }

        double new_xi, r;
        if (have_target) {
          new_xi = target_wiggle(i-i_start);
          r = (new_xi - combined_posterior_mean) * sqrt(combined_posterior_precision);
        } else {
          srand(i);
          r = randn();
          new_xi = r / sqrt(combined_posterior_precision) + combined_posterior_mean;
        }

        sample_lp -= (r*r)/2.0;

        x(i) = new_xi;

        update_Lambda_squiggle(x_Lambda_hat, x_Lambda_squiggle, x_filtered_covs, i);
        update_lambda_squiggle(x_lambda_hat, x_lambda_squiggle, x_filtered_covs, i, new_xi - x_filtered_means(i,0));
        double new_yi = -999;
        if (i+yx_offset < len_obs_mask && !obs_mask(i+yx_offset) ) {
            new_yi = observed_nonrepeatable(i) - shape(i)*new_xi;
            y(i+yx_offset) = new_yi;

            update_Lambda_squiggle(y_Lambda_hat, y_Lambda_squiggle, y_filtered_covs, i+yx_offset);
            update_lambda_squiggle(y_lambda_hat, y_lambda_squiggle, y_filtered_covs, i+yx_offset, new_yi - y_filtered_means(i+yx_offset,0));
        } else {
            // copy y_hats to y_squiggles.
            // note: this works because blitz++ treats assignment as a copy
            y_lambda_squiggle = y_lambda_hat;
            y_Lambda_squiggle = y_Lambda_hat;
        }

        if (debug) {
            printf("time %d filtered_mean_x %f filtered_cov_x %f filtered_mean_y %f filtered_cov_y %f mean_x %f cov_x %f mean_y %f cov_y %f mean_combined %f cov_combined %f sampled_x %f sampled_y %f\\n", i, x_filtered_means(i,0), x_filtered_covs(i,0,0), y_filtered_means(i+yx_offset, 0), y_filtered_covs(i+yx_offset,0,0), smoothed_mean_x(0), smoothed_cov_x(0,0), smoothed_mean_y(0), smoothed_cov_y(0,0), combined_posterior_mean, 1.0/combined_posterior_precision, new_xi, new_yi);
        }

        update_Lambda_hat(x_Lambda_hat, x_Lambda_squiggle, x_params);
        update_lambda_hat(x_lambda_hat, x_lambda_squiggle, x_params);
        update_Lambda_hat(y_Lambda_hat, y_Lambda_squiggle, y_params);
        update_lambda_hat(y_lambda_hat, y_lambda_squiggle, y_params); // hello world6
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



void print_array(blitz::Array<double, 1> x) {
    printf("dumping array:\\n[");
    for (int i=0; i < x.size(); ++i) {
         printf("%f, ", x(i));
    }
    printf("] \\n");
}

void print_subarray2(blitz::Array<double, 2> x, int i) {
    printf("[");
    for (int j=0; j < x.shape()(1); ++j) {
         printf("%f, ", x(i, j));
    }
    printf("] \\n");
}


void print_array2(blitz::Array<double, 2> x) {
    printf("[");
    for (int j=0; j < x.shape()(1); ++j) {
         for (int k=0; k < x.shape()(2); ++k) {
             printf("%f, ", x(j, k));
         }
         printf("\\n");
    }
    printf("] \\n");
}

void print_subarray3(blitz::Array<double, 3> x, int i) {
    printf("[");
    for (int j=0; j < x.shape()(1); ++j) {
         for (int k=0; k < x.shape()(2); ++k) {
             printf("%f, ", x(i, j, k));
         }
         printf("\\n");
    }
    printf("] \\n");
}


void update_obs(blitz::Array<double, 2> filtered_means, blitz::Array<double, 3> filtered_covs, int t, double obs) {
    int n_p = filtered_means.shape()(1);
    double r = obs - filtered_means(t,0);

    for (int i=0; i < n_p; ++i) {
        filtered_means(t+1,i) = filtered_means(t,i) + filtered_covs(t,i,0) * r/filtered_covs(t,0,0);
        for (int j=0; j < n_p; ++j) {
            filtered_covs(t+1,i,j) = filtered_covs(t,i,j) - filtered_covs(t,i,0) * filtered_covs(t,0,j)/filtered_covs(t,0,0);
        }
    }
}

void update_u(blitz::Array<double, 2> filtered_means, int t, blitz::Array<double, 1> params) {
    int n_p = params.size();
    double newval = 0;
    for (int i=0; i < n_p; ++i) {
        newval += filtered_means(t,i) * params(i);
    }
    for (int i=n_p-1; i >= 0; --i) {
        filtered_means(t, i) = filtered_means(t, i-1);
    }
    filtered_means(t,0) = newval;
}

void updateK(blitz::Array<double, 3> filtered_covs, int t, blitz::Array<double, 1> params, blitz::Array<double, 2> tmp) {
    int n_p = params.size();
    for (int i=0; i < n_p; ++i) {
        tmp(i,0) = 0;

        for (int j=0; j < n_p; ++j) {
            tmp(i,0) += filtered_covs(t,i,j) * params(j);
        }
        for (int j=1; j < n_p; ++j) {
            tmp(i,j) = filtered_covs(t,i,j-1);
        }
    }

    for (int i=0; i < n_p; ++i) {
        filtered_covs(t,0,i) = 0;
        for (int j=0; j < n_p; ++j) {
            filtered_covs(t,0,i) += tmp(j,i) * params(j);
        }
        for (int j=1; j < n_p; ++j) {
            filtered_covs(t,j,i) = tmp(j-1, i);
        }
   }
}

void filter_AR(blitz::Array<double, 1> x, blitz::Array<bool, 1> mask, blitz::Array<double, 1> params, double sigma2, blitz::Array<double, 2> filtered_means, blitz::Array<double, 3> filtered_covs, blitz::Array<double, 2> cov_tmp) {
    int n = x.size();
    int n_p = params.size();

    bool stationary_cov = false;
    bool stationary_mean = false;

    for(int t=0; t < n-1; ++t) {
        if (mask(t)) {

            for (int i=0; i < n_p; ++i) {
                filtered_means(t+1, i) = filtered_means(t, i);
                for (int j=0; j < n_p; ++j) {
                    filtered_covs(t+1, i, j) = filtered_covs(t, i, j);
                }
            }

            if (!stationary_mean) {
               update_u(filtered_means, t+1, params);
               if (fabs(filtered_means(t+1,0) - filtered_means(t,0)) < 0.00000001) {
                  stationary_mean = true;
               }
            }
            if (!stationary_cov) {
               updateK(filtered_covs, t+1, params, cov_tmp);
               filtered_covs(t+1, 0, 0) += sigma2;
               if (fabs(filtered_covs(t+1,0,0) - filtered_covs(t,0,0)) < 0.00000000000001) {
                  stationary_cov = true;
               }
            }
        } else {
          stationary_mean = false;
          stationary_cov = false;
          update_obs(filtered_means, filtered_covs, t, x(t) );
          update_u(filtered_means, t+1, params);
          updateK(filtered_covs, t+1, params, cov_tmp);
          filtered_covs(t+1, 0, 0) += sigma2;
        }
    }
}


void smooth_mean_inplace(blitz::Array<double, 2> filtered_means, blitz::Array<double, 3> filtered_covs, int t, blitz::Array<double, 1> lambda_squiggle) {
     int n_p = filtered_means.shape()[1];
     for (int i=0; i < n_p; ++i) {
         double dot = 0;
         for (int j=0; j < n_p; ++j) {
             dot += lambda_squiggle(j) * filtered_covs(t,i,j);
         }
         filtered_means(t, i) -= dot;
     }
}

void smooth_cov_inplace(blitz::Array<double, 3> filtered_covs, int t, blitz::Array<double, 2> Lambda_squiggle, blitz::Array<double, 2> tmp, blitz::Array<double, 2> tmp2) {
     int n_p = filtered_covs.shape()[1];


     // first do tmp = dot(Lambda_squiggle, filtered_covs(t,:,:)
     for (int i = 0; i < n_p; ++i ) {
         for (int j = 0; j < n_p; ++j ) {
             tmp(i,j) = 0;
             for (int k = 0; k < n_p; ++k ) {
                 tmp(i,j) += Lambda_squiggle(i,k) * filtered_covs(t, k, j);
             }
         }
     }

     // then do tmp2 = dot(Lambda_squiggle, filtered_covs(t,:,:)
     for (int i = 0; i < n_p; ++i ) {
         for (int j = 0; j < n_p; ++j ) {
             tmp2(i,j) = 0;
             for (int k = 0; k < n_p; ++k ) {
                 tmp2(i,j) += filtered_covs(t, i,k) * tmp(k,j);
             }
         }
     }

     for (int i = 0; i < n_p; ++i ) {
         for (int j = 0; j < n_p; ++j ) {
             filtered_covs(t,i,j) -= tmp2(i,j);
         }
     }

}

void smooth_mean(blitz::Array<double, 2> filtered_means, blitz::Array<double, 3> filtered_covs, int t, blitz::Array<double, 1> lambda_squiggle, blitz::Array<double, 1> smoothed_mean) {
     int n_p = filtered_means.shape()[1];
     for (int i=0; i < n_p; ++i) {
         double dot = 0;
         for (int j=0; j < n_p; ++j) {
             dot += lambda_squiggle(j) * filtered_covs(t,i,j);
         }
         smoothed_mean(i) = filtered_means(t, i) - dot;
     }
}

void smooth_cov(blitz::Array<double, 3> filtered_covs, int t, blitz::Array<double, 2> Lambda_squiggle, blitz::Array<double, 2> tmp, blitz::Array<double, 2> smoothed_cov) {
     int n_p = filtered_covs.shape()[1];
     for (int i = 0; i < n_p; ++i ) {
         for (int j = 0; j < n_p; ++j ) {
             tmp(i,j) = 0;
             for (int k = 0; k < n_p; ++k ) {
                 tmp(i,j) += Lambda_squiggle(i,k) * filtered_covs(t, k, j);
             }
         }
     }
     for (int i = 0; i < n_p; ++i ) {
         for (int j = 0; j < n_p; ++j ) {
             smoothed_cov(i,j) = filtered_covs(t,i,j);
             for (int k = 0; k < n_p; ++k ) {
                 smoothed_cov(i,j) -= filtered_covs(t, i,k) * tmp(k,j);
             }
         }
     }

    /*
    printf("smooth cov. given filtered cov\\n");
    print_subarray3(filtered_covs, t);
    printf(" and squiggle\\n");
    print_array2(Lambda_squiggle);
    printf(" got smoothed cov\\n");
    print_array2(smoothed_cov);
    */
}

void update_Lambda_squiggle(blitz::Array<double, 2> Lambda_hat, blitz::Array<double, 2> Lambda_squiggle, blitz::Array<double, 3> filtered_covs, int t) {
     int n_p = filtered_covs.shape()[1];

     Lambda_squiggle = Lambda_hat;

     double accum = 0; // this will equal dot(hat_g, gain)

     for (int i=0; i < n_p; ++i) {
         double hat_gi = 0;
         double hat_gti = 0;
         for (int j=0; j < n_p; ++j) {
             hat_gi += Lambda_hat(i, j) * filtered_covs(t, j, 0);
             hat_gti += Lambda_hat(j, i) * filtered_covs(t, j, 0);
         }
         hat_gi /= filtered_covs(t,0,0);
         hat_gti /= filtered_covs(t,0,0);

         accum += hat_gi * filtered_covs(t, i, 0);

         Lambda_squiggle(i,0) -= hat_gi;
         Lambda_squiggle(0,i) -= hat_gti;
     }
     accum /= filtered_covs(t,0,0);

     Lambda_squiggle(0,0) += accum;
     Lambda_squiggle(0,0) += 1.0/filtered_covs(t,0,0);
}

void update_lambda_squiggle(blitz::Array<double, 1> lambda_hat, blitz::Array<double, 1> lambda_squiggle, blitz::Array<double, 3> filtered_covs, int t, double obs) {
     int n_p = filtered_covs.shape()[1];

     // compute dot(lambda_hat, gain) where gain = filtered_cov[:, 0] / filtered_cov[0,0]
     double adj = 0;
     for (int i=0; i < n_p; ++i) {
         adj += lambda_hat(i) * filtered_covs(t, i,0);
     }
     adj /= filtered_covs(t,0,0);
     lambda_squiggle = lambda_hat;
     lambda_squiggle(0) -= adj;
     lambda_squiggle(0) -= obs / filtered_covs(t,0,0);
}

void update_Lambda_hat(blitz::Array<double, 2> Lambda_hat, blitz::Array<double, 2> Lambda_squiggle, blitz::Array<double, 1> params) {
    Lambda_hat = Lambda_squiggle;

    int n_p = params.size();

    // tmp = K * A
    // here we use Lambda_squiggle as tmp
    for (int i=0; i < n_p; ++i) {
        for (int j=0; j < n_p; ++j) {
            Lambda_squiggle(i,j) = params(j) * Lambda_hat(i,0);
        }
        for (int j = 0; j < n_p-1; ++j) {
            Lambda_squiggle(i,j) += Lambda_hat(i, j+1);
        }
    }

    // K = A.T * tmp
    for (int j=0; j < n_p; ++j) {
        for (int i=0; i < n_p; ++i) {
            Lambda_hat(i,j) = Lambda_squiggle(0,j) * params(i);
        }
        for (int i=0; i < n_p-1; ++i) {
            Lambda_hat(i,j) += Lambda_squiggle(i+1,j);
        }
    }
}

void update_lambda_hat(blitz::Array<double, 1> lambda_hat, blitz::Array<double, 1> lambda_squiggle, blitz::Array<double, 1> params) {
    int n_p = lambda_hat.size();
    for (int i=0; i < n_p-1; ++i) {
        lambda_hat(i) = params(i)*lambda_squiggle(0) + lambda_squiggle(i+1);
    }
    lambda_hat(n_p-1) = params(n_p-1)*lambda_squiggle(0);
}

void smooth_AR(blitz::Array<double, 1> x, blitz::Array<bool, 1> mask, blitz::Array<double, 1> params, blitz::Array<double, 2> filtered_means, blitz::Array<double, 3> filtered_covs, int i_end, blitz::Array<double, 1> lambda_hat, blitz::Array<double, 2> Lambda_hat, blitz::Array<double, 1> lambda_squiggle, blitz::Array<double, 2> Lambda_squiggle, blitz::Array<double, 2> cov_tmp) {
    int n = x.size();

    for (int t = n-1; t >= i_end; --t) {

        if (!mask(t)) {
           update_Lambda_squiggle(Lambda_hat, Lambda_squiggle, filtered_covs, t);
           update_lambda_squiggle(lambda_hat, lambda_squiggle, filtered_covs, t, x(t) - filtered_means(t,0));
        } else {
          // uses blitz++ copy-on-assignment semantics
          lambda_squiggle = lambda_hat;
          Lambda_squiggle = Lambda_hat;
        }

        smooth_mean_inplace(filtered_means, filtered_covs, t, lambda_squiggle);

        // here we pass Lambda_hat just as a tmp/scratch variable
        smooth_cov_inplace(filtered_covs, t, Lambda_squiggle, Lambda_hat, cov_tmp);

        update_Lambda_hat(Lambda_hat, Lambda_squiggle, params);
        update_lambda_hat(lambda_hat, lambda_squiggle, params);
    }

}


"""

    sample_lp = weave.inline(code, ['i_start', 'i_end', 'x', 'y',
                                    'x_filtered_means', 'y_filtered_means',
                                    'x_filtered_covs', 'y_filtered_covs',
                                    'x_lambda_hat', 'y_lambda_hat',
                                    'x_Lambda_hat', 'y_Lambda_hat',
                                    'x_lambda_squiggle', 'y_lambda_squiggle',
                                    'x_Lambda_squiggle', 'y_Lambda_squiggle',
                                    'smoothed_mean_x', 'smoothed_mean_y',
                                    'smoothed_cov_x', 'smoothed_cov_y',
                                    'shape', 'observed_nonrepeatable',
                                    'yx_offset',
                                    'x_params', 'y_params',
                                    'x_var', 'y_var',
                                    'seed', 'have_target',
                                    'target_wiggle',
                                    'x_mask', 'y_mask',
                                    'obs_mask', 'debug'
                                ],
                             type_converters=converters.blitz,
                             support_code=support, compiler='gcc')

    x += x_model.c

    final_x = np.copy(x)

    if not have_target:
        if start_idx is None and end_idx is None:
            latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=0, set_signal_length=True)
        else:
            latent.set_nonrepeatable_wiggle(x, shape_env=shape, repeatable_wiggle=repeatable, start_idx=clipped_x_start_idx)
    return sample_lp
