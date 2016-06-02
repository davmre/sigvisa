import numpy as np
import sys
import os
import traceback
import itertools
import pickle
import time
import hashlib
from functools32 import lru_cache

from sigvisa.models.ttime import tt_predict
from sigvisa.models.distributions import Laplacian, Exponential
from sigvisa.models.logistic_regression import LogisticRegressionModel
from sigvisa.graph.sigvisa_graph import SigvisaGraph, ModelNotFoundError
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.source.event import Event, get_event
from sigvisa.utils.geog import wrap_lonlat, dist_km
import sigvisa.source.brune_source as brune

from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from sigvisa.plotting.plot import savefig

import scipy.weave as weave
from scipy.weave import converters
import scipy.stats
import scipy.integrate

"""

This file contains methods to propose new events using a Hough
transform, given a graph containing unassociated templates. We use a
vaguely Bayesian interpretation of the Hough transform, in which the
number of votes for a particular bin is roughly analagous to a
log-probability that there is an event within that bin.

The main logic is in generate_hough_array, which then calls the other methods.

"""

DEG_WIDTH_KM = 111.32
P_WAVE_VELOCITY_KM_PER_S = 6.0

def get_amp_transfer_model(sg, sta, phase, chan, band):
    try:
        modelid = sg.get_param_model_id(runids=sg.runids, sta=sta,
                                        phase=phase, model_type=sg._tm_type("amp_transfer"),
                                        param="amp_transfer", template_shape=sg.template_shape,
                                        chan=chan, band=band)
        model = sg.load_modelid(modelid)
    except ModelNotFoundError:
        model = sg.dummy_prior["amp_transfer"]
    return model

def generate_sta_hough(sta, sta_hough_array, atimes, amps, tmids,
                       ttimes_centered, ttimes_corners, amp_transfers,
                       amp_transfer_stds, detprobs, stime,
                       time_tick_s, bin_width_deg, mb_bin_width,
                       min_mb, bin_width_km, source_logamps,
                       uatemplate_rate, ua_amp_model, valid_len,
                       lognoise, save_assoc=False):
    lonbins, latbins, depthbins, mbbins, timebins  = sta_hough_array.shape
    time_radius = time_tick_s / 2.0
    lonbin_deg = 360.0/lonbins

    ntemplates = len(atimes)

    amps = np.asarray(amps, dtype=np.float)
    atimes = np.asarray(atimes, dtype=np.float)
    tmids = np.asarray(tmids, dtype=np.int)

    ua_amp_lps = np.array([ua_amp_model.log_p(amp) for amp in amps])

    # HACK
    #uatemplate_rate = (len(atimes)+1) / float(valid_len)

    ua_poisson_lp_full = ntemplates * np.log(uatemplate_rate) - (uatemplate_rate * valid_len)
    ua_poisson_lp_incr = float(np.log(uatemplate_rate)) # change in the ua_poisson_lp if we have one fewer uatemplate

    nphases = ttimes_centered.shape[3]


    #detection_lps = np.log(detection_probs)
    #nodetection_lps =np.log(1-detection_probs)

    null_ll = float(ua_poisson_lp_full + np.sum(ua_amp_lps))

    cdfs = """
    // cdf of Laplace(0.0, 5.0) computed for half-integer x values [-40, 40] inclusive
    double laplace_table[] ={0.00017, 0.00020, 0.00025, 0.00031, 0.00037, 0.00046, 0.00056, 0.00068, 0.00083, 0.00101, 0.00124, 0.00151, 0.00185, 0.00226, 0.00276, 0.00337, 0.00411, 0.00503, 0.00614, 0.00750, 0.009\
16, 0.01119, 0.01366, 0.01669, 0.02038, 0.02489, 0.03041, 0.03714, 0.04536, 0.05540, 0.06767, 0.08265, 0.10095, 0.12330, 0.15060, 0.18394, 0.22466, 0.27441, 0.33516, 0.40937, 0.50000, 0.59063, 0.66484, 0.72\
559, 0.77534, 0.81606, 0.84940, 0.87670, 0.89905, 0.91735, 0.93233, 0.94460, 0.95464, 0.96286, 0.96959, 0.97511, 0.97962, 0.98331, 0.98634, 0.98881, 0.99084, 0.99250, 0.99386, 0.99497, 0.99589, 0.99663, 0.9\
9724, 0.99774, 0.99815, 0.99849, 0.99876, 0.99899, 0.99917, 0.99932, 0.99944, 0.99954, 0.99963, 0.99969, 0.99975, 0.99980, 0.99983};
    // cdf of Normal(0, 1) computed for 81 values [-4, 4] inclusive
    double gaussian_table[]={0.00003, 0.00005, 0.00007, 0.00011, 0.00016, 0.00023, 0.00034, 0.00048, 0.00069, 0.00097, 0.00135, 0.00187, 0.00256, 0.00347, 0.00466, 0.00621, 0.00820, 0.01072, 0.01390, 0.01786, 0.02275, 0.02872, 0.03593, 0.04457, 0.05480, 0.06681, 0.08076, 0.09680, 0.11507, 0.13567, 0.15866, 0.18406, 0.21186, 0.24196, 0.27425, 0.30854, 0.34458, 0.38209, 0.42074, 0.46017, 0.50000, 0.53983, 0.57926, 0.61791, 0.65542, 0.69146, 0.72575, 0.75804, 0.78814, 0.81594, 0.84134, 0.86433, 0.88493, 0.90320, 0.91924, 0.93319, 0.94520, 0.95543, 0.96407, 0.97128, 0.97725, 0.98214, 0.98610, 0.98928, 0.99180, 0.99379, 0.99534, 0.99653, 0.99744, 0.99813, 0.99865, 0.99903, 0.99931, 0.99952, 0.99966, 0.99977, 0.99984, 0.99989, 0.99993, 0.99995, 0.99997};

    double laplace_cdf(double x) {
        int ix = (int) floor(x);
        int ix2 = ix+1;
        if (ix2 <= -40) return 0.0;
        if (ix >= 40) return 1.0;
        double tbl1 = laplace_table[ix+40];
        double tbl2 = laplace_table[ix2+40];
        double y = x-ix;
        return (1-y)*tbl1 + y*tbl2;
    }

    double gaussian_cdf(double x) {
        x *= 10; // convert from [-4, 4] to index range of [-40, 40]
        if (isinf(x)) { if (x > 0) return 1.0; else return 0.0; }
        int ix = (int) floor(x);
        int ix2 = ix+1;
        if (ix2 <= -40) return 0.0;
        if (ix >= 40) return 1.0;
        double tbl1 = gaussian_table[ix+40];
        double tbl2 = gaussian_table[ix2+40];
        double y = x-ix;
        return (1-y)*tbl1 + y*tbl2;
    }

    long timespec_diff_us(timespec &x, timespec &y) {
         long diff_usec = 0;
         diff_usec += 1000000 * (y.tv_sec - x.tv_sec);
         diff_usec += (y.tv_nsec - x.tv_nsec)/1000;
         return diff_usec;
    }
    """

    # ttimes, amp_transfers, amp_transfer_stds: loc bins
    # phase_score: timebin*mbbin
    # assoc: int, nphases*timebin*mbbin
    # atimes, amps
    # background_cost_lp
    # detection

    phase_score = np.empty((mbbins, timebins))
    assoc = np.empty((nphases, mbbins, timebins), dtype=np.uint8)
    timebin_lps = np.empty((timebins,))

    save_assoc = 1 if save_assoc else 0 # make c++ happy
    if save_assoc:
        full_assoc = np.zeros((lonbins, latbins, depthbins, mbbins, timebins,  nphases,), dtype=np.uint8)
        phase_scores = np.zeros((lonbins, latbins, depthbins, mbbins, timebins,  nphases,), dtype=np.float)
    else:
        full_assoc = np.zeros((1,), dtype=np.uint8)
        phase_scores = np.zeros((1,), dtype=np.float)


    # null_ll is the likelihood of all templates under the
    # all-uatemplate, no-event explanation.

    # the thing we compute in the bins is the likelihood under an
    # event-based explanation. the *default* event-based explanation
    # is that no templates have been associated, so all phases of this
    # event are undetected at the current station. so we have to start
    # by paying the no-detection penalty in each bin. this gets
    # canceled out below, if/when we *do* associate a template.
    sta_hough_array.fill(null_ll)

    for phase_idx in range(nphases):
        adj = np.log(1 - detprobs[:,:,:,:,phase_idx])
        #print "adj min %f max %f" % (np.min(adj), np.max(adj))
        sta_hough_array += adj[:,:,:,:,np.newaxis] 

    #sha = sta_hough_array.copy()

    # optimization to avoid looping through all timebins for each
    # lon/lat/depth/mb bin. We represent the set of timebins with
    # nonzero scores as a boolean array (timebins_used_flag) which
    # admits constant-time insertions and lookups. We also store a
    # list of these timebins in timebins_used, for efficient
    # iteration.
    timebins_used_flag_phase = np.zeros((timebins,), dtype=np.int8)
    timebins_used_phase = np.empty((timebins,), dtype=np.int)

    timebins_used_flag = np.zeros((timebins,), dtype=np.int8)
    timebins_used = np.empty((timebins,), dtype=np.int)


    #print "source logamps"
    #print source_logamps

    debug_lonbin = -1
    debug_latbin = -1
    debug_depthbin = -1

        
    code = """
long init_time = 0;
long plausible_time = 0;
long laplace_time = 0;
long mbscore_time = 0;
long fullscore_time = 0;
long copy_result_time = 0;

for (int i=0; i < lonbins; ++i) {
     for (int j=0; j < latbins; ++j) {
         for (int k=0; k < depthbins; ++k) {
            bool verbose = (i==debug_lonbin && j==debug_latbin && k==debug_depthbin);
            memset(&assoc(0,0,0), -1, nphases*timebins*mbbins);
            int n_timebins_used = 0;

            for (int phaseidx=0; phaseidx < nphases; ++phaseidx) {
                double ttime_center = ttimes_centered(i,j, k, phaseidx);
                double amp_transfer = amp_transfers(i, j, k, phaseidx);
                double amp_transfer_std = amp_transfer_stds(i, j, k, phaseidx);

                timespec t0, t1;
                clock_gettime(CLOCK_REALTIME, &t0);

                memset(&phase_score(0,0), -999.0, timebins*mbbins*sizeof(double));

                int n_timebins_used_phase = 0;


                clock_gettime(CLOCK_REALTIME, &t1);
                init_time += timespec_diff_us(t0, t1);

                for (int t=0; t < ntemplates; ++t) {

                    clock_gettime(CLOCK_REALTIME, &t0);
                    double atime = atimes(t);
                    double amp = amps(t);
                    double origin_time_center = atime - ttime_center;
                    double source_logamp = amp - amp_transfer;

                    // mean (backprojected) origin times assuming locations at the
                    // corners of the lon/lat/depth bin
                    double origin_time_000 = atime - ttimes_corners(i,j,k,phaseidx);
                    double origin_time_001 = atime - ttimes_corners(i,j,k+1,phaseidx);
                    double origin_time_010 = atime - ttimes_corners(i,j+1,k,phaseidx);
                    double origin_time_011 = atime - ttimes_corners(i,j+1,k+1,phaseidx);
                    double origin_time_100 = atime - ttimes_corners(i+1,j,k,phaseidx);
                    double origin_time_101 = atime - ttimes_corners(i+1,j,k+1,phaseidx);
                    double origin_time_110 = atime - ttimes_corners(i+1,j+1,k,phaseidx);
                    double origin_time_111 = atime - ttimes_corners(i+1,j+1,k+1,phaseidx);

                    if (origin_time_center < stime  || origin_time_center > stime + timebins*time_tick_s) {
                       if (verbose) {
                          printf("skipping t=%d for implausible origin time %.1f\\n", t, origin_time_center);
                       }
                       continue;
                    }

                    double min_plausible_time = origin_time_center;

                    if (origin_time_000 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_000);
                    if (origin_time_001 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_001);
                    if (origin_time_010 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_010);
                    if (origin_time_011 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_011);
                    if (origin_time_100 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_100);
                    if (origin_time_101 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_101);
                    if (origin_time_110 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_110);
                    if (origin_time_111 > stime) min_plausible_time = std::min(min_plausible_time, origin_time_111);

                    double max_plausible_time = std::max(origin_time_center, origin_time_000);
                    max_plausible_time = std::max(max_plausible_time, origin_time_001);
                    max_plausible_time = std::max(max_plausible_time, origin_time_010);
                    max_plausible_time = std::max(max_plausible_time, origin_time_011);
                    max_plausible_time = std::max(max_plausible_time, origin_time_100);
                    max_plausible_time = std::max(max_plausible_time, origin_time_101);
                    max_plausible_time = std::max(max_plausible_time, origin_time_110);
                    max_plausible_time = std::max(max_plausible_time, origin_time_111);


                    int min_plausible_timebin = std::max(0, int((min_plausible_time-30.0-stime) / time_tick_s));
                    int max_plausible_timebin = std::min(timebins-1, int((max_plausible_time + 30.0-stime) / time_tick_s));

                    int min_plausible_mbbin = 0;
                    int max_plausible_mbbin = 0;
                    double min_logamp = source_logamp - 3*amp_transfer_std;
                    double max_logamp = source_logamp + 3*amp_transfer_std;
                    for (int mbbin=0; mbbin < mbbins; ++mbbin) {
                        double left_logamp = source_logamps(mbbin, phaseidx);
                        // double right_logamp = source_logamps(mbbin+1, phaseidx)
                        if (left_logamp < min_logamp) { min_plausible_mbbin = mbbin; }
                        if (left_logamp < max_logamp) { max_plausible_mbbin = mbbin; }
                    }

                    //int min_plausible_mbbin = std::max(0, int((mb-3*amp_transfer_std - min_mb) / mb_bin_width) );
                    //int max_plausible_mbbin = std::min(mbbins-1, int((mb+3*amp_transfer_std - min_mb) / mb_bin_width) );

                    if (verbose) {
                       printf("plausible timebins %d %d\\n", min_plausible_timebin, max_plausible_timebin);
                      printf("plausible mbbins %d %d (amp %.1f at %.1f std %.1f) (logamp range %.1f %.1f %.1f)\\n", min_plausible_mbbin, max_plausible_mbbin, amp, amp_transfer, amp_transfer_std, min_logamp, source_logamp, max_logamp);
                    }

                    clock_gettime(CLOCK_REALTIME, &t1);
                    plausible_time += timespec_diff_us(t0, t1);;
                    clock_gettime(CLOCK_REALTIME, &t0);
                    for (int timebin=min_plausible_timebin; timebin <= max_plausible_timebin; timebin++) {
                               //printf("timebins %d used %d\\n", timebins, n_timebins_used);
                            if (!timebins_used_flag_phase(timebin)) {
                               timebins_used_flag_phase(timebin) = 1;
                               timebins_used_phase(n_timebins_used_phase++) = timebin;
                               if (n_timebins_used_phase > timebins) {
                                   printf("something went wildly wrong in propose_hough.py\\n");
                                  exit(0);
                                }
                            }
                            if (!timebins_used_flag(timebin)) {
                               timebins_used_flag(timebin) = 1;
                               timebins_used(n_timebins_used++) = timebin;
                               if (n_timebins_used > timebins) {
                                   printf("something went wildly wrong in propose_hough.py\\n");
                                  exit(0);
                                }
                            }


                            /* p(atime | bin) = int_loc int_time p(atime | loc, time)p(loc|bin)p(time|bin)
                               We assume p(atime | loc, time) is Laplace(atime; time+tt, 2.5).
                               Integrating this over the time bin involves the CDF of the Laplace
                               distribution, giving p(atime | loc) as a difference of laplace CDFs.
                               We approximate the integral over locations in the bin numerically,
                               by evaluating p(atime | loc) at the eight corners and the center,
                               and averaging.
                            */
                            double timebin_left = (stime + timebin * time_tick_s);
                            double timebin_right = timebin_left + time_tick_s;

                            double bin_weight = laplace_cdf(timebin_right-origin_time_000) - laplace_cdf(timebin_left-origin_time_000);
                            bin_weight += laplace_cdf(timebin_right-origin_time_001) - laplace_cdf(timebin_left-origin_time_001);
                            bin_weight += laplace_cdf(timebin_right-origin_time_010) - laplace_cdf(timebin_left-origin_time_010);
                            bin_weight += laplace_cdf(timebin_right-origin_time_011) - laplace_cdf(timebin_left-origin_time_011);
                            bin_weight += laplace_cdf(timebin_right-origin_time_100) - laplace_cdf(timebin_left-origin_time_100);
                            bin_weight += laplace_cdf(timebin_right-origin_time_101) - laplace_cdf(timebin_left-origin_time_101);
                            bin_weight += laplace_cdf(timebin_right-origin_time_110) - laplace_cdf(timebin_left-origin_time_110);
                            bin_weight += laplace_cdf(timebin_right-origin_time_111) - laplace_cdf(timebin_left-origin_time_111);
                            bin_weight += laplace_cdf(timebin_right-origin_time_center) - laplace_cdf(timebin_left-origin_time_center);
                            bin_weight /= 9;


                            double atime_lp = log(bin_weight) - log(time_tick_s);

                            /*
                            if (verbose) {
                               printf("%d %f %f %f %f (%f %f %f %f %f %f %f %f %f)\\n", timebin, bin_weight, atime_lp, timebin_left, timebin_right, origin_time_000-timebin_left, origin_time_001-timebin_left, origin_time_010-timebin_left, origin_time_011-timebin_left, origin_time_100-timebin_left, origin_time_101-timebin_left, origin_time_110-timebin_left, origin_time_111-timebin_left, origin_time_center-timebin_left);
                            }*/


                            timebin_lps(timebin) = atime_lp;
                    }

                    clock_gettime(CLOCK_REALTIME, &t1);
                    laplace_time += timespec_diff_us(t0, t1);
                    // every bin should track p(event | best assoc) and idx(best assoc)
                    // for each phase, this means we track the "score" of the best assoc.
                    // there is a default score corresponding to the "none" assoc.
                    for (int mbbin = min_plausible_mbbin; mbbin <= max_plausible_mbbin; mbbin++) {
                        clock_gettime(CLOCK_REALTIME, &t0);
                        double logamp_left = source_logamps(mbbin, phaseidx);
                        double logamp_right = source_logamps(mbbin+1, phaseidx);
                        double at_residual_left =  (amp - logamp_left) - amp_transfer;
                        double at_residual_right = (amp - logamp_right) - amp_transfer;

                        //  this subtraction appears backwards (left-right), but is correct
                        //  because the residual *decreases* as source logamp increases
                        double bin_weight = gaussian_cdf(at_residual_left/amp_transfer_std)
                                               - gaussian_cdf(at_residual_right/amp_transfer_std);
                        double mb_lp = log(bin_weight) - log(at_residual_left-at_residual_right);

                        if(verbose) {
    printf("phase %d tmid %d (idx %d) mbbin %d at_residuals %.2f %.2f weight %f lp %f\\n", phaseidx, tmids(t), t, mbbin, at_residual_left, at_residual_right, bin_weight, mb_lp);
    }

                        // probability the signal is above the noise floor
                        /*
                        // double noise_z_left = (lognoise - 1 - ( logamp_left + amp_transfer))/amp_transfer_std;
                        // double noise_z_right = (lognoise - 1 - ( logamp_right + amp_transfer))/amp_transfer_std;
                        // double detection_prob = .5*gaussian_cdf(noise_z_left) + .5*gaussian_cdf(noise_z_right);
                        // detection_prob = detection_prob < 0.01 ? 0.01 : detection_prob;
                        // detection_prob = detection_prob > 0.99 ? 0.99 : detection_prob;
                        */
                        // double logdet_odds = log(detection_prob / (1-detection_prob));
                        // double logdet_odds = (detection_lps(phaseidx) - nodetection_lps(phaseidx));
                        double logdet_odds = log(  detprobs(i,j,k,mbbin,phaseidx) / (1.0 - detprobs(i,j,k,mbbin,phaseidx)));
                       clock_gettime(CLOCK_REALTIME, &t1);
                        mbscore_time += timespec_diff_us(t0, t1);
                        clock_gettime(CLOCK_REALTIME, &t0);

                       for (int timebin=min_plausible_timebin; timebin <= max_plausible_timebin; timebin++) {


                            // don't allow templates that have previously
                            // been associated with other phases.
                            bool exclude_tmpl_from_bin = 0;
                            for (int oldphase = 0; oldphase < phaseidx; ++oldphase) {
                                if (assoc(oldphase, mbbin, timebin) == t) {
                                    exclude_tmpl_from_bin = 1;
                                    break;
                                }
                            }
                            if (exclude_tmpl_from_bin) { continue; }

                            double tmpl_score = logdet_odds + (mb_lp - ua_amp_lps(t)) + timebin_lps(timebin) - ua_poisson_lp_incr;
                            double oldval = phase_score(mbbin, timebin);
/*
                            if (verbose) {
                                  printf("lon %d lat %d depth %d time %d mbbin %d phase %d processing template %d with score %f vs oldval %f (amp %.1f, bin_weight %f, atime_lp %f)\\n", i, j, k, timebin, mbbin, phaseidx, t, tmpl_score, oldval, amp, bin_weight, timebin_lps(timebin));
                            }
*/
                            if (tmpl_score > oldval) {
                                phase_score(mbbin, timebin) = tmpl_score;
                                assoc(phaseidx, mbbin, timebin) = t;
                            }
                        }
                        clock_gettime(CLOCK_REALTIME, &t1);
                        fullscore_time += timespec_diff_us(t0, t1);
                    }
                } // t (template idx)

                clock_gettime(CLOCK_REALTIME, &t0);

                //for (int timebin=0; timebin < timebins; ++timebin) {
                for (int i_timebin=0; i_timebin < n_timebins_used_phase; ++i_timebin) {
                    int timebin = timebins_used_phase(i_timebin);
                    timebins_used_flag_phase(timebin) = 0;
                    for (int mbbin=0; mbbin < mbbins; ++mbbin) {
                        sta_hough_array(i,j,k,mbbin,timebin) +=   phase_score(mbbin, timebin);

                        if (save_assoc) {
                           full_assoc(i,j,k,mbbin,timebin, phaseidx) = assoc(phaseidx,  mbbin, timebin)+1;
                           phase_scores(i,j,k,mbbin,timebin, phaseidx) = phase_score(mbbin, timebin);
                        }
                    }
                }
                clock_gettime(CLOCK_REALTIME, &t1);
                copy_result_time += timespec_diff_us(t0, t1);;

            } // phaseidx

            // logsumexp the best detection explanation with the null/undetected explanation
            for (int mbbin=0; mbbin < mbbins; ++mbbin) {
               for (int phase_idx=0; phase_idx < nphases; ++phase_idx) {
                 double base = null_ll + log(1-detprobs(i,j,k,mbbin,phase_idx));
                 for (int i_timebin=0; i_timebin < n_timebins_used; ++i_timebin) {
                    int timebin = timebins_used(i_timebin);
                    timebins_used_flag(timebin) = 0;
                    double s1 = sta_hough_array(i,j,k,mbbin, timebin);
                    
                    double tmp = s1-base;
                    if (tmp > 0) {
                       sta_hough_array(i,j,k,mbbin, timebin) = s1 + log1p(exp(-tmp));
                    } else {
                       sta_hough_array(i,j,k,mbbin,timebin) = base + log1p(exp(tmp));
                    }
                }
             }
          }


        } // k (depth)
    } // j (latbin)
} // i (lonbin)
//printf("times: init %ld plausible %ld laplace %ld mbscore %ld fullscore %ld copy %ld, total %ld\\n", init_time, plausible_time, laplace_time, mbscore_time, fullscore_time, copy_result_time, init_time+plausible_time+ laplace_time+ mbscore_time+ fullscore_time+ copy_result_time);
    """


    stime = float(stime)
    weave.inline(code,['latbins', 'lonbins', 'depthbins', 'timebins', 'mbbins', 'nphases',
                       'sta_hough_array', 'ttimes_centered', 'ttimes_corners', 'amp_transfers', 'amp_transfer_stds',
                       'source_logamps', 'phase_score', 'assoc',
                       'bin_width_deg', 'time_tick_s', 'mb_bin_width',
                       'bin_width_km', 'stime', 'min_mb', 'detprobs',
                       'amps', 'atimes', 'tmids', 'ntemplates', 'debug_lonbin', 'debug_depthbin', 'debug_latbin', 'ua_poisson_lp_incr', 'ua_amp_lps', 'full_assoc', 'phase_scores', 'save_assoc', 'lognoise', 'timebin_lps', 'timebins_used_phase', 'timebins_used_flag_phase', 'timebins_used', 'timebins_used_flag', 'null_ll'],
                 support_code=cdfs, headers=["<math.h>", "<unordered_set>", "<time.h>"], type_converters = converters.blitz,verbose=2,compiler='gcc', extra_compile_args=["-std=c++11"], extra_link_args=["-lrt",])

    

    #sta_hough_array[:,:,:,:,:] = np.logaddexp(sha, sta_hough_array)

    #print "added template"

    #for phase_idx in range(nphases):
    #    adj = np.log(1 - detprobs[:,:,:,:,phase_idx])
    #    delta = sta_hough_array - null_ll - adj[:,:,:,np.newaxis,:] 
    #    diff = delta - phase_scores[:,:,:,:,:,phase_idx]
    #    import pdb; pdb.set_trace()


    

    return null_ll, full_assoc, phase_scores


def categorical_sample_array(a):
    """

    Sample a bin from the Hough accumulator. Assume that our log
    probabilities have previously been exponentied, and are now just
    probabilities.

    """

    lonbins, latbins, depthbins, mbbins, timebins = a.shape
    s = float(np.sum(a))
    u = float(np.random.rand())

    v = np.zeros((5,), dtype=np.int)

    t0 = time.time()
    code = """
double accum = 0;
double goal = u*s;
int done = 0;
for (int i=0; i < lonbins; ++i) {
    for (int j=0; j < latbins; ++j) {
        for (int k=0; k < depthbins; ++k) {
          for (int m=0; m < mbbins; ++m)  {
            for (int l=0; l < timebins; ++l)  {
                   accum += a(i,j,k,m,l);
                    if (accum >= goal) {
                       v(0) = i;
                       v(1) = j;
                       v(2) = k;
                       v(3) = m;
                       v(4) = l;
                       done = 1;
                       break;
                    }
                }
                if (done) { break; }
            }
            if (done) { break; }
        }
        if (done) { break; }
    }
    if (done) { break; }
}
    """
    weave.inline(code,['latbins', 'lonbins', 'depthbins','mbbins', 'timebins',  'a', 'u', 's', 'v'],type_converters = converters.blitz,verbose=2,compiler='gcc')

    return tuple(v)

def categorical_prob(a, idx):
    s = np.sum(a)
    try:
        p = a[idx]/s
    except IndexError:
        # the Hough distribution is only supported within the
        # time/space period defined by the array, so index out of
        # bounds means zero probability
        p = 0.0
    return p


def event_from_bin(hc, idx):
    (left_lon, right_lon), (bottom_lat, top_lat), (min_depth, max_depth), (min_mb, max_mb), (min_time, max_time) = hc.index_to_coords(idx)

    lonwidth=right_lon-left_lon
    latwidth = top_lat-bottom_lat
    depthwidth = max_depth-min_depth
    timewidth = max_time-min_time
    mbwidth = max_mb-min_mb

    lon = left_lon + np.random.rand()*lonwidth
    lat = bottom_lat + np.random.rand()*latwidth
    depth = min_depth + np.random.rand()*depthwidth
    t = min_time + np.random.rand()*timewidth
    mb = min_mb + np.random.rand()*mbwidth

    # make sure we return a valid lon/lat even if grid is offset
    lon, lat = wrap_lonlat(lon, lat)

    evlp = -np.log(lonwidth) - np.log(latwidth) -np.log(depthwidth) -np.log(timewidth) -np.log(mbwidth)
    ev = Event(lon=lon, lat=lat, time=t, depth=depth, mb=mb, natural_source=True)

    return ev, evlp

def visualize_hough_array(hough_array, sites, fname=None, ax=None, timeslice=None, normalize=True, location_array=None, region=None):
    """

    Save an image visualizing the given Hough accumulator array. If
    timeslice is the index of a time bin, display only that bin,
    otherwise sum over all time bins.

    """

    if location_array is not None:
        lonbins, latbins = location_array.shape
    else:
        lonbins, latbins, depthbins, mbbins, timebins = hough_array.shape
        
        if timeslice is None:
            location_array = np.sum(hough_array, axis=(2,3,4))
        else:
            location_array = hough_array[:,:,timeslice]


    if region is not None:
        left_lon = region.left_lon
        right_lon = region.right_lon
        bottom_lat = region.bottom_lat
        top_lat = region.top_lat
    else:
        left_lon = -180
        right_lon = 180
        bottom_lat = -90
        top_lat = 90

    width_deg = float(right_lon - left_lon)
    height_deg = float(top_lat - bottom_lat)

    latbin_deg = height_deg/latbins
    lonbin_deg = width_deg/lonbins


    # set up the map
    if ax is None:
        fig = Figure(figsize=(8, 5), dpi=300)
        ax = fig.add_subplot(111)

    if width_deg == 360 and height_deg==180:
        bmap = Basemap(resolution="c", projection = "robin", lon_0 = 0, ax=ax)
    else:
        bmap = Basemap(resolution="c", projection = "cyl", llcrnrlon=left_lon, urcrnrlon=right_lon, llcrnrlat=bottom_lat, urcrnrlat=top_lat, ax=ax)

    bmap.drawcoastlines(zorder=10)
    bmap.drawmapboundary()
    parallels = [int(k) for k in np.linspace(bottom_lat, top_lat, 10)]
    bmap.drawparallels(parallels, labels=[False, True, True, False], fontsize=8, zorder=4)
    meridians = [int(k) for k in np.linspace(left_lon, right_lon, 10)]
    bmap.drawmeridians(meridians, labels=[True, False, False, True], fontsize=5, zorder=4)

    # shade the bins according to their probabilities
    def draw_cell( left_lon, bottom_lat, alpha=1.0, facecolor="red"):
        right_lon = min(left_lon + lonbin_deg, 180)
        top_lat = min(bottom_lat+latbin_deg, 90)
        lons = [left_lon, left_lon, right_lon, right_lon]
        lats = [bottom_lat, top_lat, top_lat, bottom_lat]
        x, y = bmap( lons, lats )
        xy = zip(x,y)
        poly = Polygon( xy, facecolor=facecolor, alpha=alpha, edgecolor="none", linewidth=0 )
        ax.add_patch(poly)
    m = np.max(location_array)
    for i in range(lonbins):
        lon = max(left_lon + i * lonbin_deg, left_lon)
        for j in range(latbins):
            lat = max(bottom_lat + j * latbin_deg, bottom_lat)

            if normalize:
                alpha = location_array[i,j]/m
            else:
                alpha = location_array[i,j]
            # compress dynamic range so that smaller probabilities still show up
            #alpha = 1 - (1-alpha)/1.5
            if alpha > 1e-2:
                #alpha = np.sqrt(alpha)
                draw_cell(left_lon = lon, bottom_lat = lat, alpha = alpha)

    # draw the stations
    s = Sigvisa()
    for site in sites:
        sta_location = s.earthmodel.site_info(site, 0)[0:2]
        x1, x2 = bmap(sta_location[0], sta_location[1])
        bmap.plot([x1], [x2], marker="x", ms=4, mfc="none", mec="blue", mew=1, alpha=1, zorder=10)
        x_off = 3
        y_off = 3
        ax.annotate(site,xy=(x1, x2),xytext=(x_off, y_off),textcoords='offset points',
                      size=6,color = 'blue',zorder=10)

    if fname is not None:
        savefig(fname, fig, dpi=300, bbox_inches='tight')




def exp_in_place(hough_array):
    lonbins, latbins, timebins = hough_array.shape
    code = """

for (int i=0; i < lonbins; ++i) {
    for (int j=0; j < latbins; ++j) {
        for (int k=0; k < timebins; ++k)  {
hough_array(i,j,k) = exp(hough_array(i,j,k));
        }
    }
}
    """
    weave.inline(code,['latbins', 'lonbins', 'timebins', 'hough_array'],type_converters = converters.blitz,verbose=2,compiler='gcc')


def get_uatemplates(sg, stas=None, tmid_whitelist=None):
    if stas is None:
        stas = sg.station_waves.keys()

    uatemplates_by_sta = {}
    for sta in stas:
        for wn in sg.station_waves[sta]:
            atimes = []
            amps = []
            tmids = []
            for i, (eid, phase) in enumerate(wn.arrivals()):
                if phase != "UA": continue
                if tmid_whitelist is not None and -i not in tmid_whitelist: continue
                v, _ = wn.get_template_params_for_arrival(eid, phase)
                atimes.append(v['arrival_time'])
                amps.append(v['coda_height'])
                tmids.append(-eid)
            uatemplates_by_sta[wn]= (atimes, amps, tmids)

    return uatemplates_by_sta

def synth_templates(sg, n_events=3, phases=("P", "S")):

    evs = [sg.prior_sample_event(3.5, sg.event_start_time, sg.end_time) for i in range(n_events)]

    uatemplates_by_sta = {}
    tmid = 0
    detection_probs = {"P": 0.8, "S": 0.1}
    for (sta, wns) in sg.station_waves.items():
        for wn in wns:
            atimes = []
            amps = []
            tmids = []
            for phase in phases:
                model = get_amp_transfer_model(sg, wn.sta, phase, wn.chan, wn.band)
                detection_prob = detection_probs[phase]
                for ev in evs:
                    ev.depth = 0
                    if np.random.rand() > detection_prob: continue

                    try:
                        tt = tt_predict(ev, sta, phase=phase)
                    except:
                        continue

                    atime = ev.time + tt + Laplacian(0.0, 3.0).sample()
                    source_logamp = brune.source_logamp(mb=ev.mb, band="freq_0.8_4.5", phase=phase)
                    at = model.sample(cond=ev)
                    amp = source_logamp+at

                    if amp < np.log(wn.nm_env.c)-1: continue

                    tmid += 1
                    atimes.append(atime)
                    amps.append(amp)
                    tmids.append(tmid)

        uatemplates_by_sta[sta]=(atimes, amps, tmids)
    return uatemplates_by_sta, evs

def claim_associated_templates(bin_idx, assoc_by_sta, uatemplates_by_sta):
    # given an event in a bin, remove all of its assoications from the atimes/amp lists
    #
    ev_uatemplates_by_sta = {}
    for sta, assoc in assoc_by_sta.items():
        atimes, amps, tmids = uatemplates_by_sta[sta]
        nphases = assoc.shape[-1]

        evatimes, evamps, evtmids = [], [], []
        for phaseidx in range(nphases):
            idx = tuple(list(bin_idx) + [phaseidx,])
            associated_t = assoc[idx]
            if associated_t == 255:
                pass
                #evtmids.append(None)
                #evatimes.append(None)
                #evatimes.append(None)
            else:
                evtmids.append(tmids[associated_t])
                evatimes.append(atimes[associated_t])
                evamps.append(amps[associated_t])

        [tmids.remove(tmid)  for tmid in evtmids if tmid is not None]
        [atimes.remove(atime)  for atime in evatimes if atime is not None]
        [amps.remove(amp)  for amp in evamps if amp is not None]

        ev_uatemplates_by_sta[sta] = evatimes, evamps, evtmids
    return ev_uatemplates_by_sta



def iterative_mixture_hough(sg, hc):
    np.random.seed(2)
    uatemplates_by_sta, evs = synth_templates(sg)
    print "true evs"
    for ev in evs:
        print ev

    #uatemplates_by_sta = get_uatemplates(sg)
    ev_associations = [] # list of events, each consisting of a dictionary mapping stations to (score, list of atimes, list of amps) with the lists ordered by phaseid
    ev_arrays = []

    global_array, debug_by_sta, global_null_lp = global_hough(sg, hc, uatemplates_by_sta, save_debug=True)
    map_event_idx = np.unravel_index(np.argmax(global_array), global_array.shape)
    map_event_score = global_array[map_event_idx] - global_null_lp

    while map_event_score > 1:
        print "event score", map_event_score
        ev_associations.append(claim_associated_templates(map_event_idx, assoc_by_sta, uatemplates_by_sta))
        ev_arrays.append(global_array)

        global_array, assoc_by_sta, global_null_lp = global_hough(sg, hc, uatemplates_by_sta, save_debug=False)
        map_event_idx = np.unravel_index(np.argmax(global_array), global_array.shape)
        map_event_score = global_array[map_event_idx] - global_null_lp


    mixture_array = hc.create_array(dtype=np.float32, fill_val=0.0)
    for i, eva in enumerate(ev_associations):
        #ev_array, _, _ = global_hough(sg, hc, eva, save_debug=False)
        ev_array = ev_arrays[i]
        ev_array = normalize_global(ev_array, 0.0, 1.0, one_event_semantics=True, hc=hc)
        #ev_array *= prob_ev_exists # (expected number of events: can we get a score for this from event likelihood vs unass model?)
        mixture_array += ev_array # expected number of events in a cell

        fname = "newhough_%d_ev%d" % (hc.bin_width_deg, i)
        visualize_hough_array(ev_array, uatemplates_by_sta.keys(), fname=fname+".png", ax=None, timeslice=None)
        print fname

    fname = "newhough_%d_global_mixture" % (hc.bin_width_deg)
    visualize_hough_array(mixture_array, uatemplates_by_sta.keys(), fname=fname+".png", ax=None, timeslice=None)
    print fname

@lru_cache(maxsize=64)
def load_detprob_model(sta, phase, phase_context, runid, enable_dummy=True):

    s = Sigvisa()
    phase_context_str = ','.join(sorted(phase_context))
    sql_query = "select model_fname from sigvisa_hough_detection_model where sta='%s' and phase='%s' and phase_context='%s' and fitting_runid=%d" % (sta, phase, phase_context_str, runid)
    r = s.sql(sql_query)

    if len(r) == 0:
        if enable_dummy:
            weights = np.zeros((6,),)
            intercept = -2
            model = LogisticRegressionModel(weights, intercept, sta=sta, phase=phase)
            print "WARNING: using dummy detmodel for %s %s %s" % (sta, phase, phase_context)
        else:
            raise Exception("no detmodel for query %s" % sql_query)
    else:
        assert(len(r) == 1)
        model_fname = r[0][0]
    
        with open(os.path.join(s.homedir, model_fname), 'rb') as f:
            model = pickle.load(f)

    return model

class HoughConfig(object):
    def __init__(self, stime, len_s, phases = ("P",), bin_width_deg=2.0, min_mb=2.5, max_mb=8.5, mbbins=1, depthbin_bounds=[0, 10, 50, 150, 400, 700], uatemplate_rate=1e-3, left_lon=-180, right_lon=180, bottom_lat=-90, top_lat=90, min_depth=0, max_depth=700, time_tick_s=10.0, global_event_rate=1e-4):

        assert(len_s > 0)

        self.left_lon = left_lon
        self.right_lon = right_lon
        self.bottom_lat=bottom_lat
        self.top_lat = top_lat

        self.lonbins = int((right_lon-left_lon)/bin_width_deg)
        self.latbins = int((top_lat-bottom_lat)/bin_width_deg)
        self.bin_width_deg = (right_lon-left_lon)/float(self.lonbins)

        self.bin_width_km = (40000.0 * (right_lon-left_lon)/360.0)/self.latbins

        self.stime = stime
        #self.time_tick_s = (1.41 * bin_width_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S)
        #self.time_tick_s /= 10
        self.time_tick_s = time_tick_s
        self.timebins = int(float(len_s)/self.time_tick_s)
        self.min_mb = min_mb
        self.max_mb = max_mb
        self.mbbins = mbbins
        self.mb_bin_width = float(max_mb-min_mb)/mbbins

        self.depthbin_bounds = depthbin_bounds
        self.n_depthbins = len(depthbin_bounds)-1
        self.depthbin_widths = np.diff(depthbin_bounds)

        self.phases = phases
        s = Sigvisa()
        self.phaseids = [s.phaseids[phase] for phase in phases]

        self.global_event_rate = global_event_rate
        self.uatemplate_rate=uatemplate_rate
        self.ttime_cache = {}
        self.amp_transfer_cache = {}
        self.detprob_cache = {}

        if self.lonbins <= 2:
            self.cached_ev_prior_log, self.cached_noev_prior_log = self._compute_ev_prior()
        else:
            key = repr([left_lon, right_lon, 
                   bottom_lat, top_lat, 
                   self.bin_width_deg,
                   min_mb,
                   max_mb,
                   mbbins,
                   depthbin_bounds,
                   global_event_rate
                   ])

            hashkey = hashlib.md5(key).hexdigest()[:12]
            cachefile = os.path.join(s.homedir, "db_cache", "ev_prior_grid_%s.npz" % hashkey)
            try:
                c = np.load(cachefile)
                ev_prior_log, noev_prior_log = c['ev_prior_log'], c['noev_prior_log']
            except IOError as e:
                ev_prior_log, noev_prior_log = self._compute_ev_prior()
                np.savez(cachefile, ev_prior_log=ev_prior_log, noev_prior_log=noev_prior_log)
                print "saved prior cache to", cachefile
            self.cached_ev_prior_log, self.cached_noev_prior_log = ev_prior_log, noev_prior_log


    def ev_prior(self):
        return self.cached_ev_prior_log, self.cached_noev_prior_log

    def _compute_ev_prior(self):

        global_event_rate=self.global_event_rate #0.00126599049

        #ev_prior = self.create_array(fill_val=bin_prior)
        ev_prior = np.empty((self.lonbins, self.latbins, self.n_depthbins, self.mbbins), dtype=np.float)


        s = Sigvisa()
        depthbin_probs = []
        for k in range(self.n_depthbins):
            min_depth = self.depthbin_bounds[k]
            max_depth = self.depthbin_bounds[k+1]
            dp = lambda depth : np.exp(s.sigmodel.event_depth_prior_logprob(depth))
            binp, binperr = scipy.integrate.quad(dp, min_depth, max_depth, epsabs=1e-6)
            depthbin_probs.append(binp)
        #depth_0_lp = s.sigmodel.event_depth_prior_logprob(0.0)


        for i in range(self.lonbins):
            left_lon = self.left_lon + self.bin_width_deg * i
            right_lon = left_lon + self.bin_width_deg
            print left_lon
            for j in range(self.latbins):
                bottom_lat = self.bottom_lat + self.bin_width_deg * j
                top_lat = bottom_lat + self.bin_width_deg
                def p(lat, lon):
                    wlon, wlat = wrap_lonlat(lon, lat)
                    return np.exp(s.sigmodel.event_location_prior_logprob(wlon, wlat, -1))

                binp, binperr = scipy.integrate.dblquad(p, left_lon, right_lon, lambda x : bottom_lat, lambda x : top_lat, epsrel=1.49e-02, epsabs=1.49e-06)

                ev_prior[i,j,:,:] = binp
                
        # normalize to give a prior prior over event location for a single event
        #ev_prior /= np.sum(ev_prior[:,:,0,0])

        mbbin_prior_dist = scipy.stats.expon(loc=2.5, scale=1.0/np.log(10.0))
        mbbin_prior_probs = [mbbin_prior_dist.cdf( self.min_mb+(i+1)*self.mb_bin_width) - mbbin_prior_dist.cdf(self.min_mb+i*self.mb_bin_width) for i in range(self.mbbins) ]
        

        #bin_centers = np.linspace(self.min_mb+self.mb_bin_width/2.0, self.max_mb-self.mb_bin_width/2.0, self.mbbins)
        #mbbin_prior_probs = [self.mb_bin_width * np.exp(mbbin_prior_dist.log_p(mb)) for mb in bin_centers]
        for mbbin in range(self.mbbins):
            ev_prior[:,:,:,mbbin] *= mbbin_prior_probs[mbbin]

        for depthbin in range(self.n_depthbins):
            ev_prior[:,:,depthbin,:] *= depthbin_probs[depthbin]

        # scale the location prior by the probability of an event ocurring worldwide in this time period
        ev_prior *= self.time_tick_s * global_event_rate

        ev_prior_log = np.log(ev_prior)
        noev_prior_log = np.log(1-ev_prior)
        return ev_prior_log, noev_prior_log


    def create_array(self, with_phases=False, dtype=np.float, fill_val=None):
        if with_phases:
            dims = (self.lonbins, self.latbins, self.n_depthbins, self.mbbins, self.timebins, len(self.phaseids))
        else:
            dims = (self.lonbins, self.latbins, self.n_depthbins, self.mbbins, self.timebins)

        array = np.empty(dims, dtype=dtype)
        if fill_val is not None:
            array.fill(fill_val)
        return array

    def coords_to_index(self, coords):
        lon, lat, depth, mb, t = coords

        # subtract epsilon from the bin guesses so we don't give nonexistent bins.
        # eg with two depth bins, width 350km, if depth==700 then
        # this calculation gives int(2.0)=2 which fails, while
        # returning int(2.0-1e-8) = 1 which is correct.
        # on the other end, we'd have int(0-1e-8) = 0 which is fine.
        lonbin = int((lon-self.left_lon) / self.bin_width_deg  - 1e-8)
        latbin = int((lat-self.bottom_lat) / self.bin_width_deg - 1e-8)
        timebin = int((t-self.stime) / self.time_tick_s - 1e-8)
        mbbin = int((mb-self.min_mb) / self.mb_bin_width- 1e-8)

        depthbin = np.searchsorted(self.depthbin_bounds, depth, side="right") - 1
        if depth == self.depthbin_bounds[-1]:
            depthbin = self.n_depthbins-1

        return (lonbin, latbin, depthbin, mbbin, timebin)

    def index_to_coords(self, v):
        left_lon = self.left_lon + v[0] * self.bin_width_deg
        right_lon = left_lon + self.bin_width_deg

        bottom_lat = self.bottom_lat + v[1] * self.bin_width_deg
        top_lat = bottom_lat + self.bin_width_deg

        min_depth = self.depthbin_bounds[v[2]]
        max_depth = self.depthbin_bounds[v[2]+1]

        min_mb = self.min_mb + v[3] * self.mb_bin_width
        max_mb = min_mb + self.mb_bin_width

        min_time = self.stime + v[4] * self.time_tick_s
        max_time = min_time + self.time_tick_s

        return (left_lon, right_lon), (bottom_lat, top_lat), (min_depth, max_depth),  (min_mb, max_mb), (min_time, max_time)

    def precompute_ttime_grid(self, sta):
        key = sta
        if key not in self.ttime_cache:
            tmp1, tmp2= [], []
            for phaseid in self.phaseids:
                grid_centered = self._travel_times(sta, phaseid=phaseid, centered=True)
                grid_corners = self._travel_times(sta, phaseid=phaseid, centered=False)
                tmp1.append(grid_centered[:,:,:,np.newaxis])
                tmp2.append(grid_corners[:,:,:,np.newaxis])
            ttimes_centered = np.concatenate(tmp1, axis=3)
            ttimes_corners = np.concatenate(tmp2, axis=3)
            self.ttime_cache[key] =ttimes_centered, ttimes_corners
        return self.ttime_cache[key]

    def precompute_amp_transfer_grid(self, sg, sta, chan, band):
        key = (sta, chan, band)
        if key not in self.amp_transfer_cache:
            tmp1, tmp2 = [], []
            for phase in self.phases:
                model = get_amp_transfer_model(sg, sta, phase, chan, band)
                ats, atvs = self._amp_transfers(model)
                tmp1.append(ats[:,:,:,np.newaxis])
                tmp2.append(np.sqrt(atvs[:,:,:,np.newaxis]))
            amp_transfers = np.concatenate(tmp1, axis=3)
            amp_transfer_stds = np.concatenate(tmp2, axis=3)
            self.amp_transfer_cache[key] = amp_transfers, amp_transfer_stds
        return self.amp_transfer_cache[key]

    def precompute_detprob_grid(self, sg, sta, chan, band):

        s = Sigvisa()
        slon, slat = s.earthmodel.site_info(sta, 0)[:2]

        key = (sta, chan, band, self.phases)
        if key not in self.detprob_cache:

            ttimes_centered, ttimes_corners = self.precompute_ttime_grid(sta)
            tt_mask = np.asarray(np.isfinite(ttimes_centered), dtype=np.int)

            #at_grid, at_std_grid = self.precompute_amp_transfer_grid(sg, sta, chan, band)
            #wn = sg.get_wave_node_by_atime(sta, band, chan, self.stime, allow_out_of_bounds=True)

            # assume we detect iff the energy is one stddev above the noise mean
            #noise_base = np.log(wn.nm_env.c + np.sqrt(wn.nm_env.marginal_variance()))

            eps = 1e-8

            detprobs = np.empty((self.lonbins, self.latbins, self.n_depthbins, self.mbbins, len(self.phases)), dtype=float)
            detprobs.fill(eps)

            mb_bounds = np.linspace(self.min_mb, self.max_mb, self.mbbins+1)
            mbs = (mb_bounds[:-1] + mb_bounds[1:])/2.0
            for phase_idx in range(len(self.phases)):
                model = load_detprob_model(sta, self.phases[phase_idx], self.phases, sg.runids[0])

                for i in range(self.lonbins):
                    lon = self.left_lon + (i+.5) * self.bin_width_deg
                    for j in range(self.latbins):
                        lat = self.bottom_lat + (j+.5) * self.bin_width_deg
                        dist = dist_km((lon, lat), (slon, slat))
                        for k in range(self.n_depthbins):
                            # skip phases that are not defined for this lon, lat, depth
                            if not tt_mask[i,j,k,phase_idx]: continue

                            depth = self.depthbin_bounds[k] + .5* self.depthbin_widths[k]
                            for mbbin, mb in enumerate(mbs):
                                x = (mb, np.exp(mb), depth, np.sqrt(depth), dist, np.log(dist))
                                predprob = model.predict_prob(x)
                                predprob = min(1-eps, max(eps, predprob))
                                detprobs[i, j, k, mbbin, phase_idx] = predprob
                                
            self.detprob_cache[key] = detprobs
        return self.detprob_cache[key]


    def source_logamps(self, band):
        return np.array([[brune.source_logamp(mb=mb, band=band, phase=phase) for phase in self.phases ] for mb in np.linspace(self.min_mb, self.max_mb, self.mbbins+1) ], dtype=np.float)

    def _amp_transfers(self, model):
        at_means = np.zeros((self.lonbins, self.latbins, self.n_depthbins), dtype=float)
        at_vars = np.zeros((self.lonbins, self.latbins, self.n_depthbins), dtype=float)

        d = {'lon': 0.0, 'lat': 0.0, 'depth': 0.0, 'mb': 4.0}
        for i in range(self.lonbins):
            lon = self.left_lon + (i+.5) * self.bin_width_deg
            d['lon'] = lon
            for j in range(self.latbins):
                lat = self.bottom_lat + (j+.5) * self.bin_width_deg
                d['lat'] = lat
                for k in range(self.n_depthbins):
                    depth = self.depthbin_bounds[k] + .5* self.depthbin_widths[k]
                    d['depth'] = depth
                    at_means[i,j, k] = model.predict(d)
                    at_vars[i,j, k] = model.variance(d, include_obs=True)
        return at_means, at_vars

    def _travel_times(self, sta, phaseid=1, centered=False):
        s = Sigvisa()

        offset = .5 if centered else 0.0
        extra_bin = 0 if centered else 1

        times = np.zeros((self.lonbins+extra_bin, self.latbins+extra_bin, self.n_depthbins+extra_bin), dtype=float)

        for i in range(self.lonbins+extra_bin):
            lon = self.left_lon + (i+offset) * self.bin_width_deg
            for j in range(self.latbins+extra_bin):
                lat = self.bottom_lat + (j+offset) * self.bin_width_deg
                for k in range(self.n_depthbins+extra_bin):
                    depth = self.depthbin_bounds[k] 
                    if centered:
                        depth += offset * self.depthbin_widths[k]
                    try:
                        meantt = s.sigmodel.mean_travel_time(lon, lat, depth, 0.0, sta, phaseid - 1)
                        if meantt < 0:
                            raise ValueError
                        times[i,j, k] = meantt
                    except ValueError:
                        times[i,j, k] = np.inf

        return times



def station_hough(sg, hc, wn, uatemplates, fill_assoc=False):

    sta, chan, band = wn.sta, wn.chan, wn.band
    atimes, amps, tmids = uatemplates
    array = hc.create_array(dtype=np.float32)


    t0 = time.time()
    ttimes_centered, ttimes_corners = hc.precompute_ttime_grid(sta)
    t1 = time.time()
    amp_transfer_means, amp_transfer_stds = hc.precompute_amp_transfer_grid(sg, sta, chan, band)
    t2 = time.time()

    valid_len = np.sum([wn.valid_len for wn in sg.station_waves[sta]])

    tg = sg.template_generator(phase="UA")
    ua_amp_model = tg.unassociated_model(param="coda_height", nm=wn.nm_env)

    source_logamps = np.array([[brune.source_logamp(mb=mb, band=band, phase=phase) for phase in hc.phases ] for mb in np.linspace(hc.min_mb, hc.max_mb, hc.mbbins+1) ], dtype=np.float)

    detprobs = hc.precompute_detprob_grid(sg, sta, chan, band)

    print sta
    #for mbbin in range(hc.mbbins):
    #    fname = "/home/dmoore/public_html/detprobs_%s_%d_%d.png" % (sta, mbbin, 0)
    #    visualize_hough_array(None, [sta,], fname=fname, location_array=detprobs[:,:,0,mbbin,0], normalize=False)
    #    print fname
  

    #source_logamps[0] = -np.inf
    #source_logamps[-1] = np.inf

    lognoise = float(np.log(wn.nm_env.c))
    null_ll, full_assoc, phase_scores = generate_sta_hough(sta, array,
                                                           atimes, amps, tmids,
                                                           ttimes_centered,
                                                           ttimes_corners,
                                                           amp_transfer_means,
                                                           amp_transfer_stds,
                                                           detprobs, hc.stime,
                                                           hc.time_tick_s,
                                                           hc.bin_width_deg,
                                                           hc.mb_bin_width,
                                                           hc.min_mb,
                                                           hc.bin_width_km,
                                                           source_logamps,
                                                           hc.uatemplate_rate,
                                                           ua_amp_model, valid_len,
                                                           lognoise=lognoise,
                                                           save_assoc=fill_assoc)
    t3 = time.time()
    #print sta, "hough", t1-t0, t2-t1, t3-t2

    return array, full_assoc, phase_scores, null_ll


def normalize_global(global_array, global_noev_ll, hc, one_event_semantics=False):
    # take an array of p(templates | ev in bin) likelihoods, unnormalized,
    # and convert to p(ev in bin | templates)

    ev_prior_log, noev_prior_log = hc.ev_prior()
        
    #ev_prior_log = np.log(ev_prior)

    global_array += ev_prior_log[:,:,:,:,np.newaxis]
    armax = np.max(global_array)
    global_lik = np.exp(global_array-armax)

    if one_event_semantics:
        # treat the array as distribution on location of a single event
        return global_lik/np.sum(global_lik)
    else:
        # treat the array as probability that an event is in each bin,
        # independent of other bins
        #noev_prior_log = np.log(1-ev_prior)
        global_noev_lik = np.exp(global_noev_ll-armax + noev_prior_log)
        normalizer = global_lik + global_noev_lik[:,:,:,:,np.newaxis]
        global_lik /= normalizer
        return global_lik

def global_hough(sg, hc, uatemplates_by_sta, save_debug=False, save_debug_stas=False):
    t0 = time.time()
    global_array = hc.create_array(dtype=np.float32, fill_val=0.0)
    global_debug = {}
    global_noev_ll = 0
    t1 = time.time()


    sta_hough_time = 0
    adj_time = 0


    
    for wn, uatemplates in uatemplates_by_sta.items():
        sta = wn.sta
        t2 = time.time()
        sta_array, assocs, phase_scores, null_ll = station_hough(sg, hc, wn, uatemplates, fill_assoc=save_debug)
        t3 = time.time()
        sta_hough_time += t3-t2

        global_debug[wn.label]=(assocs, phase_scores, null_ll)
        global_array += sta_array
        global_noev_ll += null_ll
        t4 = time.time()
        adj_time += t4-t3

        if save_debug_stas:
            sta_array = np.exp(sta_array - np.max(sta_array))
            fname = "newhough_%d_%s" % (hc.bin_width_deg, sta)
            visualize_hough_array(sta_array, [sta], fname=fname+".png", ax=None, timeslice=None)
            np.save(fname + ".npy", sta_array)
            print "wrote to", fname


    if save_debug:
        global_lik = normalize_global(global_array.copy(), global_noev_ll, one_event_semantics=False, hc=hc)
        fname = "newhough_%d_global.png" % hc.bin_width_deg
        visualize_hough_array(global_lik, sg.station_waves.keys(), fname=fname, ax=None, timeslice=None)
        print "wrote to", fname

        global_dist = normalize_global(global_array.copy(), global_noev_ll, one_event_semantics=True, hc=hc)
        fname = "newhough_%d_global_norm.png" % hc.bin_width_deg
        visualize_hough_array(global_dist, sg.station_waves.keys(), fname=fname, ax=None, timeslice=None)
        print "wrote to", fname

    t5 = time.time()
    print "global hough: init %f stahough %f adj %f total %f" % (t1-t0, sta_hough_time, adj_time, t5-t0)
    return global_array, global_debug, global_noev_ll

def debug_assocs(sg, hc):


    np.random.seed(2)
    uatemplates_by_sta, evs = synth_templates(sg)
    print "true evs"
    for ev in evs:
        print ev

    #uatemplates_by_sta = get_uatemplates(sg)
    array,debug, nll = global_hough(sg, hc, uatemplates_by_sta, save_debug=True)

    def print_bin(bin):
        print "bin", bin, "center", "score", array[bin]-nll, "(%f-%f)" % (array[bin], nll)
        print " center ", (bin[0]+.5)*hc.bin_width_deg+hc.left_lon, (bin[1]+.5)*hc.bin_width_deg+hc.bottom_lat, hc.depthbin_bounds[bin[2]] + .5*hc.depthbin_widths[bin[2]], (bin[3]+.5)*hc.time_tick_s+hc.stime
        for sta in debug.keys():
            print " ", sta + ": ",
            for phaseidx in range(2):
                bidx = tuple(list(bin) + [phaseidx,])
                tmid = debug[sta][0][bidx]
                print "None" if tmid==255 else uatemplates_by_sta[sta][0][tmid],
            print


    flatarray = array.flatten()
    idxs = np.arange(len(flatarray))[flatarray > np.max(flatarray) - 5]
    print idxs
    bins = [np.unravel_index(idx, array.shape) for idx in idxs]
    for bin in bins:
        print_bin(bin)


def score_assoc(sg, hc, uatemplates_by_sta, debug, bin_idx):
    coords = hc.index_to_coords(bin_idx)
    left, right = zip(*coords)
    left, right = np.asarray(left), np.asarray(right)
    clon, clat, cdepth, cmb, ctime = left + (right-left)/2.0    

    full_score = 0
    gnll =0
    for wn_label in debug.keys():
        wn = sg.all_nodes[wn_label]
        sta = wn.sta

        assocs, scores, nll = debug[wn_label]
        atimes, amps, tmids = uatemplates_by_sta[wn]
        
        for i_phase, phase in enumerate(hc.phases):

            bidx = list(bin_idx)
            bidx.append(i_phase)
            bidx = tuple(bidx)

            tm_idx = int(assocs[bidx])-1
            score = scores[bidx]
            tmid = tmids[tm_idx] if tm_idx > 0 else None

            detprobs = hc.precompute_detprob_grid(sg, sta, wn.chan, wn.band)
            dbin = (bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3], i_phase)
            detprob = detprobs[dbin]
            dscore = np.log(1-detprob)

            print sta, phase, "assoc", tmid, score, dscore, score+dscore

            full_score += score +dscore

        gnll += nll


        #ttgrid_centered, ttgrid_corners = hc.precompute_ttime_grid(sta)

        """
        loc_bin = bin_idx[:3]
        tt = ttgrid_centered[loc_bin]

        assocs, scores, nll = debug[sta][0]
        tm_idx = int(assocs[bin_idx])-1
        if tm_idx<0: continue
        atimes, amps, tmids = uatemplates_by_sta[sta]
        atime, amp = atimes[tm_idx], amps[tm_idx]
        
        pred_atime = ctime + tt
        atime_residual = atime-pred_atime
        print sta, atime_residual, detprob, np.log(detprob/(1-detprob)), 
        """
    print "full score", full_score
    print "predicted total", full_score+gnll

class CTFProposer(object):

    def __init__(self, sg, bin_widths, depthbin_bounds, mbbins, 
                 phases=("P",), offset=False, stime =None, 
                 etime=None, **kwargs):
        # precompute ttime and amp_transfer patterns
        global_bin_width = bin_widths[0]

        if sg.inference_region is not None:
            stime = sg.inference_region.stime if stime is None else stime
            etime = sg.inference_region.etime if etime is None else etime
            left_lon, right_lon = sg.inference_region.left_lon, sg.inference_region.right_lon
            bottom_lat, top_lat = sg.inference_region.bottom_lat, sg.inference_region.top_lat

        else:
            left_lon, right_lon, bottom_lat, top_lat = -180, 180, -90, 90
            stime = sg.event_start_time  if stime is None else stime
            try:
                etime = sg.event_end_time if etime is None else etime
            except AttributeError:
                print "WARNING: no event end time specified"
                etime = sg.end_time

        self.bin_widths = bin_widths
        self.stime = stime
        self.etime = etime
        self.phases = phases
        self.depthbin_bounds = depthbin_bounds
        self.mbbins = mbbins


        if offset:
            left_lon += global_bin_width/2.0
            right_lon += global_bin_width/2.0
            top_lat += global_bin_width/2.0
            bottom_lat += global_bin_width/2.0


        # we DON'T want to use the sg uatemplate rate because our
        # model is a bit different.  in the Hough
        # non-one-event-semantics setup, we consider the probability
        # of an event in *each bin* independently of the other
        # bins. so if uatemplates are really unlikely, will give high
        # probability to events in even really bad bins, as long as
        # they sort-of explain a few uatemplates. whereas the real
        # sigvisa model would a) be much stricter about things like
        # traveltimes, and b) would immediately move any event in 
        # a bad bin towards a better area - so it really would put
        # close-to-zero probability on an event in that bin.
        uatemplate_rate = 1e-3

        hc = HoughConfig(stime, etime-stime, bin_width_deg=global_bin_width,
                         phases=phases, depthbin_bounds=depthbin_bounds, time_tick_s = 20.0,
                         mbbins=mbbins[0],
                         top_lat=top_lat, bottom_lat=bottom_lat,
                         left_lon=left_lon, right_lon=right_lon,
                         uatemplate_rate=uatemplate_rate, 
                         global_event_rate=sg.event_rate,
                         **kwargs)
        self.global_hc = hc

    def propose_event(self, sg, uatemplates_by_sta=None, 
                      fix_result=None, one_event_semantics=True, 
                      fixed_return_global_dist=False):
        if uatemplates_by_sta is None:
            uatemplates_by_sta = get_uatemplates(sg)

        hc = self.global_hc

        global_array,debug, nll = global_hough(sg, hc, uatemplates_by_sta, save_debug=False)

        # use 64-bit floats to avoid underflow in normalization
        ga64 = np.array(global_array, dtype=np.float64)
        global_dist = normalize_global(ga64, nll, one_event_semantics=one_event_semantics, hc=hc)

        if fix_result:
            ev = fix_result
            coord = (ev.lon, ev.lat, ev.depth, ev.mb, ev.time)
            v = hc.coords_to_index(coord)
        else:
            v = categorical_sample_array(global_dist)
        prob = categorical_prob(global_dist, v)
        (left_lon, right_lon), (bottom_lat, top_lat), (min_depth, max_depth), (min_mb, max_mb), _ = hc.index_to_coords(v)

        for i, fine_width in enumerate(self.bin_widths[1:]):
            depth_bounds = [min_depth, min_depth + (max_depth-min_depth)/2.0, max_depth]

            mb_bins = self.mbbins[i+1]
            hc = HoughConfig(self.stime, self.etime-self.stime, bin_width_deg=fine_width, 
                             phases=self.phases,
                             depthbin_bounds=depth_bounds, top_lat=top_lat, bottom_lat=bottom_lat,
                             left_lon=left_lon, right_lon=right_lon, min_depth=min_depth,
                             max_depth=max_depth, time_tick_s = 10.0,
                             min_mb=min_mb, max_mb=max_mb,
                             global_event_rate=hc.global_event_rate,
                             uatemplate_rate=hc.uatemplate_rate,
                             mbbins=mb_bins)
            array,debug, nll = global_hough(sg, hc, uatemplates_by_sta, save_debug=False)
            dist = normalize_global(np.asarray(array, dtype=np.float64), 
                                    nll, 
                                    one_event_semantics=one_event_semantics, 
                                    hc=hc)
            if fix_result:
                v = hc.coords_to_index(coord)
            else:
                v = categorical_sample_array(dist)
            prob *= categorical_prob(dist, v)

            (left_lon, right_lon), (bottom_lat, top_lat), (min_depth, max_depth), (min_mb, max_mb), _ = hc.index_to_coords(v)

        if fix_result:
            _, evlp = event_from_bin(hc, v)
            if fixed_return_global_dist:
                return evlp, global_dist
            return evlp
        else:
            ev, evlp = event_from_bin(hc, v)

            log_qforward = np.log(prob) + evlp
            
            return ev, log_qforward, global_dist

def hough_location_proposal(sg, fix_result=None, proposal_dist_seed=None,
                            offset=None, one_event_semantics=True, mbbins=None,
                            phases=None, stime=None, etime=None, bin_width_multiplier=1.0):
    s = Sigvisa()
    if proposal_dist_seed is not None:
        np.random.seed(proposal_dist_seed)

    #if offset is None:
    #    # random choice of proposal distribution, not based on current
    #    # state so cancels out in the acceptance ratio
    #    offset = np.random.choice([True, False])
    #if one_event_semantics is None:
    #    one_event_semantics = np.random.choice([True, False])

    if phases is None:
        phases = ("Pg", "Lg") #sg.phases

    if mbbins is None:
        mbbins = (12, 2, 2)

    phases = tuple(sorted(phases))

    settings = (offset, phases, stime, etime, bin_width_multiplier, mbbins)
    try:
        ctf = s.hough_proposer[settings]
    except KeyError:
        bin_widths = [10,5,2]        
        if sg.inference_region is not None:
            bin_area = sg.inference_region.area_deg() / 1500.
            bw1 = np.round(np.sqrt(bin_area) * bin_width_multiplier, decimals=2)
            bin_widths = [bw1, bw1/2., bw1/4.]

        ctf = CTFProposer(sg, bin_widths, phases=phases,
                          depthbin_bounds=[0,10,50,150,400,700], 
                          mbbins=mbbins, offset=offset, 
                          min_mb=sg.min_mb, 
                          stime=stime, etime=etime)
        s.hough_proposer[settings] = ctf

    r = ctf.propose_event(sg, fix_result=fix_result,
                           one_event_semantics=one_event_semantics)

    #if fix_result is None:
    #    ev = Event(lon=-105.427, lat=43.731, depth=0.0, time=1239041017.07, mb=4.0)
    #    lp, global_dist = ctf.propose_event(sg, fix_result=ev,one_event_semantics=one_event_semantics, fixed_return_global_dist=True)
    #    r = ev, lp, global_dist
    return r




def main():
    #sfile = "/home/dmoore/python/sigvisa/logs/mcmc/02135/step_000000/pickle.sg"
    sfile = "/home/dmoore/python/sigvisa/logs/mcmc/02138/step_000489/pickle.sg"
    with open(sfile, 'rb') as f:
        sg = pickle.load(f)
    print "read sg"

    august_debug(sg)

    """
    np.random.seed(1)
    t0 = time.time()
    ctf = CTFProposer(sg, [5,], depthbins=2, mbbins=1, offset=False)
    ev, evlp, _ = ctf.propose_event(sg)
    t1 = time.time()
    #ev, evlp, _ = ctf.propose_event(sg)
    t2 = time.time()

    print "built in", t1-t0, "sampled in", t2-t1
    print ev
    print evlp
    """

    #_, evlp2 = ctf.propose_event(sg, fix_result=ev)
    #print evlp2
    #visualize_coarse_to_fine(sg, bc, sg.station_waves.keys())


if __name__ =="__main__":
    main()
