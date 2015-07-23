import numpy as np
import sys
import os
import traceback
import itertools
import pickle
import time

from sigvisa.models.ttime import tt_predict
from sigvisa.models.distributions import Laplacian
from sigvisa.graph.sigvisa_graph import SigvisaGraph, ModelNotFoundError
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.source.event import Event, get_event
import sigvisa.source.brune_source as brune

from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from sigvisa.plotting.plot import savefig

import scipy.weave as weave
from scipy.weave import converters


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

travel_time_cache = dict()
amp_transfer_cache = dict()


class HoughArray(object):

    def __init__(self, stime, etime, bin_width_deg=3.0, latbins=None, time_tick_s=None):
        if latbins is None:
            latbins = int(180.0 / bin_width_deg)
        if time_tick_s is None:
            time_tick_s = bin_width_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S

        self.latbins = latbins
        self.time_tick_s = time_tick_s
        self.stime = stime
        self.etime = etime


def init_hough_array(stime, etime, time_tick_s=20, latbins=18, sta_array=False, prev_array=None):
    """

    Initialize a new Hough accumulator array.

    If this array is specific to a station, initialize all bins to a
    logprob of log(.5): this corresponds to saying that the probability of
    a cell goes down by .5 for every station at which we don't find a
    corresponding detection.

    If this is a global array, we init to a logprob of 0.0, since this
    will just be used to sum the station arrays.

    """

    lonbins = latbins * 2
    timebins = int((etime-stime)/time_tick_s)
    #init_val = -.7 if sta_array else 0.0
    #init_val = np.exp(-.7) if sta_array else 1.0

    bins_total = latbins * lonbins * timebins
    init_val = 0.0 if sta_array else 0.0

    if prev_array is None:
        hough_array = np.empty((lonbins, latbins, timebins))
    else:
        hough_array=prev_array
    hough_array.fill(init_val)

    return hough_array

def precompute_amp_transfer(model, latbins=18, depthbins=1):
    lonbins = latbins * 2
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins
    depthbin_km = 700.0/depthbins

    at_means = np.zeros((lonbins, latbins, depthbins), dtype=float)
    at_vars = np.zeros((lonbins, latbins, depthbins), dtype=float)

    d = {'lon': 0.0, 'lat': 0.0, 'depth': 0.0, 'mb': 4.0}
    for i in range(lonbins):
        lon = -180.0 + (i+.5) * lonbin_deg
        d['lon'] = lon
        for j in range(latbins):
            lat = -90.0 + (j+.5) * latbin_deg
            d['lat'] = lat
            for k in range(depthbins):
                #depth = 0 + (k+.5)*depthbin_km
                depth = 0 + k*depthbin_km
                d['depth'] = depth
                at_means[i,j, k] = model.predict(d)
                at_vars[i,j, k] = model.variance(d, include_obs=True)
    return at_means, at_vars


def precompute_travel_times(sta, phaseid=1, latbins=18, depthbins=1):
    s = Sigvisa()

    lonbins = latbins * 2
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins
    depthbin_km = 700.0/depthbins
    times = np.zeros((lonbins, latbins, depthbins), dtype=float)

    for i in range(lonbins):
        lon = -180.0 + (i+.5) * lonbin_deg
        for j in range(latbins):
            lat = -90.0 + (j+.5) * latbin_deg
            for k in range(depthbins):
                #depth = (k+.5) * depthbin_km
                depth = k * depthbin_km
                try:
                    meantt = s.sigmodel.mean_travel_time(lon, lat, depth, 0.0, sta, phaseid - 1)
                    if meantt < 0:
                        raise ValueError
                    times[i,j, k] = meantt
                except ValueError:
                    times[i,j, k] = -1e9

    return times


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

def template_amp_transfer(sg, wn, phase, latbins=18, depthbins=1):
    key = (wn.sta, wn.chan, wn.band, phase, latbins, depthbins)
    if key not in amp_transfer_cache:
        sta, band, chan = wn.sta, wn.band, wn.chan
        model = get_amp_transfer_model(sg, sta, phase, chan, band)
        amp_transfer_cache[key]= precompute_amp_transfer(model, latbins=latbins, depthbins=depthbins)
    at_means, at_vars = amp_transfer_cache[key]
    return at_means, at_vars


def template_origin_times(sta, time, phaseid=1, latbins=18, depthbins=1):
    """

    Return, for each spatial bin, the inverted origin time
    corresponding to the center of that bin. This is used below (in
    add_template_to_sta_hough) to determine which time bin(s) this
    template should "vote" for at each point in space.

    """
    key = (sta, phaseid, latbins, depthbins)
    if key not in travel_time_cache:
        travel_time_cache[key]= precompute_travel_times(sta, phaseid=phaseid, latbins=latbins, depthbins=depthbins)
    ttimes = travel_time_cache[key]
    return time - ttimes

def generate_sta_hough(sta_hough_array, atimes, amps, ttimes, amp_transfers, amp_transfer_stds,
                       stime, time_tick_s, bin_width_deg, mb_bin_width, min_mb, bin_width_km,
                       source_logamps, uatemplate_rate, ua_amp_model, valid_len, lognoise, save_assoc=False):
    lonbins, latbins, depthbins, timebins, mbbins = sta_hough_array.shape
    time_radius = time_tick_s / 2.0
    lonbin_deg = 360.0/lonbins

    ntemplates = len(atimes)

    amps = np.asarray(amps, dtype=np.float)
    atimes = np.asarray(atimes, dtype=np.float)

    ua_amp_lps = np.array([ua_amp_model.log_p(amp) for amp in amps])

    ua_poisson_lp_full = ntemplates * np.log(uatemplate_rate) - (uatemplate_rate * valid_len)
    ua_poisson_lp_incr = float(np.log(uatemplate_rate)) # change in the ua_poisson_lp if we have one fewer uatemplate

    detection_probs = np.array([0.8, 0.4], dtype=np.float) # TODO calibrate by magnitude, phaseid
    detection_lps = np.log(detection_probs)
    nodetection_lps =np.log(1-detection_probs)

    null_ll = np.sum(nodetection_lps) + ua_poisson_lp_full + np.sum(ua_amp_lps)


    cdfs = """
    // cdf of Laplace(0.0, 5.0) computed for integer x values [-40, 40] inclusive
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
        int ix = (int) floor(x*10);
        int ix2 = ix+1;
        if (ix2 <= -40) return 0.0;
        if (ix >= 40) return 1.0;
        double tbl1 = gaussian_table[ix+40];
        double tbl2 = gaussian_table[ix2+40];
        double y = x-ix;
        return (1-y)*tbl1 + y*tbl2;
    }
    """

    # ttimes, amp_transfers, amp_transfer_stds: loc bins
    # phase_score: timebin*mbbin
    # assoc: int, nphases*timebin*mbbin
    # atimes, amps
    # background_cost_lp
    # detection

    nphases = ttimes.shape[3]
    phase_score = np.empty((timebins, mbbins))
    assoc = np.empty((nphases, timebins, mbbins), dtype=np.uint8)

    save_assoc = 1 if save_assoc else 0 # make c++ happy
    if save_assoc:
        full_assoc = np.empty((lonbins, latbins, depthbins, timebins, mbbins, nphases,), dtype=np.uint8)
    else:
        full_assoc = np.empty((), dtype=np.uint8)


    sta_hough_array.fill(null_ll)

    #print "source logamps"
    #print source_logamps

    debug_lon = None
    debug_lat = None
    debug_depth = 0
    debug_lonbin = -1
    debug_latbin = -1
    debug_depthbin = -1
    if debug_lon is not None:
        debug_lonbin = int(np.floor((debug_lon + 180) / bin_width_deg))
        debug_latbin = int(np.floor((debug_lat + 90) / bin_width_deg))
        debug_depthbin  = int(np.floor(debug_depth / bin_width_km))

    code = """

for (int i=0; i < lonbins; ++i) {
    for (int j=0; j < latbins; ++j) {
        for (int k=0; k < depthbins; ++k) {
            bool verbose = (i==debug_lonbin && j==debug_latbin && k==debug_depthbin);
            for (int phaseidx=0; phaseidx < nphases; ++phaseidx) {
                double ttime = ttimes(i,j, k, phaseidx);
                double amp_transfer = amp_transfers(i, j, k, phaseidx);
                double amp_transfer_std = amp_transfer_stds(i, j, k, phaseidx);

                // TODO use memset
                for (int timebin=0; timebin < timebins; ++timebin) {
                    for (int mbbin=0; mbbin < mbbins; ++mbbin) {
                        phase_score(timebin, mbbin) = 0;
                        assoc(phaseidx, timebin, mbbin) = -1;
                    }
                }

                for (int t=0; t < ntemplates; ++t) {
                    double atime = atimes(t);
                    double amp = amps(t);
                    double origin_time = atime - ttime;
                    double source_logamp = amp - amp_transfer;


                    if (origin_time < stime  || origin_time > stime + timebins*time_tick_s) {
                       if (verbose) {
                          printf("skipping t=%d for implausible origin time %.1f\\n", t, origin_time);
                       }
                       continue;
                    }

                    int min_plausible_timebin = std::max(0, int((origin_time - 30.0-stime) / time_tick_s));
                    int max_plausible_timebin = std::min(timebins-1, int((origin_time + 30.0-stime) / time_tick_s));


                    int min_plausible_mbbin = 0;
                    int max_plausible_mbbin = -1;
                    double min_logamp = source_logamp - 3*amp_transfer_std;
                    double max_logamp = source_logamp + 3*amp_transfer_std;
                    for (int mbbin=0; mbbin < mbbins; ++mbbin) {
                        double bin_logamp = (source_logamps(mbbin, phaseidx) + source_logamps(mbbin+1, phaseidx))/2.0;
                        if (bin_logamp < min_logamp) { min_plausible_mbbin = mbbin+1; }
                        if (bin_logamp > min_logamp && bin_logamp < max_logamp) { max_plausible_mbbin = mbbin; }
                    }

                    //int min_plausible_mbbin = std::max(0, int((mb-3*amp_transfer_std - min_mb) / mb_bin_width) );
                    //int max_plausible_mbbin = std::min(mbbins-1, int((mb+3*amp_transfer_std - min_mb) / mb_bin_width) );

                    if (verbose) {
                       printf("plausible timebins %d %d\\n", min_plausible_timebin, max_plausible_timebin);
                      printf("plausible mbbins %d %d (amp %.1f at %.1f std %.1f) (logamp range %.1f %.1f %.1f)\\n", min_plausible_mbbin, max_plausible_mbbin, amp, amp_transfer, amp_transfer_std, min_logamp, source_logamp, max_logamp);
                    }
                    // every bin should track p(event | best assoc) and idx(best assoc)
                    // for each phase, this means we track the "score" of the best assoc.
                    // there is a default score corresponding to the "none" assoc.
                    for (int mbbin = min_plausible_mbbin; mbbin <= max_plausible_mbbin; mbbin++) {
                        double logamp_left = source_logamps(mbbin, phaseidx);
                        double logamp_right = source_logamps(mbbin+1, phaseidx);
                        double at_residual_left =  (amp - logamp_left) - amp_transfer;
                        double at_residual_right = (amp - logamp_right) - amp_transfer;
                        double bin_weight = gaussian_cdf(at_residual_left/amp_transfer_std)
                                               - gaussian_cdf(at_residual_right/amp_transfer_std);
                        double mb_lp = log(bin_weight) - log(mb_bin_width);

                        // probability the signal is above the noise floor
                        /*
                        double noise_z_left = (lognoise - 1 - ( logamp_left + amp_transfer))/amp_transfer_std;
                        double noise_z_right = (lognoise - 1 - ( logamp_right + amp_transfer))/amp_transfer_std;
                        double detection_prob = .5*gaussian_cdf(noise_z_left) + .5*gaussian_cdf(noise_z_right);
                        detection_prob = detection_prob < 0.01 ? 0.01 : detection_prob;
                        detection_prob = detection_prob > 0.99 ? 0.99 : detection_prob;
                        */
                        // double logdet_odds = log(detection_prob / (1-detection_prob));
                        double logdet_odds = (detection_lps(phaseidx) - nodetection_lps(phaseidx));

                        for (int timebin=min_plausible_timebin; timebin <= max_plausible_timebin; timebin++) {
                            // p(atime | bin)
                            double timebin_left = (stime + timebin * time_tick_s) - origin_time;
                            double timebin_right = timebin_left + time_tick_s;
                            double bin_weight = laplace_cdf(timebin_right) - laplace_cdf(timebin_left);
                            double atime_lp = log(bin_weight) - log(time_tick_s);



                            // don't allow templates that have previously
                            // been associated with other phases.
                            bool exclude_tmpl_from_bin = 0;
                            for (int oldphase = 0; oldphase < phaseidx; ++oldphase) {
                                if (assoc(oldphase, timebin, mbbin) == t) {
                                    exclude_tmpl_from_bin = 1;
                                    break;
                                }
                            }
                            if (exclude_tmpl_from_bin) { continue; }

                            double tmpl_score = logdet_odds + (mb_lp - ua_amp_lps(t)) + atime_lp - ua_poisson_lp_incr;
                            double oldval = phase_score(timebin, mbbin);

                            if (verbose) {
                                  printf("lon %d lat %d depth %d time %d mbbin %d phase %d processing template %d with score %f vs oldval %f (ttr %.1f-%.1f, atr %.1f-%.1f, amp %.1f)\\n", i, j, k, timebin, mbbin, phaseidx, t, tmpl_score, oldval, timebin_left, timebin_right, at_residual_left, at_residual_right, amp);
                            }

                            if (tmpl_score > oldval) {
                                phase_score(timebin, mbbin) = tmpl_score;
                                assoc(phaseidx, timebin, mbbin) = t;
                            }

                        }
                    }
                } // t (template idx)

                for (int timebin=0; timebin < timebins; ++timebin) {
                    for (int mbbin=0; mbbin < mbbins; ++mbbin) {
                        sta_hough_array(i,j,k,timebin,mbbin) +=   phase_score(timebin, mbbin);

                        if (save_assoc) {
                           full_assoc(i,j,k,timebin,mbbin, phaseidx) = assoc(phaseidx, timebin, mbbin);
                        }
                    }
                }

            } // phaseidx
        } // k (depth)
    } // j (latbin)
} // i (lonbin)
    """
    weave.inline(code,['latbins', 'lonbins', 'depthbins', 'timebins', 'mbbins', 'nphases',
                       'sta_hough_array', 'ttimes', 'amp_transfers', 'amp_transfer_stds',
                       'source_logamps', 'phase_score', 'assoc',
                       'bin_width_deg', 'time_tick_s', 'mb_bin_width',
                       'bin_width_km', 'stime', 'min_mb', 'detection_lps', 'nodetection_lps',
                       'amps', 'atimes', 'ntemplates', 'debug_lonbin', 'debug_depthbin', 'debug_latbin', 'ua_poisson_lp_incr', 'ua_amp_lps', 'full_assoc', 'save_assoc', 'lognoise'],
                 support_code=cdfs, headers=["<math.h>", "<unordered_set>"], type_converters = converters.blitz,verbose=2,compiler='gcc', extra_compile_args=["-std=c++11",])
    #print "added template"

    if save_assoc:
        return null_ll, full_assoc
    else:
        return null_ll


def categorical_sample_array(a):
    """

    Sample a bin from the Hough accumulator. Assume that our log
    probabilities have previously been exponentied, and are now just
    probabilities.

    """

    lonbins, latbins, timebins = a.shape
    s = float(np.sum(a))
    u = float(np.random.rand())

    t0 = time.time()
    code = """
double accum = 0;
double goal = u*s;
int done = 0;
for (int i=0; i < lonbins; ++i) {
    for (int j=0; j < latbins; ++j) {
        for (int k=0; k < timebins; ++k)  {
            accum += a(i,j,k);
            if (accum >= goal) {
               return_val = timebins*latbins*i+timebins*j+k;
               done = 1;
               break;
            }
        }
        if (done) { break; }
    }
    if (done) { break; }
}
    """
    v = weave.inline(code,['latbins', 'lonbins', 'timebins', 'a', 'u', 's'],type_converters = converters.blitz,verbose=2,compiler='gcc')

    k = v % timebins
    v1 = (v-k)/timebins
    j = v1 % latbins
    i = (v1-j) / latbins

    return (i,j,k)

def categorical_prob(a, idx):
    s = np.sum(a)
    return a[idx]/s

def event_prob_from_hough(ev, hough_array, stime, etime):
    lonbins, latbins, timebins = hough_array.shape
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins
    time_tick_s = float(etime-stime)/timebins

    timeidx = int(np.floor((ev.time-stime) / time_tick_s))
    lonidx = int(np.floor((ev.lon+180)) / lonbin_deg)
    latidx = int(np.floor((ev.lat+90)) / latbin_deg)
    ev_prob = categorical_prob(hough_array, (lonidx, latidx, timeidx))
    ev_prob /= (lonbin_deg * latbin_deg * time_tick_s)

    return ev_prob

def propose_event_from_hough(hough_array, stime, etime):
    """

    Sample an event using the Hough probabilities. First sample a bin,
    then a uniform location within that bin. (TODO: technically this
    samples a uniform lon/lat, rather than a uniform location, so will
    be a bit problematic near the poles. really this is a broader
    issue with our approach of dividing the world into lon/lat bins
    rather than fixed-area bins).

    """

    lonbins, latbins, timebins = hough_array.shape
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins
    time_tick_s = float(etime-stime)/timebins

    t0 = time.time()
    lonidx, latidx, timeidx = categorical_sample_array(hough_array)
    t1 = time.time()
    ev_prob = categorical_prob(hough_array, (lonidx, latidx, timeidx))
    t2 = time.time()

    # sample an event location uniformly within each bin
    lonidx += np.random.rand()
    latidx += np.random.rand()
    timeidx += np.random.rand()
    ev_prob /= (lonbin_deg * latbin_deg * time_tick_s)

    lon = -180.0 + lonidx * lonbin_deg
    lat = -90.0 + latidx * latbin_deg
    t = stime + timeidx * time_tick_s

    print "proposal time", t1-t0, t2-t1

    ev = Event(lon=lon, lat=lat, time=t, depth=0, mb=4.0, natural_source=True)
    return ev, ev_prob

def visualize_hough_array(hough_array, sites, fname=None, ax=None, timeslice=None):
    """

    Save an image visualizing the given Hough accumulator array. If
    timeslice is the index of a time bin, display only that bin,
    otherwise sum over all time bins.

    """

    lonbins, latbins, depthbins, timebins, mbbins = hough_array.shape
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    if timeslice is None:
        location_array = np.sum(hough_array, axis=(2,3,4))
    else:
        location_array = hough_array[:,:,timeslice]

    # set up the map
    if ax is None:
        fig = Figure(figsize=(8, 5), dpi=300)
        ax = fig.add_subplot(111)

    bmap = Basemap(resolution="c", projection = "robin", lon_0 = 0, ax=ax)
    bmap.drawcoastlines(zorder=10)
    bmap.drawmapboundary()
    parallels = [int(k) for k in np.linspace(-90, 90, 10)]
    bmap.drawparallels(parallels, labels=[False, True, True, False], fontsize=8, zorder=4)
    meridians = [int(k) for k in np.linspace(-180, 180, 10)]
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
        lon = max(-180 + i * lonbin_deg, -180)
        for j in range(latbins):
            lat = max(-90 + j * latbin_deg, -90)
            alpha = location_array[i,j]/m
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
        atimes = []
        amps = []
        tmids = []
        for wn in sg.station_waves[sta]:
            for i, (eid, phase) in enumerate(wn.arrivals()):
                if tmid_whitelist is not None and -i not in tmid_whitelist: continue
                v, _ = wn.get_template_params_for_arrival(eid, phase)
                atimes.append(v['arrival_time'])
                amps.append(v['coda_height'])
                tmids.append(-i)
        uatemplates_by_sta[sta]= (atimes, amps, tmids)

    return uatemplates_by_sta

def synth_templates(sg, n_events=3, phases=("P", "S")):

    evs = [sg.prior_sample_event(3.5, sg.event_start_time, sg.end_time) for i in range(n_events)]

    uatemplates_by_sta = {}
    tmid = 0
    detection_probs = {"P": 0.8, "S": 0.4}
    for sta in sg.station_waves.keys():
        wn_prototype = sg.station_waves[sta][0]
        atimes = []
        amps = []
        tmids = []
        for phase in phases:
            model = get_amp_transfer_model(sg, sta, phase, wn_prototype.chan, wn_prototype.band)
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

                if amp < np.log(wn_prototype.nm.c)-1: continue

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

    global_array, assoc_by_sta, global_null_lp = global_hough(sg, hc, uatemplates_by_sta, save_debug=True)
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
        ev_array = normalize_global(ev_array, 0.0, 1.0, one_event_semantics=True)
        #ev_array *= prob_ev_exists # (expected number of events: can we get a score for this from event likelihood vs unass model?)
        mixture_array += ev_array # expected number of events in a cell

        fname = "newhough_%d_ev%d" % (hc.bin_width_deg, i)
        visualize_hough_array(ev_array, uatemplates_by_sta.keys(), fname=fname+".png", ax=None, timeslice=None)
        print fname

    fname = "newhough_%d_global_mixture" % (hc.bin_width_deg)
    visualize_hough_array(mixture_array, uatemplates_by_sta.keys(), fname=fname+".png", ax=None, timeslice=None)
    print fname

class HoughConfig(object):
    def __init__(self, stime, len_s, phases = ("P", "S"), bin_width_deg=2.0, min_mb=3.0, max_mb=8.0, mbbins=1, depthbins=1, uatemplate_rate=1e-3):
        self.bin_width_deg = bin_width_deg
        self.lonbins = int(360.0/bin_width_deg)
        self.latbins = int(180.0/bin_width_deg)
        self.stime = stime
        #self.time_tick_s = (1.41 * bin_width_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S)
        #self.time_tick_s /= 10
        self.time_tick_s = 10.0
        print "time tick", self.time_tick_s
        self.timebins = int(float(len_s)/self.time_tick_s)
        self.min_mb = min_mb
        self.max_mb = max_mb
        self.mbbins = mbbins
        self.mb_bin_width = float(max_mb-min_mb)/mbbins

        self.max_depth = 700
        self.bin_width_km = 40000.0/self.latbins
        self.depthbins = depthbins #int(np.ceil(max_depth / bin_width_km))
        self.depthbin_width_km = self.max_depth/float(depthbins)

        self.phases = phases
        s = Sigvisa()
        self.phaseids = [s.phaseids[phase] for phase in phases]

        self.uatemplate_rate=uatemplate_rate
        #sites = lonbins*latbins*depthbins
        #bins_at_site = timebins * mbbins
        #print sites, bins_at_site

    def create_array(self, with_phases=False, dtype=np.float, fill_val=None):
        if with_phases:
            dims = (self.lonbins, self.latbins, self.depthbins, self.timebins, self.mbbins, len(self.phaseids))
        else:
            dims = (self.lonbins, self.latbins, self.depthbins, self.timebins, self.mbbins)

        array = np.empty(dims, dtype=dtype)
        if fill_val is not None:
            array.fill(fill_val)
        return array

def station_hough(sg, hc, sta, uatemplates):
    atimes, amps, tmids = uatemplates
    array = hc.create_array(dtype=np.float32)

    ttimes = np.concatenate([-template_origin_times(sta, 0, phaseid, hc.latbins, hc.depthbins)[:,:,:,np.newaxis] for phaseid in hc.phaseids], axis=3)



    valid_len = np.sum([wn.valid_len for wn in sg.station_waves[sta]])
    wn = sg.station_waves[sta][0]

    ats, atvs = zip(*[template_amp_transfer(sg, wn, phase, latbins=hc.latbins, depthbins=hc.depthbins) for phase in hc.phases])
    amp_transfers = np.concatenate([at[:,:,:,np.newaxis] for at in ats], axis=3)
    amp_transfer_stds = np.concatenate([np.sqrt(atv)[:,:,:,np.newaxis] for atv in atvs], axis=3)

    source_logamps = np.array([[brune.source_logamp(mb=mb, band="freq_0.8_4.5", phase=phase) for phase in hc.phases ] for mb in np.linspace(hc.min_mb, hc.max_mb, hc.mbbins+1) ], dtype=np.float)

    tg = sg.template_generator(phase="UA")
    ua_amp_model = tg.unassociated_model(param="coda_height", nm=wn.nm)

    lognoise = float(np.log(wn.nm.c))
    null_ll, full_assoc = generate_sta_hough(array, atimes, amps, ttimes,
                                             amp_transfers, amp_transfer_stds,
                                             hc.stime, hc.time_tick_s, hc.bin_width_deg,
                                             hc.mb_bin_width, hc.min_mb, hc.bin_width_km,
                                             source_logamps, hc.uatemplate_rate, ua_amp_model,
                                             valid_len, lognoise=lognoise, save_assoc=True)
    return array, full_assoc, null_ll


def normalize_global(global_array, global_noev_ll, ev_prior=None, one_event_semantics=False):
    # take an array of p(templates | ev in bin) likelihoods, unnormalized,
    # and convert to p(ev in bin | templates)

    if ev_prior is None:
        ev_prior = 1.0 / global_array.size

    ev_prior_log = np.log(ev_prior)
    global_array += ev_prior_log
    armax = np.max(global_array)
    global_lik = np.exp(global_array-armax)

    if one_event_semantics:
        # treat the array as distribution on location of a single event
        return global_lik/np.sum(global_lik)
    else:
        # treat the array as probability that an event is in each bin,
        # independent of other bins
        noev_prior_log = np.log(1-ev_prior)
        global_noev_lik = np.exp(global_noev_ll-armax + noev_prior_log)
        global_lik /= (global_lik + global_noev_lik)
        return global_lik

def global_hough(sg, hc, uatemplates_by_sta, save_debug=False, save_debug_stas=False):
    global_array = hc.create_array(dtype=np.float32, fill_val=0.0)
    global_assocs = {}
    global_noev_ll = 0
    for sta, uatemplates in uatemplates_by_sta.items():
        sta_array, assocs, null_ll = station_hough(sg, hc, sta, uatemplates)

        global_assocs[sta]=assocs
        global_array += sta_array
        global_noev_ll += null_ll
        sta_array = np.exp(sta_array - np.max(sta_array))

        if save_debug_stas:
            fname = "newhough_%d_%s" % (hc.bin_width_deg, sta)
            visualize_hough_array(sta_array, [sta], fname=fname+".png", ax=None, timeslice=None)
            np.save(fname + ".npy", sta_array)
            print "wrote to", fname

    if save_debug:
        global_lik = normalize_global(global_array, global_noev_ll, one_event_semantics=False)
        fname = "newhough_%d_global.png" % hc.bin_width_deg
        visualize_hough_array(global_lik, uatemplates_by_sta.keys(), fname=fname, ax=None, timeslice=None)
        print "wrote to", fname

        global_dist = normalize_global(global_array, global_noev_ll, one_event_semantics=True)
        fname = "newhough_%d_global_norm.png" % hc.bin_width_deg
        visualize_hough_array(global_dist, uatemplates_by_sta.keys(), fname=fname, ax=None, timeslice=None)
        print "wrote to", fname

    return global_array, global_assocs, global_noev_ll

def main():
    #sfile = "/home/dmoore/python/sigvisa/logs/mcmc/02135/step_000000/pickle.sg"
    sfile = "/home/dmoore/python/sigvisa/logs/mcmc/02138/step_000489/pickle.sg"
    with open(sfile, 'rb') as f:
        sg = pickle.load(f)
    print "read sg"

    hc = HoughConfig(sg.event_start_time, 7200, bin_width_deg=10.0, phases=("P", "S"))

    #stas = ['AS12', 'FITZ', 'WR1']
    #uatemplates_by_sta = get_uatemplates(sg)
    #array,assocs, nll = global_hough(sg, hc, uatemplates_by_sta, save_debug=True)

    iterative_mixture_hough(sg, hc)

if __name__ =="__main__":
    main()
