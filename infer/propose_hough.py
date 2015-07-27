import numpy as np
import sys
import os
import traceback
import itertools
import pickle
import time

from sigvisa.models.ttime import tt_predict
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.source.event import Event, get_event

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

def precompute_amp_transfer(model, latbins=18):
    lonbins = latbins * 2
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    at_means = np.zeros((lonbins, latbins), dtype=float)
    at_vars = np.zeros((lonbins, latbins), dtype=float)

    d = {'lon': 0.0, 'lat': 0.0, 'depth': 0.0, 'mb': 4.0}
    for i in range(lonbins):
        lon = -180.0 + (i+.5) * lonbin_deg
        d['lon'] = lon
        for j in range(latbins):
            lat = -90.0 + (j+.5) * latbin_deg
            d['lat'] = lat
            at_means[i,j] = model.predict(d)
            at_vars[i,j] = model.variance(d)
    return at_means, at_vars


def precompute_travel_times(sta, phaseid=1, latbins=18):
    s = Sigvisa()

    lonbins = latbins * 2
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    times = np.zeros((lonbins, latbins), dtype=float)

    for i in range(lonbins):
        lon = -180.0 + (i+.5) * lonbin_deg
        for j in range(latbins):
            lat = -90.0 + (j+.5) * latbin_deg

            try:
                meantt = s.sigmodel.mean_travel_time(lon, lat, 0.0, 0.0, sta, phaseid - 1)
                if meantt < 0:
                    raise ValueError
                times[i,j] = meantt
            except ValueError:
                times[i,j] = -1e9

    return times


def template_amp_transfer(sg, wn, phase, latbins=18):
    key = (wn.sta, wn.chan, wn.band, phase, latbins)
    if key not in amp_transfer_cache:
        sta, band, chan = wn.sta, wn.band, wn.chan

        modelid = sg.get_param_model_id(runids=sg.runids, sta=sta,
                                        phase=phase, model_type=sg._tm_type("amp_transfer"),
                                        param="amp_transfer", template_shape=sg.template_shape,
                                        chan=chan, band=band)
        model = sg.load_modelid(modelid)
        amp_transfer_cache[key]= precompute_amp_transfer(model, latbins=latbins)
    at_means, at_vars = amp_transfer_cache[key]
    return at_means, at_vars


def template_origin_times(sta, time, phaseid=1, latbins=18):
    """

    Return, for each spatial bin, the inverted origin time
    corresponding to the center of that bin. This is used below (in
    add_template_to_sta_hough) to determine which time bin(s) this
    template should "vote" for at each point in space.

    """

    if (sta, phaseid, latbins) not in travel_time_cache:
        travel_time_cache[(sta, phaseid, latbins)]= precompute_travel_times(sta, phaseid=phaseid, latbins=latbins)
    ttimes = travel_time_cache[(sta, phaseid, latbins)]
    return time - ttimes

def add_template_to_sta_hough_smooth(sta_hough_array, sta, template_snr, stime, atime, time_tick_s=20, vote_size = 1.0, phaseids=(1,5), amp_transfer={}):
    """

    Incorporate a template into the Hough accumulator for a particular
    station, using the 2D array returned by template_origin_times.

    This one spreads the probability mass over as many bins as seems warrented.

    """
    lonbins, latbins, timebins = sta_hough_array.shape
    time_radius = time_tick_s / 2.0
    lonbin_deg = 360.0/lonbins

    # log-"probability" that this is a "genuine" detection.
    # this is taken to be a logistic function of snr
    det_prob = 1.0/(1 + np.exp(-2.0*(template_snr-2.0)))

    # log-"probability" that the travel-time model is correct in
    # assigning this particular time-cell for the detected template


    #plausible_error_s=5.0
    #bin_width_s = 1.41 * lonbin_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S

    stime = float(stime)
    time_tick_s = float(time_tick_s)
    #vote =float(np.exp(lp))

    vote = float(vote_size * det_prob)

    #lp = tt_logprob + det_logprob



    """Forward model: we assume the event is in a given lat/lon/depth
    bin, with a uniform distribution over the precise location. Marginalizing
    over the location gives a distribution on travel times from the bin.
    Simulating this experimentally (sample a uniform location, then sample
    from the travel time model at that location, repeat until satisfied) it seems
    that the marginal TT distribution is reasonably approximated by a Laplacian
    distribution (see iPython notebook "hough_tt_uncertainty").
    The exact spread of this distribution depends on the tt model
    and also on the event-station distance, but we'll just fix it at a constant
    5.0s for current purposes.

    Inverting this, given an arrival time we have a Laplacian
    distribution on origin times for events occurring within a given
    spatial bin. To allocate probability mass to time bins, we
    integrate the Laplacian density over the relevant region of time.
    This means subtracting the CDF at the early boundary of the time
    bin from the CDF at the late boundary.  Laplacian CDFs are easy to
    compute (it's just an exponential), but for efficiency we
    precompute and just use a lookup table.

    """

    laplacian_cdf = """
    // cdf of Laplace(0.0, 5.0) computed for integer x values [-40, 40] inclusive
    double table[] ={0.00017, 0.00020, 0.00025, 0.00031, 0.00037, 0.00046, 0.00056, 0.00068, 0.00083, 0.00101, 0.00124, 0.00151, 0.00185, 0.00226, 0.00276, 0.00337, 0.00411, 0.00503, 0.00614, 0.00750, 0.00916, 0.01119, 0.01366, 0.01669, 0.02038, 0.02489, 0.03041, 0.03714, 0.04536, 0.05540, 0.06767, 0.08265, 0.10095, 0.12330, 0.15060, 0.18394, 0.22466, 0.27441, 0.33516, 0.40937, 0.50000, 0.59063, 0.66484, 0.72559, 0.77534, 0.81606, 0.84940, 0.87670, 0.89905, 0.91735, 0.93233, 0.94460, 0.95464, 0.96286, 0.96959, 0.97511, 0.97962, 0.98331, 0.98634, 0.98881, 0.99084, 0.99250, 0.99386, 0.99497, 0.99589, 0.99663, 0.99724, 0.99774, 0.99815, 0.99849, 0.99876, 0.99899, 0.99917, 0.99932, 0.99944, 0.99954, 0.99963, 0.99969, 0.99975, 0.99980, 0.99983};
    double laplace_cdf(double x) {
        int ix = (int) floor(x);
        int ix2 = ix+1;
        if (ix2 <= -40) return 0.0;
        if (ix >= 40) return 1.0;
        double tbl1 = table[ix+40];
        double tbl2 = table[ix2+40];
        double y = x-ix;
        return (1-y)*tbl1 + y*tbl2;
    }
    """

    template_times = np.dstack([template_origin_times(sta, atime, latbins=latbins, phaseid=phaseid) for phaseid in phaseids])

    #at_means, at_vars = amp_transfer[phaseid]

    nphases = len(phaseids)
    phase_prior = np.array([0.7, 0.3]) # P vs S

    uniform_weight = 1.0/(latbins*lonbins*timebins)
    timebin_weights = np.zeros((timebins,))

    code = """

for (int i=0; i < lonbins; ++i) {
    for (int j=0; j < latbins; ++j) {

        // std::unordered_set<int> bins(10);

        for (int phaseidx=0; phaseidx < nphases; ++phaseidx) {
            double origin_time = template_times(i,j, phaseidx);
            if (origin_time < stime  || origin_time > stime + timebins*time_tick_s) { continue; }

            double min_plausible= origin_time - 30.0;
            double max_plausible= origin_time + 30.0;
            int min_plausible_bin = std::max(0, int((min_plausible-stime) / time_tick_s));
            int max_plausible_bin = std::min(timebins-1, int((max_plausible-stime) / time_tick_s));

            for (int timebin=min_plausible_bin; timebin <= max_plausible_bin; timebin++) {

                double timebin_left = (stime + timebin * time_tick_s) - origin_time;
                double timebin_right = timebin_left + time_tick_s;
                double bin_weight = laplace_cdf(timebin_right) - laplace_cdf(timebin_left);
                timebin_weights(timebin) += bin_weight * phase_prior(phaseidx);
            }
        }

        for (int k=0; k < timebins; ++k) {
            double w = timebin_weights(k);
            double oldval = sta_hough_array(i,j,k);
            sta_hough_array(i,j,k) += log(uniform_weight + vote * w);
            timebin_weights(k) = 0;
        }

    }
}
    """
    weave.inline(code,['stime', 'latbins', 'lonbins', 'timebins', 'time_tick_s', 'template_times', 'timebin_weights', 'phase_prior', 'nphases', 'sta_hough_array', 'vote', 'uniform_weight'], support_code=laplacian_cdf, headers=["<math.h>", "<unordered_set>"], type_converters = converters.blitz,verbose=2,compiler='gcc', extra_compile_args=["-std=c++11"])
    #print "added template"


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

    lonbins, latbins, timebins = hough_array.shape
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    if timeslice is None:
        location_array = np.sum(hough_array, axis=2)
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



def synthetic_hough_array(ev, stas, stime, etime, bin_width_deg):
    """

    For debugging: given an event, generate the "ideal" Hough array
    corresponding to exactly one P arrival from that event at each
    station, at the moment predicted by the travel time model.

    """

    assert(180.0 % bin_width_deg == 0)

    DEG_WIDTH_KM = 111.32
    P_WAVE_VELOCITY_KM_PER_S = 6.0
    time_tick_s = bin_width_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S
    latbins = int(180.0 / bin_width_deg)

    hough_array = init_hough_array(stime=stime, etime=etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=False)
    for sta in stas:
        sta_hough_array = init_hough_array(stime=stime, etime=etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=True)
        pred_atime = ev.time + tt_predict(event=ev, sta=sta, phase='P')
        template_times = template_origin_times(sta, pred_atime, latbins=latbins)
        add_template_to_sta_hough(sta_hough_array, template_times, template_snr=100.0, stime=stime, time_tick_s = time_tick_s)
        hough_array += sta_hough_array
    hough_array = np.exp(hough_array)
    return hough_array


def generate_sta_hough_array(sg, wn, stime, etime, latbins, time_tick_s, prev_array=None, phaseids=(1,5), uaids=None):
    sta_hough_array = init_hough_array(stime=stime, etime=etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=True, prev_array=prev_array)

    amp_transfer= {}
    #phases = {1: "P", 5: "S"}
    #for phaseid in phaseids:
    #    amp_transfer[phaseid] = template_amp_transfer(sg, wn, phases[phaseid], latbins)

    #if debug_ev is not None:
    #    pred_atime = debug_ev.time + tt_predict(event=debug_ev, sta=sta, phase='P')
    #    print "%s pred: %.1f" % (sta, pred_atime)
    for eid, phase in wn.arrivals():
        if phase != "UA": continue
        uaid=-eid

        atime = sg.uatemplates[uaid]['arrival_time'].get_value()
        amp = sg.uatemplates[uaid]['coda_height'].get_value()
        snr = np.exp(amp) / wn.nm.c



        #if debug_ev is not None:
        #    print wn.sta, ":", uaid, ':', np.exp(amp), wn.nm.c, snr, ";", atime


            #template_times = template_origin_times(wn.sta, atime, latbins=latbins, phaseid=phaseid)
            #18
            #77
            #9
            #41
            #38
            #70
            #2
            #44
            #88
            #13
            #32

            # 38
        # 9, 41 generates diffraction
        # whereas 9, 38 is legit potential P/S
        #if uaid not in (9, 41 , ): continue
        if uaids is not None and uaid not in uaids: continue
        #print uaid, stime, atime, snr

        add_template_to_sta_hough_smooth(sta_hough_array, wn.sta, stime=stime, atime=atime, template_snr=snr, time_tick_s = time_tick_s, phaseids=phaseids, amp_transfer=amp_transfer)

    return sta_hough_array

def generate_hough_array(sg, stime, etime, bin_width_deg=1.0, time_tick_s=10.0, exclude_sites=None, force_stas=None):

    """

    Generate a Hough array from a graph containing unassociated templates.

    """

    assert(180.0 % bin_width_deg == 0)

    # We choose the size of the time bins in order to guarantee that,
    # if an event's true origin is located anywhere within a
    # particular location bin, then the inverted time from assuming an
    # origin location at the bin center should be within the same time
    # bin, or at least the adjacent bin, as the true origin time.

    # An event can have distance at most sqrt(2)/2 * bin_width from
    # the center of its bin. If the true origin time is exactly in the
    # center of a time bin, and we want the inverted origin time to be
    # in that same bin, then the width of the time bin should be twice
    # the time required by the wave to travel from the origin location
    # to the spatial bin center (since by the triangle equality this
    # is the largest possible discrepancy in arrival times), i.e. at
    # most sqrt(2) * bin_width.
    if time_tick_s is None:
        time_tick_s = 1.41 * bin_width_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S
    latbins = int(180.0 / bin_width_deg)

    # For each site, loop over its unassociated template and add each
    # to a site-specific Hough accumulator (this computes a max over
    # all templates at each site, approximating the MAP
    # association). Then combine the site-specific accumulators into a
    # global accumulator, and exponentiate to convert log-probs into
    # probs.
    exclude_sites = [] if exclude_sites is None else exclude_sites
    hough_array = init_hough_array(stime=stime, etime=etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=False)
    sta_hough_array=None

    if force_stas is not None:
        stas = force_stas
    else:
        stas = []
        for site in sg.site_elements.keys():
            if site in exclude_sites: continue
            for sta in sg.site_elements[site]:
                stas.append(sta)


    t0 = time.time()
    for sta in stas:
        t1 = time.time()
        for wn in sg.station_waves[sta]:
            sta_hough_array = generate_sta_hough_array(sg, wn, stime, etime,
                                                       latbins=latbins,
                                                       time_tick_s=time_tick_s,
                                                       prev_array=sta_hough_array)
            #print "generated at", sta
            hough_array += sta_hough_array
        t2 = time.time()
            #print "station %s time %f" % (sta, t2-t1)

    hough_array -= np.max(hough_array)
    #t2 = time.time()
    exp_in_place(hough_array)
    #t3 = time.time()

    print "total hough time", t2-t0 #, 'exp time', t3-t2, 'shape', hough_array.shape
    return hough_array


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

def hough_location_proposal(sg, fix_result=None, proposal_dist_seed=None):
    bin_width_deg=2.0
    hough_array = generate_hough_array(sg, stime=sg.event_start_time, etime=sg.end_time,
                                       bin_width_deg=bin_width_deg)
    if fix_result:
        return np.log(event_prob_from_hough(fix_result, hough_array, sg.event_start_time, sg.end_time))
    else:
        proposed_ev, ev_prob = propose_event_from_hough(hough_array, sg.event_start_time, sg.end_time)

        #proposed_ev.lon=135.73
        #proposed_ev.lat=-3.66
        #proposed_ev.depth=0.0
        #proposed_ev.time=1238894038.7
        #ev_prob = 0.1
        return proposed_ev, np.log(ev_prob), hough_array

def main():
    sfile = sys.argv[1]
    with open(sfile, 'rb') as f:
        sg = pickle.load(f)
    print "read sg"


    bin_width_deg=1.0
    time_tick_s = (1.41 * bin_width_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S)
    latbins = int(180.0 / bin_width_deg)


    sta_hough_array = None
    for wns in sg.station_waves.values():
        for wn in wns:
            #if wn.sta not in ("FITZ",): continue
            sta_hough_array = generate_sta_hough_array(sg, wn,
                                                       stime=sg.event_start_time,
                                                       etime=sg.end_time,
                                                       latbins=latbins,
                                                       time_tick_s=time_tick_s,
                                                       prev_array=sta_hough_array)
            fname = "hough_%s.png" % wn.sta
            sta_hough_array -= np.max(sta_hough_array)
            #import pdb; pdb.set_trace()
            sta_hough_array = np.exp(sta_hough_array)
            sta_hough_array /= np.max(sta_hough_array)
            visualize_hough_array(sta_hough_array, [wn.sta,], fname=fname, ax=None, timeslice=None)
            print "saved array to", fname


    hough_array = generate_hough_array(sg, stime=sg.event_start_time, etime=sg.end_time, force_stas=["FITZ", "MK31", "AS12"], bin_width_deg=bin_width_deg)
    fname = "hough_FMA.png"
    visualize_hough_array(hough_array, sg.station_waves.keys(), fname=fname, ax=None, timeslice=None)
    print "saved array to", fname

    hough_array = generate_hough_array(sg, stime=sg.event_start_time, etime=sg.end_time, bin_width_deg=bin_width_deg)
    fname = "hough.png"
    visualize_hough_array(hough_array, sg.station_waves.keys(), fname=fname, ax=None, timeslice=None)
    print "saved array to", fname




    #import pdb; pdb.set_trace()



if __name__ =="__main__":
    main()
