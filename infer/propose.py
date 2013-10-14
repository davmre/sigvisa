import numpy as np
import sys
import os
import traceback
import itertools
import pickle
import time

from sigvisa.models.ttime import tt_predict
from sigvisa.graph.sigvisa_graph import SigvisaGraph, predict_phases
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
    init_val = np.exp(-.7) if sta_array else 1.0

    if prev_array is None:
        hough_array = np.empty((lonbins, latbins, timebins))
    else:
        hough_array=prev_array
    hough_array.fill(init_val)

    return hough_array

def precompute_travel_times(sta, phaseid=1, latbins=18):
    s = Sigvisa()

    lonbins = latbins * 2
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    times = np.zeros((lonbins, latbins), dtype=int)

    for i in range(lonbins):
        lon = -180.0 + (i+.5) * lonbin_deg
        for j in range(latbins):
            lat = -90.0 + (j+.5) * latbin_deg

            meantt = s.sigmodel.mean_travel_time(lon, lat, 0.0, 0.0, sta, phaseid - 1)
            times[i,j] = meantt

    return times


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

def add_template_to_sta_hough(sta_hough_array, template_times, template_snr, stime, time_tick_s=20, tt_sharpness = 5.0):
    """

    Incorporate a template into the Hough accumulator for a particular
    station, using the 2D array returned by template_origin_times.

    """

    lonbins, latbins, timebins = sta_hough_array.shape
    time_radius = time_tick_s / 2.0

    # log-"probability" that this is a "genuine" detection.
    # this is taken to be a logistic function of snr
    det_logprob = - np.log(1 + np.exp(-2.0*(template_snr-4.0)))

    # log-"probability" that the travel-time model is correct in
    # assigning this particular time-cell for the detected template
    tt_logprob = tt_sharpness

    lp = tt_logprob + det_logprob

    for i in range(lonbins):
        for j in range(latbins):
            origin_time = template_times[i,j]
            time_offset = (origin_time-stime) % time_tick_s
            timebin = int(np.floor((origin_time-stime) / time_tick_s))

            # if the inverted time is near the edge of a time bin, we spread the vote over both bins.

            # this quantity is 0 if exactly in the center of the time bin, and .5 if exactly on the border
            bleeding =  np.abs(float(time_radius)-time_offset) / float(time_tick_s)
            # we then add a nonlinearity in order to restrict the
            # "bleeding" effect to only the borders of the bin
            bleeding = (bleeding * 2.0) ** 3 / 2.0
            # finally, we split the probability mass according to the proportion we computed
            lp1 = lp + np.log(1-bleeding)
            lp2 = lp + np.log(bleeding)

            if 0 <= timebin < timebins:
                sta_hough_array[i, j, timebin] = max(lp1, sta_hough_array[i, j, timebin])

            bin2 = timebin-1 if time_offset*2 < time_tick_s else timebin +1
            if 0 <= bin2 < timebins:
                sta_hough_array[i, j, bin2] = max(lp2, sta_hough_array[i, j, bin2])


def add_template_to_sta_hough_smooth(sta_hough_array, template_times, template_snr, stime, time_tick_s=20, tt_sharpness = 5.0):
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
    det_logprob = - np.log(1 + np.exp(-2.0*(template_snr-4.0)))

    # log-"probability" that the travel-time model is correct in
    # assigning this particular time-cell for the detected template
    tt_logprob = tt_sharpness

    lp = tt_logprob + det_logprob
    plausible_error_s=5.0
    bin_width_s = 1.41 * lonbin_deg * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S

    stime = float(stime)
    time_tick_s = float(time_tick_s)
    lp =float(np.exp(lp))



    code = """

for (int i=0; i < lonbins; ++i) {
    for (int j=0; j < latbins; ++j) {
        double origin_time = template_times(i,j);
        double min_plausible= origin_time - bin_width_s-plausible_error_s;
        int min_plausible_bin = std::max(0, int((min_plausible-stime) / time_tick_s));
        double max_plausible = origin_time + bin_width_s + plausible_error_s;
        int max_plausible_bin = std::min(timebins-1, int((max_plausible-stime) / time_tick_s));

        for (int timebin=min_plausible_bin; timebin <= max_plausible_bin; timebin++) {
            double oldval = sta_hough_array(i, j, timebin);
            sta_hough_array(i,j,timebin) = std::max(lp, oldval);
        }
    }
}
    """
    weave.inline(code,['stime', 'latbins', 'lonbins', 'timebins', 'bin_width_s', 'plausible_error_s', 'time_tick_s', 'template_times', 'sta_hough_array', 'lp'],type_converters = converters.blitz,verbose=2,compiler='gcc')

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

    ev = Event(lon=lon, lat=lat, time=t, depth=0, mb=3.5, natural_source=True)
    return ev, ev_prob

def visualize_hough_array(hough_array, sites, fname, timeslice=None):
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
    fig = Figure(figsize=(8, 5), dpi=300)
    axes = fig.add_subplot(111)
    bmap = Basemap(resolution="c", projection = "robin", lon_0 = 0, ax=axes)
    bmap.drawcoastlines(zorder=10)
    bmap.drawmapboundary()
    parallels = [int(k) for k in np.linspace(-90, 90, 10)]
    bmap.drawparallels(parallels, labels=[False, True, True, False], fontsize=8, zorder=4)
    meridians = [int(k) for k in np.linspace(-180, 180, 10)]
    bmap.drawmeridians(meridians, labels=[True, False, False, True], fontsize=5, zorder=4)

    # shade the bins according to their probabilities
    def draw_cell( left_lon, bottom_lat, alpha=0.8, facecolor="red"):
        right_lon = min(left_lon + lonbin_deg, 180)
        top_lat = min(bottom_lat+latbin_deg, 90)
        lons = [left_lon, left_lon, right_lon, right_lon]
        lats = [bottom_lat, top_lat, top_lat, bottom_lat]
        x, y = bmap( lons, lats )
        xy = zip(x,y)
        poly = Polygon( xy, facecolor=facecolor, alpha=alpha )
        axes.add_patch(poly)
    m = np.max(location_array)
    for i in range(lonbins):
        lon = max(-180 + i * lonbin_deg, -180)
        for j in range(latbins):
            lat = max(-90 + j * latbin_deg, -90)
            draw_cell(left_lon = lon, bottom_lat = lat, alpha = location_array[i,j]/m)

    # draw the stations
    s = Sigvisa()
    for site in sites:
        sta_location = s.earthmodel.site_info(site, 0)[0:2]
        x1, x2 = bmap(sta_location[0], sta_location[1])
        bmap.plot([x1], [x2], marker="x", ms=4, mfc="none", mec="blue", mew=1, alpha=1, zorder=10)
        x_off = 3
        y_off = 3
        axes.annotate(site,xy=(x1, x2),xytext=(x_off, y_off),textcoords='offset points',
                      size=6,color = 'blue',zorder=10)


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


def generate_hough_array(sg, stime, etime, bin_width_deg, time_tick_s=None, exclude_sites=None, smoothbins=False, debug_ev=None):
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

    t0 = time.time()
    for site in sg.site_elements.keys():
        if site in exclude_sites: continue
        for sta in sg.site_elements[site]:
            t1 = time.time()
            for wn in sg.station_waves[sta]:
                chan, band = wn.chan, wn.band

                sta_hough_array = init_hough_array(stime=stime, etime=etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=True, prev_array=sta_hough_array)

                if debug_ev is not None:
                    pred_atime = debug_ev.time + tt_predict(event=debug_ev, sta=sta, phase='P')
                    print "%s pred: %.1f" % (sta, pred_atime)
                for uaid in sg.uatemplate_ids[(sta,chan,band)]:
                    atime = sg.uatemplates[uaid]['arrival_time'].get_value()
                    amp = sg.uatemplates[uaid]['coda_height'].get_value()
                    snr = np.exp(amp) / wn.nm.c
                    if debug_ev is not None:
                        print wn.sta, ":", uaid, ':', np.exp(amp), wn.nm.c, snr, ";", atime
                    template_times = template_origin_times(sta, atime, latbins=latbins)
                    if smoothbins:
                        add_template_to_sta_hough_smooth(sta_hough_array, template_times, stime=stime, template_snr=snr, time_tick_s = time_tick_s)
                    else:
                        add_template_to_sta_hough(sta_hough_array, template_times, stime=stime, template_snr=snr, time_tick_s = time_tick_s)

                hough_array *= sta_hough_array
            t2 = time.time()
            print "station %s time %f" % (sta, t2-t1)

    #t2 = time.time()
    #exp_in_place(hough_array)
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

def main():
    """

    Test/debug Hough proposals using the 2009 DPRK event.

    """

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    ev = get_event(evid=5393637)


    sites = ['AKASG', 'YKA', 'JNU', 'ILAR', 'WRA', 'FINES', 'ASAR', 'NVAR', 'STKA']
    statimes = [ev.time + tt_predict(event=ev, sta=sta, phase=phase) for (sta, phase) in itertools.product(sites, ['P',])]
    sig_stime = np.min(statimes) - 60
    sig_etime = np.max(statimes) + 640

    infer_stime = ev.time - 700
    infer_etime = sig_etime

    """
    sg = SigvisaGraph(template_shape = "paired_exp", template_model_type = "gp_lld",
                      wiggle_family = "dummy", wiggle_model_type = "dummy",
                      dummy_fallback = False, nm_type = "ar",
                      runid=14, phases='P', gpmodel_build_trees=False)


    segments = load_segments(cursor, sites, sig_stime, sig_etime, chans = ['BHZ', 'SHZ'])
    segments = [seg.with_filter('env;hz_%.3f' % 5.0) for seg in segments]

    wave_nodes = []
    for seg in segments:
        for band in ['freq_2.0_3.0']:
            filtered_seg = seg.with_filter(band)
            for chan in filtered_seg.get_chans():
                wave = filtered_seg[chan]
                wave_nodes.append(sg.add_wave(wave))

    run_open_world_MH(sg, wave_nodes, wiggles=False, steps=125)
    f = open('cached_templates2.sg', 'wb')
    pickle.dump(sg, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    sg.debug_dump("templates")
    """

    f = open('cached_templates2.sg', 'rb')
    sg = pickle.load(f)
    f.close()

    #"""

    exclude_sites = ['STKA']
    #exclude_sites = []
    hough_array = generate_hough_array(sg, stime=infer_stime, etime=infer_etime, bin_width_deg=1.0, exclude_sites=exclude_sites, time_tick_s=10, debug_ev=ev, smoothbins=True)

    #stas = ['AKBB', 'FITZ', 'JNU']
    #hough_array = synthetic_hough_array(ev, stas=stas, stime=infer_stime, etime=infer_etime, bin_width_deg=5)

    ev, ev_prob = propose_event_from_hough(hough_array, infer_stime, infer_etime)
    print ev
    print ev_prob
    visualize_hough_array(hough_array, sites, fname="hough.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
