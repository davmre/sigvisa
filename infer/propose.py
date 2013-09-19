import numpy as np
import sys
import os
import traceback
import itertools
import pickle


from sigvisa.models.ttime import tt_predict
from sigvisa.graph.sigvisa_graph import SigvisaGraph, predict_phases
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_segments
from sigvisa.source.event import Event, get_event
from sigvisa.infer.template_mcmc import run_open_world_MH

from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from sigvisa.plotting.plot import savefig

"""

This file contains methods to propose new events using a Hough
transform, given a graph containing unassociated templates. We use a
vaguely Bayesian interpretation of the Hough transform, in which the
number of votes for a particular bin is roughly analagous to a
log-probability that there is an event within that bin.

The main logic is in generate_hough_array, which then calls the other methods.

"""


def init_hough_array(stime, etime, time_tick_s=20, latbins=18, sta_array=False):
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
    init_val = -.7 if sta_array else 0.0
    return np.ones((lonbins, latbins, timebins)) * init_val

def template_origin_times(sta, time, phaseid=1, latbins=18):

    """

    Return, for each spatial bin, the inverted origin time
    corresponding to the center of that bin. This is used below (in
    add_template_to_sta_hough) to determine which time bin(s) this
    template should "vote" for at each point in space.

    """

    s = Sigvisa()

    lonbins = latbins * 2
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    times = np.zeros((lonbins, latbins), dtype=int)

    for i in range(lonbins):
        lon = -180.0 + (i+.5) * lonbin_deg
        for j in range(latbins):
            lat = -90.0 + (j+.5) * latbin_deg

            meantt = s.sigmodel.mean_travel_time(lon, lat, 0.0, time, sta, phaseid - 1)
            origin_time = time - meantt
            times[i,j] = origin_time

    return times

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


def categorical_sample_array(a):
    """

    Sample a bin from the Hough accumulator. Assume that our log
    probabilities have previously been exponentied, and are now just
    probabilities.

    """

    s = np.sum(a)
    cdf = np.cumsum(a)/s

    u = np.random.rand()
    idx = np.searchsorted(cdf, u)

    return np.unravel_index(idx, a.shape)

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

    lonidx, latidx, timeidx = categorical_sample_array(hough_array)
    ev_prob = categorical_prob(hough_array, (lonidx, latidx, timeidx))

    # sample an event location uniformly within each bin
    lonidx += np.random.rand()
    latidx += np.random.rand()
    timeidx += np.random.rand()
    ev_prob /= (lonbin_deg * latbin_deg * time_tick_s)

    lon = -180.0 + lonidx * lonbin_deg
    lat = -90.0 + latidx * latbin_deg
    time = stime + timeidx * time_tick_s

    ev = Event(lon=lon, lat=lat, time=time, depth=0, mb=3.5, natural_source=True)
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


def generate_hough_array(sg, stime, etime, bin_width_deg, exclude_sites=None, debug_ev=None):
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
    DEG_WIDTH_KM = 111.32
    P_WAVE_VELOCITY_KM_PER_S = 6.0
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
    for site in sg.site_elements.keys():
        if site in exclude_sites: continue
        for sta in sg.site_elements[site]:
            for wn in sg.station_waves[sta]:
                chan, band = wn.chan, wn.band
                sta_hough_array = init_hough_array(stime=stime, etime=etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=True)

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
                    add_template_to_sta_hough(sta_hough_array, template_times, stime=stime, template_snr=snr, time_tick_s = time_tick_s)
                hough_array += sta_hough_array
    hough_array = np.exp(hough_array)
    return hough_array

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
    sig_etime = np.max(statimes) + 240

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
    hough_array = generate_hough_array(sg, stime=infer_stime, etime=infer_etime, bin_width_deg=4.0, exclude_sites=exclude_sites, debug_ev=ev)

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
