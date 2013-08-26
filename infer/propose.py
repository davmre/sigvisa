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
from sigvisa.source.event import get_event
from sigvisa.infer.template_mcmc import run_open_world_MH

from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from sigvisa.plotting.plot import savefig

def init_hough_array(stime, etime, time_tick_s=20, latbins=18, sta_array=False):
    lonbins = latbins * 2
    timebins = int((etime-stime)/time_tick_s)

    # since we max over detections at stations, init them to logprobs of log(.5).
    # this corresponds to saying that the probability of a cell goes down by .5 for every station at which we don't find a corresponding detection.
    # since we sum over stations, init the global array to a logprob of 0.0
    init_val = -.7 if sta_array else 0.0

    return np.ones((lonbins, latbins, timebins)) * init_val

def template_origin_times(sta, time, phaseid=1, latbins=18, offset=False):

    """

    Input: a station, phaseid, and arrival time for that phaseid. Also, a starting and ending time

    """

    s = Sigvisa()

    lonbins = latbins * 2
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    center_offset_deg = .5 * lonbin_deg if not offset else 0.0

    times = np.zeros((lonbins, latbins), dtype=int)

    for i in range(lonbins):
        lon = -180.0 + i * lonbin_deg + center_offset_deg
        for j in range(latbins):
            lat = -90.0 + j * latbin_deg + center_offset_deg

            meantt = s.sigmodel.mean_travel_time(lon, lat, 0.0, time, sta, phaseid - 1)
            origin_time = time - meantt
            times[i,j] = origin_time

    return times

def add_template_to_sta_hough(sta_hough_array, template_timebins, template_snr, stime, time_tick_s=20, tt_sharpness = 5.0, offset=False):
    lonbins, latbins, timebins = sta_hough_array.shape
    time_radius = time_tick_s / 2.0

    # log-"probability" that this is a genuine detection
    # taken as a logistic function of snr
    det_logprob = - np.log(1 + np.exp(-2.0*(template_snr-2.0)))

    # log-"probability" that the travel-time model is correct in
    # assigning this particular time-cell for the detected template
    tt_logprob = tt_sharpness

    lp = tt_logprob + det_logprob

    for i in range(lonbins):
        for j in range(latbins):
            origin_time = template_timebins[i,j]
            timebin = int(np.floor((origin_time - stime)/time_tick_s))
            time_offset = (origin_time-stime) % time_tick_s

            # this is 0 if exactly in the center of the time bin, and .5 if exactly on the border
            bleeding =  np.abs(float(time_radius)-time_offset) / float(time_tick_s)

            lp1 = lp + np.log(1-bleeding)
            lp2 = lp + np.log(bleeding)

            if 0 <= timebin < timebins:
                sta_hough_array[i, j, timebin] = max(lp1, sta_hough_array[i, j, timebin])

            bin2 = timebin-1 if time_offset*2 < time_tick_s else timebin +1
            if 0 <= bin2 < timebins:
                sta_hough_array[i, j, bin2] = max(lp2, sta_hough_array[i, j, bin2])

def categorical_sample_array(a):
    s = np.sum(a)
    cdf = np.cumsum(a)/s

    u = np.random.rand()
    idx = np.searchsorted(cdf, u)

    return np.unravel_index(idx, a.shape)

def categorical_prob(a, idx):
    s = np.sum(a)
    return a[idx]/s

def propose_event_from_hough(hough_array, stime, etime):
    lonbins, latbins, timebins = hough_array.shape
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins
    timebin_s = float(etime-stime)/timebins

    lonidx, latidx, timeidx = categorical_sample_array(hough_array)
    ev_prob = categorical_prob(hough_array, (lonidx, latidx, timeidx))

    # sample an event location uniformly within each bin
    lonidx += np.random.rand()
    latidx += np.random.rand()
    timeidx += np.random.rand()
    ev_prob /= (lonbin_deg * latbin_deg * timebin_s)

    lon = -180.0 + lonidx * lonbin_deg
    lat = -90.0 + latidx * latbin_deg
    time = stime + timeidx * timebin_s

    ev = Event(lon=lon, lat=lat, time=time, depth=0, mb=3.5, natural_source=True)
    return ev, ev_prob

def merge_offset_array(hough_array_exp, hough_array_offset_exp):
    lonbins, latbins, timebins = hough_array_exp.shape

    merged_array = np.ones((lonbins*2, latbins*2, timebins))
    for i in range(lonbins):
        ip = (i+1) % lonbins
        for j in range(latbins):
            jp = (j+1) % latbins
            for k in range(timebins):
                merged_array[i*2, j*2, k] = max(hough_array_exp[i,j,k], hough_array_offset_exp[i,j,k])
                merged_array[i*2+1, j*2, k] = max(hough_array_exp[i,j,k], hough_array_offset_exp[ip,j,k])
                merged_array[i*2, j*2+1, k] = max(hough_array_exp[i,j,k], hough_array_offset_exp[i,jp,k])
                merged_array[i*2+1, j*2+1, k] = max(hough_array_exp[i,j,k], hough_array_offset_exp[ip,jp,k])
    return merged_array

def visualize_hough_array(hough_array, sites, fname, timeslice=None, offset=False):
    lonbins, latbins, timebins = hough_array.shape
    latbin_deg = 180.0/latbins
    lonbin_deg = 360.0/lonbins

    if timeslice is None:
        location_array = np.sum(hough_array, axis=2)
    else:
        location_array = hough_array[:,:,timeslice]

    fig = Figure(figsize=(8, 5), dpi=300)
    axes = fig.add_subplot(111)
    bmap = Basemap(resolution="c", projection = "robin", lon_0 = 0, ax=axes)
    bmap.drawcoastlines(zorder=10)
    bmap.drawmapboundary()

    parallels = [int(k) for k in np.linspace(-90, 90, 10)]
    bmap.drawparallels(parallels, labels=[False, True, True, False], fontsize=8, zorder=4)
    meridians = [int(k) for k in np.linspace(-180, 180, 10)]
    bmap.drawmeridians(meridians, labels=[True, False, False, True], fontsize=5, zorder=4)

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
        lon = max(-180 + (i - .5*offset) * lonbin_deg, -180)
        for j in range(latbins):
            lat = max(-90 + (j- .5*offset) * latbin_deg, -90)

            draw_cell(left_lon = lon, bottom_lat = lat, alpha = location_array[i,j]/m)

    s = Sigvisa()
    for site in sites:
        sta_location = s.earthmodel.site_info(site, 0)[0:2]
        x1, x2 = bmap(sta_location[0], sta_location[1])
        bmap.plot([x1], [x2], marker="x", ms=4, mfc="none", mec="blue", mew=1, alpha=1, zorder=10)
        x_off = 3
        y_off = 3
        axes.annotate(
            site,
            xy=(x1, x2),
            xytext=(x_off, y_off),
            textcoords='offset points',
            size=6,
            color = 'blue',
            zorder=10)


    savefig(fname, fig, dpi=300)

def main():

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    ev = get_event(evid=5393637)


    sites = ['AKASG', 'YKA', 'JNU']
    statimes = [ev.time + tt_predict(event=ev, sta=sta, phase=phase) for (sta, phase) in itertools.product(sites, ['P',])]
    sig_stime = np.min(statimes) - 60
    sig_etime = np.max(statimes) + 240

    infer_stime = ev.time - 653
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
    f = open('cached_templates.sg', 'wb')
    pickle.dump(sg, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    """

    f = open('cached_templates.sg', 'rb')
    sg = pickle.load(f)

    latbins = 36

    latbin_width = 180.0/latbins
    DEG_WIDTH_KM = 111.32
    P_WAVE_VELOCITY_KM_PER_S = 6.0
    time_tick_s = latbin_width * DEG_WIDTH_KM / P_WAVE_VELOCITY_KM_PER_S

    hough_array = init_hough_array(stime=infer_stime, etime=infer_etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=False)
    hough_array_offset = init_hough_array(stime=infer_stime, etime=infer_etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=False)
    for site in sites:
        for sta in list(sg.site_elements[site]) + []:
            sta_hough_array = init_hough_array(stime=infer_stime, etime=infer_etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=True)
            sta_hough_array_offset = init_hough_array(stime=infer_stime, etime=infer_etime, latbins=latbins, time_tick_s = time_tick_s, sta_array=True)

            #for wn in sg.station_waves[sta]:
            #sta, chan, band = wn.sta, wn.chan, wn.band
            band = "freq_2.0_3.0"
            chan = "BHZ"
            good_atime = ev.time + tt_predict(event=ev, sta=sta, phase='P')

            timebins = template_origin_times(sta, good_atime, latbins=latbins, offset=False)
            timebins_offset = template_origin_times(sta, good_atime, latbins=latbins, offset=True)
            add_template_to_sta_hough(sta_hough_array, timebins, template_snr=3.0, stime=infer_stime, time_tick_s = time_tick_s, offset=False)
            add_template_to_sta_hough(sta_hough_array_offset, timebins_offset, template_snr=3.0, stime=infer_stime, time_tick_s = time_tick_s, offset=True)
            """continue
            for uaid in sg.uatemplate_ids[(sta,chan,band)]:
                atime = sg.uatemplates[uaid]['arrival_time'].get_value()
                #if np.abs(atime - good_atime) > 10: continue
                amp = sg.uatemplates[uaid]['coda_height'].get_value()
                snr = np.exp(amp) / wn.nm.c
                print wn.sta, ":", uaid, ':', np.exp(amp), wn.nm.c, snr
                timebins = template_origin_times(sta, atime, latbins=latbins)
                add_template_to_sta_hough(sta_hough_array, timebins, stime=infer_stime, template_snr=snr, time_tick_s = time_tick_s)"""
            hough_array += sta_hough_array
            hough_array_offset += sta_hough_array_offset
            #visualize_hough_array(np.exp(sta_hough_array), [sta,], fname="hough_%s.png" % sta)
    hough_array = np.exp(hough_array)
    hough_array_offset = np.exp(hough_array_offset)
    merged_array = merge_offset_array(hough_array, hough_array_offset)
    true_bin = int((ev.time - infer_stime)/time_tick_s)

    visualize_hough_array(merged_array, sites, fname="hough.png")
    visualize_hough_array(hough_array, sites, fname="hough1.png", offset=False)
    visualize_hough_array(hough_array_offset, sites, fname="hough2.png", offset=True)


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
