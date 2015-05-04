import numpy as np
import numpy.ma as ma
import scipy.stats
import os
import re
import sys
import pdb
import traceback
import pickle

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.gridspec as gridspec

from sigvisa.plotting.heatmap import find_center
from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa.plotting.histogram import plot_density, plot_histogram
from sigvisa.plotting.plot import basic_plot_to_file, subplot_waveform, savefig, plot_pred_atimes, plot_det_times
import sigvisa.utils.geog as geog
from sigvisa.utils.fileutils import mkdir_p
from sigvisa.models.ttime import tt_predict

EVTRACE_LON, EVTRACE_LAT, EVTRACE_DEPTH, EVTRACE_TIME, EVTRACE_MB, EVTRACE_SOURCE = range(6)

def match_true_ev(trace, true_evs):
    mean_lon, mean_lat = find_center(trace[:, 0:2])
    mean_time = np.mean(trace[:,3])

    true_ev = None
    best_distance = np.float('inf')
    true_evs = true_evs if true_evs is not None else []
    for ev in true_evs:
        dist = geog.dist_km((mean_lon, mean_lat), (ev.lon, ev.lat))
        dist += np.abs(ev.time - mean_time) * 5 # equate one second of error with 5km
        if dist < best_distance:
            best_distance = dist
            true_ev = ev
    return true_ev

def trace_stats(trace, true_evs):
    mean_lon, mean_lat = find_center(trace[:, 0:2])

    mean_time = np.mean(trace[:,3])

    # hack: if location is near international date line, rotate
    # coordinates so we get a reasonable stddev calculation.  (note
    # this doesn't affect the mean calculation above since that's
    # already smart about spherical coordinates)
    if mean_lon > 170:
        wrapped = trace[:, 0] < 0
        trace[wrapped, 0] += 360
    elif mean_lon < -170:
        wrapped = trace[:, 0] > 0
        trace[wrapped, 0] -= 360

    lon_std =  np.std(trace[:, 0])
    lat_std =  np.std(trace[:, 1])
    lon_std_km = geog.dist_km((mean_lon, mean_lat), (mean_lon+lon_std, mean_lat))
    lat_std_km = geog.dist_km((mean_lon, mean_lat), (mean_lon, mean_lat+lat_std))
    txt = "mean: %.2f, %.2f\nstd: %.2f, %.2f\nstd_km: %.1f, %.1f" % (mean_lon, mean_lat, lon_std, lat_std, lon_std_km, lat_std_km)

    results = dict()
    results['mean_lon'] = mean_lon
    results['mean_lat'] = mean_lat
    results['lon_std'] = lon_std
    results['lat_std'] = lat_std
    results['lon_std_km'] = lon_std_km
    results['lat_std_km'] = lat_std_km

    true_ev = match_true_ev(trace, true_evs)
    if true_ev is not None:
        txt += '\ntrue: %.2f, %.2f\n' % (true_ev.lon, true_ev.lat)
        dist_mean = geog.dist_km((mean_lon, mean_lat), (true_ev.lon, true_ev.lat))
        dist_proposal = geog.dist_km((trace[0, 0], trace[0, 1]), (true_ev.lon, true_ev.lat))
        txt += 'd(mean, true) = %.2f\n' % dist_mean
        txt += 'd(proposal, true) = %.2f' % dist_proposal
        results['dist_mean'] = dist_mean
        results['dist_proposal'] = dist_proposal

    return results, txt

def ev_lonlat_density(trace, ax, true_evs=None, frame=None, bounds=None, text=True, ms=8):

    lonlats = trace[:, 0:2]
    n = lonlats.shape[0]

    if bounds is None:
        bounds = {'autobounds': lonlats, 'autobounds_quantile': .9995}
    hm = EventHeatmap(f=None, calc=False, **bounds)
    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)

    alpha_fade_time = 500
    if frame is not None:
        baseline_alpha = 0.05
        alpha = np.ones((frame,)) * baseline_alpha
        t = min(frame,alpha_fade_time)
        alpha[-t:] = np.linspace(baseline_alpha, 0.2, alpha_fade_time)[-t:]
        lonlats = lonlats[:frame, :]
    else:
        alpha = 1.0 / np.log(n+1)

    #hm.plot_locations(X, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
    scplot = hm.plot_locations(lonlats, marker=".", ms=ms, mfc="red", mew=0, mec="none", alpha=alpha)



    results, txt = trace_stats(trace, true_evs)

    if text:
        ax.text(0, 1, txt, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='purple')


    if true_ev is not None:
        hm.plot_locations(np.array(((true_ev.lon, true_ev.lat),)), marker="x", ms=5, mfc="blue", mec="blue", mew=3, alpha=1.0)

    return true_ev, txt, results

def load_trace(logfile, burnin):
    try:
        trace = np.loadtxt(logfile)
    except:
        print "fixing file", logfile
        with open(logfile, 'r') as f:
            lines = f.readlines()
        with open(logfile, 'w') as f:
            f.write(''.join(lines[:-1]))
        trace = np.loadtxt(logfile)

    if len(trace.shape) == 1:
        trace = np.reshape(trace, (1, -1))
    trace = np.array([row for row in trace if row[0] > burnin])
    if len(trace) == 0:
        return trace, 0, 0
    min_step = np.min(trace[:, 0])
    max_step = np.max(trace[:, 0])

    trace = np.array(trace[:, 1:])
    return trace, min_step, max_step


def analyze_event(run_dir, eid, burnin, true_evs=None, bigimage=True, frameskip=50000):
    ev_trace_file = os.path.join(run_dir, 'ev_%05d.txt' % eid)
    ev_img_file = os.path.join(run_dir, 'ev_%05d.png' % eid)
    ev_txt = os.path.join(run_dir, 'ev_%05d_results.txt' % eid)
    ev_dir = os.path.join(run_dir, 'ev_%05d' % eid)
    mkdir_p(ev_dir)

    true_ev = None

    trace, min_step, max_step = load_trace(ev_trace_file, burnin=burnin)
    if len(trace) == 0:
        return
    print "loaded"

    if bigimage:
        # save big event image
        f = Figure((11,8))
        gs = gridspec.GridSpec(2, 4)
        lonlat_ax = f.add_subplot(gs[0:2, 0:2])
        true_ev, txt, results = ev_lonlat_density(trace, lonlat_ax, true_evs=true_evs)
        print "plotted density"
        with open(ev_txt, 'w') as fi:
            fi.write(txt + '\n')

        # also plot: depth, time, mb histograms
        depth_ax = f.add_subplot(gs[0, 2])
        plot_density(trace[:, EVTRACE_DEPTH], depth_ax, "depth", true_value = true_ev.depth if true_ev is not None else None)
        time_ax = f.add_subplot(gs[1, 2])
        plot_density(trace[:, EVTRACE_TIME], time_ax , "time", true_value = true_ev.time if true_ev is not None else None)
        mb_ax = f.add_subplot(gs[0, 3])
        plot_density(trace[:, EVTRACE_MB], mb_ax, "mb", true_value = true_ev.mb if true_ev is not None else None)
        print "plotted others"
        f.suptitle('eid %d (%d samples)' % (eid,trace.shape[0]))
        f.tight_layout()
        canvas = FigureCanvasAgg(f)
        canvas.draw()
        f.savefig(ev_img_file, bbox_inches="tight", dpi=300)

    """
    # also save individual images
    frame_dir = os.path.join(ev_dir, 'frames')
    mkdir_p(frame_dir)
    for (i,frame) in enumerate(np.arange(frameskip, max_step-min_step, frameskip)):
        frame_fname = os.path.join(frame_dir, '%06d.png' % i)
        if os.path.exists(frame_fname): continue
        f = Figure((8,8))
        ax = f.add_subplot(1,1,1)
        true_ev, txt = ev_lonlat_density(trace, ax, true_evs=true_evs, text=False, frame=frame)
        savefig(frame_fname, f, bbox_inches="tight", dpi=75)
    video_cmd = "ffmpeg -qscale 5 -f image2 -r 5 -i %s/%%06d.png %s/chain.mp4" % (frame_dir, ev_dir)
    os.system(video_cmd)


    if true_ev is not None:
        f = Figure((8,8))
        ax = f.add_subplot(1,1,1)
        bounds = {'center': (true_ev.lon, true_ev.lat), 'width_deg': 5.0, 'height_deg': 5.0}
        true_ev, txt = ev_lonlat_density(trace, ax, true_evs=true_evs, text=False, bounds=bounds, ms=4)
        savefig(os.path.join(ev_dir, 'posterior_loc_big.png'), f, bbox_inches="tight", dpi=300)

        f = Figure((8,8))
        ax = f.add_subplot(1,1,1)
        bounds = {'center': (true_ev.lon, true_ev.lat), 'width_deg': 30.0, 'height_deg': 30.0}
        true_ev, txt = ev_lonlat_density(trace, ax, true_evs=true_evs, text=False, bounds=bounds, ms=2)
        savefig(os.path.join(ev_dir, 'posterior_loc_reallybig.png'), f, bbox_inches="tight", dpi=300)
    """

    f = Figure((8,8))
    ax = f.add_subplot(1,1,1)
    plot_histogram(trace[:, EVTRACE_MB], ax, "mb", draw_stats=False, true_value=true_ev.mb, trueval_label="LEB")
    savefig(os.path.join(ev_dir, 'posterior_mb.png'), f, bbox_inches="tight", dpi=300)

    f = Figure((8,8))
    ax = f.add_subplot(1,1,1)
    plot_histogram(trace[:, EVTRACE_TIME], ax, "time", draw_stats=False, true_value=true_ev.time, trueval_label="LEB")
    savefig(os.path.join(ev_dir, 'posterior_time.png'), f, bbox_inches="tight", dpi=300)

    f = Figure((8,8))
    ax = f.add_subplot(1,1,1)
    plot_histogram(trace[:, EVTRACE_DEPTH], ax, "depth", draw_stats=False, true_value=true_ev.depth, trueval_label="LEB")
    savefig(os.path.join(ev_dir, 'posterior_depth.png'), f, bbox_inches="tight", dpi=300)

    return results

def plot_arrival_template_posterior(ev_dir, sg, eid, wn_lbl, phase, burnin, alpha=None, tmpl_color='green', ax=None, plot_predictions=True, plot_dets=False):

    lbl = 'tmpl_%d_%s_%s' % (eid, wn_lbl, phase)
    tmpl_trace_file = os.path.join(ev_dir, '%s' % lbl)
    if ax is None:
        tmpl_img_file = os.path.join(ev_dir, '%s.png' % lbl)
    else:
        tmpl_img_file = None

    wn = sg.all_nodes[wn_lbl]
    tmnodes = sg.get_template_nodes(eid, wn.sta, phase, wn.band, wn.chan)

    real_wave = wn.get_wave()
    real_wave.data = ma.masked_array(real_wave.data, copy=True)
    wn.unfix_value()

    trace, min_step, max_step = load_trace(tmpl_trace_file, burnin=burnin)
    if len(trace) == 0:
        return

    atimes = trace[:, 0]
    min_atime = np.min(atimes)
    plot_stime = min_atime - 30.0

    max_atime = np.max(atimes)
    plot_etime = max_atime + 100.0

    if ax is None:
        f = Figure((10, 5))
        ax = f.add_subplot(1,1,1)

    subplot_waveform(real_wave, ax, stime=plot_stime, etime=plot_etime, plot_dets=False, color='black', linewidth=0.5)

    if plot_predictions:
        predictions = []
        for (eid, phase) in wn.arrivals():
            if eid < 0: continue
            event = sg.get_event(eid)
            predictions.append([phase+"_%d" % eid, event.time+tt_predict(event, wn.sta, phase)])
        plot_pred_atimes(dict(predictions), real_wave, axes=ax, stime=plot_stime, etime=plot_etime, color="purple")
    if plot_dets:
        plot_det_times(real_wave, axes=ax, stime=plot_stime, etime=plot_etime, color="red", all_arrivals=True)

    print plot_stime, plot_etime, wn.st, wn.et

    n = trace.shape[0]
    max_idxs = 100
    if n > max_idxs:
        idxs = np.array(np.linspace(0, n-1, max_idxs), dtype=int)
        n = max_idxs
    else:
        idxs = np.arange(n)

    alpha = 0.4/np.sqrt(n+5) if alpha is None else alpha

    for row in trace[idxs,:]:
        v = {'arrival_time': row[0], 'peak_offset': row[1], 'coda_height': row[2], 'coda_decay': row[3]}
        sg.set_template(eid, wn.sta, phase, wn.band, wn.chan, v)
        tmpl_stime = v['arrival_time']
        tmpl_len = max(10.0, np.exp(v['peak_offset']) - (np.log(0.02 * wn.nm.c) - v['coda_height'])/np.exp(v['coda_decay']))
        tmpl_etime = min(tmpl_stime + tmpl_len, plot_etime)

        wn.parent_predict()
        subplot_waveform(wn.get_wave(), ax, stime=tmpl_stime, etime=tmpl_etime, plot_dets=False, color=tmpl_color, linewidth=0.5, alpha=alpha, fill_y2=wn.nm.c)

    if tmpl_img_file is not None:
        canvas = FigureCanvasAgg(f)
        canvas.draw()
        f.savefig(tmpl_img_file, bbox_inches="tight", dpi=300)

    wn.set_value(real_wave.data)
    wn.fix_value()


def plot_ev_template_posteriors(run_dir, sg, eid, burnin):
    ev_dir = os.path.join(run_dir, 'ev_%05d' % eid)
    mkdir_p(ev_dir)
    r = re.compile('tmpl_%d_' % eid + r'(.+)_([A-Za-z])+')
    for fname in os.listdir(ev_dir):
        print fname
        m = r.match(fname)
        if m is not None:
            wn_lbl = m.group(1)
            phase = m.group(2)
            try:
                plot_arrival_template_posterior(ev_dir, sg, eid, wn_lbl, phase, burnin=burnin)
            except ValueError as e:
                print e
                continue
        else:
            print fname, "match failed"


def combine_steps(run_dir):
    steps = sorted([fname for fname in os.listdir(run_dir) if fname.startswith('step_') and not fname.endswith('tgz')])
    imgnames = [iname for iname in os.listdir(os.path.join(run_dir, steps[0])) if iname.endswith('png')]

    for imgname in imgnames:
        sta = imgname.split('_')[1]
        command = 'convert '
        for step in steps:
            command += '%s ' % os.path.join(run_dir, step, imgname)
        command += '%s' % os.path.join(run_dir, 'signals_%s.pdf' % sta)
        print command
        os.system(command)

def plot_lp_trend(run_dir):
    lps = np.loadtxt(os.path.join(run_dir, 'lp.txt'))
    basic_plot_to_file(fname=os.path.join(run_dir, 'lp.png'), data=lps, title='overall log probability')

def summarize_times(run_dir):
    r = re.compile(r'^move_(.+)_times.txt$')
    move_fnames = [fname for fname in os.listdir(os.path.join(run_dir)) if r.match(fname) is not None]

    f = open(os.path.join(run_dir, 'move_times_summary.txt'), 'w')

    max_step = 0
    total_time = 0

    for fname in move_fnames:
        print "summarizing move", fname
        move_name = r.match(fname).group(1)

        with open(os.path.join(run_dir, fname), 'r') as move_f:
            steps = []
            times = []
            for line in move_f.readlines():
                (stepstr, timestr) = line.split()
                steps.append(int(stepstr))
                times.append(float(timestr))
            steps = np.array(steps)
            times = np.array(times)

            basic_plot_to_file(fname=os.path.join(run_dir, fname)+".png", data=times, xvals=steps, title=move_name, marker='o', linestyle='None')

            f.write('move %s:\n' % move_name)
            f.write(' attempts: %d\n' % len(times))
            f.write(' avg time: %.4f\n' % np.mean(times))
            f.write(' median time: %.4f\n' % np.median(times))
            f.write(' std time: %.4f\n' % np.std(times))
            f.write(' max time: %.4f\n' % np.max(times))
            f.write(' min time: %.4f\n' % np.min(times))
            f.write(' total time: %.4f\n' % np.sum(times))
            f.write('\n')
            total_time += np.sum(times)
            max_step = max(max_step, steps[-1])

    f.write('total time for all moves %.4f\n' % total_time)
    f.write('total steps %d\n' % max_step)
    f.write('average time per step %.4f\n' % (total_time/max_step))
    f.close()

def analyze_run(run_dir, burnin, true_evs, plot_template_posteriors=False):

    try:
        plot_lp_trend(run_dir)
    except:
        pass

    #summarize_times(run_dir)

    for i in range(1000):
        try:
            with open(os.path.join(run_dir, 'step_%06d/pickle.sg' % i), 'rb') as f:
                sg = pickle.load(f)
            break
        except IOError:
            continue

    eids = []
    ev_re = re.compile(r'ev_(\d+).txt')
    for fname in os.listdir(run_dir):
        m = ev_re.match(fname)
        if m is not None:
            eid = int(m.group(1))
            eids.append(eid)
    print eids
    eid_results = dict()
    for eid in eids:
        if plot_template_posteriors:
            plot_ev_template_posteriors(run_dir, sg, eid, burnin)
        eid_results[eid] = analyze_event(run_dir, eid, burnin, true_evs)
    #combine_steps(run_dir)

    save_eid_results(run_dir, eid_results)

def save_eid_results(run_dir, eid_results):
    eids = sorted(eid_results.keys())

    with open(os.path.join(run_dir, "location_error.txt"), 'w') as f:
        for eid in eids:
            f.write('%d: %.2f (%.2f, %.2f)\n' % (eid, eid_results[eid]['dist_mean'], eid_results[eid]['lon_std_km'], eid_results[eid]['lat_std_km']))

if __name__ == "__main__":
    try:
        burnin = int(sys.argv[2]) if len(sys.argv) > 2 else 100

        if len(sys.argv) > 3:
            with open(sys.argv[3], 'rb') as f:
                evs = pickle.load(f)
        else:
            evs = []

        plot_template_posteriors = False
        if len(sys.argv) > 4:
            plot_template_posteriors = sys.argv[4].lower().startswith("t")


        try:
            mcmc_run = int(sys.argv[1])
            run_dir = os.path.join("logs", "mcmc", "%05d" % mcmc_run)
        except ValueError:
            run_dir = sys.argv[1]
        analyze_run(run_dir, burnin=burnin, true_evs=evs, plot_template_posteriors=plot_template_posteriors  )
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
