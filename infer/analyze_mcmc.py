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

from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa.plotting.histogram import plot_density
from sigvisa.plotting.plot import basic_plot_to_file, subplot_waveform, savefig
import sigvisa.utils.geog as geog

EVTRACE_LON, EVTRACE_LAT, EVTRACE_DEPTH, EVTRACE_TIME, EVTRACE_MB, EVTRACE_SOURCE = range(6)

def ev_lonlat_density(trace, ax, true_evs=None, frame=None, bounds=None, text=True):

    lonlats = trace[:, 0:2]
    n = lonlats.shape[0]

    if bounds is None:
        bounds = {'autobounds': lonlats, 'autobounds_quantile': .9995}
    hm = EventHeatmap(f=None, calc=False, **bounds)
    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)

    baseline_alpha = 1.0 / np.log(n+1)
    alpha_fade_time = 500
    if frame is not None:
        alpha = np.ones((frame,)) * baseline_alpha
        t = min(frame,alpha_fade_time)
        alpha[-t:] = np.linspace(baseline_alpha, 0.2, alpha_fade_time)[-t:]
    else:
        alpha = baseline_alpha

    #hm.plot_locations(X, marker=".", ms=6, mfc="red", mec="none", mew=0, alpha=0.2)
    scplot = hm.plot_locations(lonlats, marker=".", ms=8, mfc="red", mew=0, mec="none", alpha=alpha)

    mean_lon = np.mean(trace[:, 0])
    mean_lat = np.mean(trace[:, 1])
    lon_std =  np.std(trace[:, 0])
    lat_std =  np.std(trace[:, 1])
    lon_std_km = geog.dist_km((mean_lon, mean_lat), (mean_lon+lon_std, mean_lat))
    lat_std_km = geog.dist_km((mean_lon, mean_lat), (mean_lon, mean_lat+lat_std))
    txt = "mean: %.2f, %.2f\nstd: %.2f, %.2f\nstd_km: %.1f, %.1f" % (mean_lon, mean_lat, lon_std, lat_std, lon_std_km, lat_std_km)

    true_ev = None
    best_distance = np.float('inf')
    for ev in true_evs:
        dist = geog.dist_km((mean_lon, mean_lat), (ev.lon, ev.lat))
        if dist < best_distance:
            best_distance = dist
            true_ev = ev

    if true_ev is not None:
        txt += '\ntrue: %.2f, %.2f\n' % (true_ev.lon, true_ev.lat)
        txt += 'd(mean, true) = %.2f\n' % geog.dist_km((mean_lon, mean_lat), (true_ev.lon, true_ev.lat))
        txt += 'd(proposal, true) = %.2f' % geog.dist_km((trace[0, 0], trace[0, 1]), (true_ev.lon, true_ev.lat))

    if text:
        ax.text(0, 1, txt, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='purple')


    if true_ev is not None:
        hm.plot_locations(np.array(((true_ev.lon, true_ev.lat),)), marker="x", ms=5, mfc="blue", mec="blue", mew=3, alpha=1.0)

    return true_ev, txt

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

    trace = np.array([row for row in trace if row[0] > burnin])
    if len(trace) == 0:
        return trace, 0, 0
    min_step = np.min(trace[:, 0])
    max_step = np.max(trace[:, 0])

    trace = np.array(trace[:, 1:])
    return trace, min_step, max_step


def analyze_event(run_dir, eid, burnin, true_evs=None):
    ev_trace_file = os.path.join(run_dir, 'ev_%05d.txt' % eid)
    ev_img_file = os.path.join(run_dir, 'ev_%05d.png' % eid)
    ev_txt = os.path.join(run_dir, 'ev_%05d_results.txt' % eid)
    ev_dir = os.path.join(run_dir, 'ev_%05d' % eid)

    trace, min_step, max_step = load_trace(ev_trace_file, burnin=burnin)
    if len(trace) == 0:
        return
    print "loaded"

    # save big event image
    f = Figure((11,8))
    gs = gridspec.GridSpec(2, 4)
    lonlat_ax = f.add_subplot(gs[0:2, 0:2])
    true_ev, txt = ev_lonlat_density(trace, lonlat_ax, true_evs=true_evs)
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


    # also save individual images
    f = Figure((8,8))
    ax = f.add_subplot(1,1,1)
    true_ev, txt = ev_lonlat_density(trace, ax, true_evs=true_evs, text=False)
    savefig(os.path.join(ev_dir, 'posterior_loc.png'), f, bbox_inches="tight", dpi=300)

    if true_ev is not None:
        f = Figure((8,8))
        ax = f.add_subplot(1,1,1)
        bounds = {'center': (true_ev.lon, true_ev.lat), 'width_deg': 5.0, 'height_deg': 5.0}
        true_ev, txt = ev_lonlat_density(trace, ax, true_evs=true_evs, text=False, bounds=bounds)
        savefig(os.path.join(ev_dir, 'posterior_loc_big.png'), f, bbox_inches="tight", dpi=300)

        f = Figure((8,8))
        ax = f.add_subplot(1,1,1)
        bounds = {'center': (true_ev.lon, true_ev.lat), 'width_deg': 30.0, 'height_deg': 30.0}
        true_ev, txt = ev_lonlat_density(trace, ax, true_evs=true_evs, text=False, bounds=bounds)
        savefig(os.path.join(ev_dir, 'posterior_loc_reallybig.png'), f, bbox_inches="tight", dpi=300)

    f = Figure((8,8))
    ax = f.add_subplot(1,1,1)
    plot_density(trace[:, EVTRACE_MB], ax, "mb", draw_stats=False)
    savefig(os.path.join(ev_dir, 'posterior_mb.png'), f, bbox_inches="tight", dpi=300)

    f = Figure((8,8))
    ax = f.add_subplot(1,1,1)
    plot_density(trace[:, EVTRACE_TIME], ax, "time", draw_stats=False)
    savefig(os.path.join(ev_dir, 'posterior_time.png'), f, bbox_inches="tight", dpi=300)

    f = Figure((8,8))
    ax = f.add_subplot(1,1,1)
    plot_density(trace[:, EVTRACE_DEPTH], ax, "depth", draw_stats=False)
    savefig(os.path.join(ev_dir, 'posterior_depth.png'), f, bbox_inches="tight", dpi=300)


def plot_arrival_template_posterior(ev_dir, sg, eid, wn_lbl, phase, burnin):

    lbl = 'tmpl_%d_%s_%s' % (eid, wn_lbl, phase)
    tmpl_trace_file = os.path.join(ev_dir, '%s' % lbl)
    tmpl_img_file = os.path.join(ev_dir, '%s.png' % lbl)

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
    plot_stime = min_atime - 10.0

    max_atime = np.max(atimes)
    plot_etime = max_atime + 100.0

    f = Figure((10, 5))
    ax = f.add_subplot(1,1,1)

    subplot_waveform(real_wave, ax, stime=plot_stime, etime=plot_etime, plot_dets=False, color='black', linewidth=0.5)

    print plot_stime, plot_etime, wn.st, wn.et

    n = trace.shape[0]
    max_idxs = 2000
    if n > max_idxs:
        idxs = np.array(np.linspace(0, n-1, max_idxs), dtype=int)
        n = max_idxs
    else:
        idxs = np.arange(n)

    alpha = 0.4/np.sqrt(n+5)

    for row in trace[idxs,:]:
        v = {'arrival_time': row[0], 'peak_offset': row[1], 'coda_height': row[2], 'coda_decay': row[3]}
        sg.set_template(eid, wn.sta, phase, wn.band, wn.chan, v)
        tmpl_stime = v['arrival_time']
        tmpl_etime = min(tmpl_stime + v['peak_offset'] + (np.log(0.02 * wn.nm.c) - v['coda_height'])/v['coda_decay'], plot_etime)

        wn.parent_predict()
        subplot_waveform(wn.get_wave(), ax, stime=tmpl_stime, etime=tmpl_etime, plot_dets=False, color='green', linewidth=0.5, alpha=alpha, fill_y2=wn.nm.c)

    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(tmpl_img_file, bbox_inches="tight", dpi=300)

    wn.set_value(real_wave.data)
    wn.fix_value()

def plot_ev_template_posteriors(run_dir, sg, eid, burnin):
    ev_dir = os.path.join(run_dir, 'ev_%05d' % eid)
    r = re.compile('tmpl_%d_' % eid + r'(.+)_([A-Za-z])+')
    for fname in os.listdir(ev_dir):
        print fname
        m = r.match(fname)
        if m is not None:
            wn_lbl = m.group(1)
            phase = m.group(2)
            plot_arrival_template_posterior(ev_dir, sg, eid, wn_lbl, phase, burnin=burnin)
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

def analyze_run(run_dir, burnin, true_evs):

    try:
        plot_lp_trend(run_dir)
    except:
        pass

    #summarize_times(run_dir)

    with open(os.path.join(run_dir, 'step_000099/pickle.sg'), 'rb') as f:
        sg = pickle.load(f)

    eids = []
    ev_re = re.compile(r'ev_(\d+).txt')
    for fname in os.listdir(run_dir):
        m = ev_re.match(fname)
        if m is not None:
            eid = int(m.group(1))
            eids.append(eid)
    print eids
    for eid in eids:
        plot_ev_template_posteriors(run_dir, sg, eid, burnin)
        analyze_event(run_dir, eid, burnin, true_evs)
    #combine_steps(run_dir)



if __name__ == "__main__":
    try:
        burnin = int(sys.argv[2]) if len(sys.argv) > 2 else 100

        if len(sys.argv) > 3:
            with open(sys.argv[3], 'rb') as f:
                evs = pickle.load(f)
        else:
            evs = []

        try:
            mcmc_run = int(sys.argv[1])
            run_dir = os.path.join("logs", "mcmc", "%05d" % mcmc_run)
        except ValueError:
            run_dir = sys.argv[1]
        analyze_run(run_dir, burnin=burnin, true_evs=evs)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)