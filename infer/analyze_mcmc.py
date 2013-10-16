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
from sigvisa.plotting.plot import basic_plot_to_file, subplot_waveform

EVTRACE_LON, EVTRACE_LAT, EVTRACE_DEPTH, EVTRACE_TIME, EVTRACE_MB, EVTRACE_SOURCE = range(6)

def ev_lonlat_density(trace, ax, true_evid=None, frame=None):

    lonlats = trace[:, 0:2]
    n = lonlats.shape[0]

    hm = EventHeatmap(f=None, autobounds=lonlats, autobounds_quantile=0.9995, calc=False)
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

    if true_evid is not None:
        ev = get_event(evid=true_evid)
        hm.plot_locations(np.array(((ev.lon, ev.lat),)), marker="x", ms=5, mfc="blue", mec="blue", mew=3, alpha=1.0)


def load_trace(logfile, burnin):
    trace = np.loadtxt(logfile)
    trace = trace[burnin:, :]
    min_step = np.min(trace[:, 0])
    max_step = np.max(trace[:, 0])

    trace = np.array(trace[:, 1:])
    return trace, min_step, max_step


def analyze_event(run_dir, eid, burnin):
    ev_trace_file = os.path.join(run_dir, 'ev_%05d.txt' % eid)
    ev_img_file = os.path.join(run_dir, 'ev_%05d.png' % eid)

    trace, min_step, max_step = load_trace(ev_trace_file, burnin=burnin)
    print "loaded"

    f = Figure((11,8))
    gs = gridspec.GridSpec(2, 4)
    lonlat_ax = f.add_subplot(gs[0:2, 0:2])
    ev_lonlat_density(trace, lonlat_ax)
    print "plotted density"

    # also plot: depth, time, mb histograms
    depth_ax = f.add_subplot(gs[0, 2])
    plot_density(trace[:, EVTRACE_DEPTH], depth_ax, "depth")
    time_ax = f.add_subplot(gs[1, 2])
    plot_density(trace[:, EVTRACE_TIME], time_ax , "time")
    mb_ax = f.add_subplot(gs[0, 3])
    plot_density(trace[:, EVTRACE_MB], mb_ax, "mb")
    print "plotted others"

    f.suptitle('eid %d (%d samples)' % (eid,trace.shape[0]))

    f.tight_layout()
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(ev_img_file, bbox_inches="tight", dpi=300)

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
    alpha = 0.2/np.log(n+5)

    for row in trace:
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

def analyze_run(run_dir, burnin):

    #plot_lp_trend(run_dir)
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
        #analyze_event(run_dir, eid, burnin)
        plot_ev_template_posteriors(run_dir, sg, eid, burnin)
    #combine_steps(run_dir)



if __name__ == "__main__":
    try:
        burnin = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        try:
            mcmc_run = int(sys.argv[1])
            run_dir = os.path.join("logs", "mcmc", "%05d" % mcmc_run)
        except ValueError:
            run_dir = sys.argv[1]
        analyze_run(run_dir, burnin=burnin)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
