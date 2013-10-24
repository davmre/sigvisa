import numpy as np


def load_ev_trace(logfile):

    trace = []
    with open(logfile, 'r') as f:
        l = f.readline()
        trace.append([float(x) for x in l.split()])
    trace = np.array(trace)
    min_step = np.min(trace[:, 0])
    max_step = np.min(trace[:, 0])
    trace = np.array(trace[:, 1:])
    return trace, min_step, max_step

def ev_lonlat_density(trace, imgfname, true_ev=None, frame=None):

    lonlats = trace[:, 0:2]

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sigvisa.plotting.event_heatmap import EventHeatmap

    f = Figure((11,8))
    ax = f.add_subplot(111)
    hm = EventHeatmap(f=None, autobounds=lonlats, autobounds_quantile=0.9995, calc=False)
    hm.init_bmap(axes=ax)
    hm.plot_earth(y_fontsize=16, x_fontsize=16)

    baseline_alpha = 0.008
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

    canvas = FigureCanvasAgg(f)
    canvas.draw()
    f.savefig(imgfname, bbox_inches="tight", dpi=300)
