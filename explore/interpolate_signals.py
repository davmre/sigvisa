import os

import numpy as np

from sigvisa import Sigvisa
from sigvisa.utils.geog import dist_km
from sigvisa.source.event import get_event
from sigvisa.infer.coarse_to_fine_init import ModelSpec, EventRunSpec, TimeRangeRunSpec, do_coarse_to_fine, initialize_from, do_inference, initialize_sg
from sigvisa.infer.template_xc import fastxc

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from sigvisa.plotting.plot import subplot_waveform

def candidate_evs(nearby_km=30, min_nearby=2, max_nearby=10, runid=18):
    s = Sigvisa()

    all_train_evids = np.loadtxt(os.path.join(s.homedir, "notebooks", "thesis", "train_evids.txt"), dtype=int)

    
    evid_query = "select distinct evid from sigvisa_coda_fit where runid=%d" % int(runid)
    actual_train_evids = [evid[0] for evid in  s.sql(evid_query)]

    s1 = set(all_train_evids)
    s2 = set(actual_train_evids)
    untrained = list(s1-s2)

    all_evs = dict([(evid, get_event(evid=evid)) for evid in all_train_evids])

    predictable_uevs = []
    for evid in untrained:
        uev = all_evs[evid]
        if uev.mb < 2.0:
            continue

        nearby = [ev for ev in all_evs.values() if dist_km((uev.lon, uev.lat), (ev.lon, ev.lat)) < nearby_km and ev.evid != uev.evid and np.abs(ev.depth - uev.depth) < 8]

        if min_nearby <= len(nearby) <= max_nearby:
            predictable_uevs.append((uev, nearby))

    return predictable_uevs


def plot_alignment(ax, signal, pred_signal, target_ev, ev, sta, best_idx=None):
    
    n = len(signal)
    t = np.linspace(0, n/10.0, n)
    ax.plot(t, signal, color="black", linewidth=1.0)
    
    print len(signal)
    print len(pred_signal)

    xcs = fastxc(pred_signal, signal)
    if best_idx is None:
        best_idx = np.argmax(xcs)
    best_xc = xcs[best_idx]

    pn = len(pred_signal)
    n1 = np.max(np.abs(signal[best_idx:best_idx + pn]))
    n2 = np.max(np.abs(pred_signal))
    npred_signal = pred_signal * (n1/n2)

    t2 = np.linspace(best_idx/10.0, (best_idx + pn)/10.0, pn)
    ax.plot(t2, npred_signal, color="blue", linewidth=1.0)

    ax.set_title("xc %.2f, dist %.1f, evid %d sta %s" % (best_xc, dist_km((ev.lon, ev.lat), (target_ev.lon, target_ev.lat)), ev.evid, sta))

    return best_idx

def main():
    
    phase = "Pg"
    candidates = candidate_evs()

    stas = "ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA".split(",")
    wiggle_family = "db4_2.0_3_20.0"
    hz = 10
    runid = 18

    pp = PdfPages("visualize_corr.pdf")


    for sta in stas:

        for uev, nearby_evs in candidates:
            rs = EventRunSpec(evs=[uev,], stas=[sta,], initialize_events=True)
            ms = ModelSpec(template_model_type="gpparam", wiggle_family=wiggle_family, 
                           min_mb=1.0,
                           phases=(phase,),
                           wiggle_model_type="gplocal+lld+none", 
                           raw_signals=True, 
                           max_hz=hz,
                           runids=(runid,))
            try:
                sg = rs.build_sg(ms)
            except Exception as e:
                print e
                continue
            initialize_sg(sg, ms, rs)
            for n in sg.extended_evnodes[1]:
                if n in sg.evnodes[1].values(): continue
                if n.deterministic(): continue
                n.parent_predict()

            try:
                wn = sg.station_waves[sta][0]
            except:
                continue
            wn._parent_values()
            lp = wn.log_p()

            actual_signal = wn.get_value().copy()

            wn.unfix_value()
            wn.parent_predict()
            pred_uev_signal = wn.get_value().copy()
            try:
                first_idx = np.min(np.arange(len(pred_uev_signal.data)-400)[pred_uev_signal.data[200:-200] != wn.nm.c]) + 200
            except:
                continue

            last_idx = first_idx + 200
            pred_uev_signal = pred_uev_signal[first_idx:last_idx]
            actual_signal = actual_signal[first_idx-200:last_idx+200]

            #pred_uev_signal = pred_signal(sg, uev, phase)
            #pred_nearby_signals = [pred_signal(sg, nev, phase) for nev in nearby_evs]

            xcs = fastxc(pred_uev_signal, actual_signal)
            best_idx = np.argmax(xcs)
            best_xc = xcs[best_idx]
            if best_xc < 0.3:
                continue
    
            fig = plt.figure()
            gs = gridspec.GridSpec(len(nearby_evs)+1, 1)
            
            ax = fig.add_subplot(gs[0, 0])
            plot_alignment(ax, actual_signal, pred_uev_signal, uev, uev, sta, best_idx=best_idx)

            for i, nev in enumerate(nearby_evs):
                ax = fig.add_subplot(gs[i+1, 0])
                sg.set_event(1, nev)
                wn.parent_predict()
                pred_nev_signal = wn.get_value().copy()[first_idx:last_idx]
                plot_alignment(ax, actual_signal, pred_nev_signal, uev, nev, sta, best_idx=best_idx)
            
            plt.tight_layout()
            pp.savefig()
            
    pp.close()

        
if __name__ == "__main__":
    main()
