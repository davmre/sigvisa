import numpy as np
from sigvisa import Sigvisa
import os
from optparse import OptionParser

from sigvisa.signals.io import fetch_waveform
from sigvisa.source.event import get_event

from sigvisa.models.ttime import tt_predict
from sigvisa.infer.template_xc import fastxc
from sigvisa.utils.fileutils import mkdir_p
from sigvisa.utils.geog import dist_km

import cPickle as pickle

from sigvisa.source.event import Event
from sigvisa.explore.correlation_baseline_extract import load_arrivals


def correlate(signal, threshold, stime, srate, train_signals, train_evs):

    detected_evs = []
    for i, (train_ev, train_signal) in enumerate(zip(train_evs, train_signals)):
        xcs = fastxc(train_signal, signal)
        detection_idxs,  = (xcs > threshold).nonzero()
        detection_times = detection_idxs / srate + stime
        detection_scores = xcs[detection_idxs]

        for t, score in zip(detection_times, detection_scores):
            dev = Event(lon=train_ev.lon, lat=train_ev.lat, depth=train_ev.depth, mb=train_ev.mb, time=t)
            detected_evs.append((dev, score))

        if i % 100 == 0:
            print "correlated %d training events, built %d potential hits" % (i, len(detected_evs))

    return detected_evs

def suppress_duplicate_events(detected_evs, dist_threshold_km=60, time_threshold_s=50):
    
    events_suppressed = np.inf
    suppressed_idxs = set()
    while events_suppressed > 0:
        events_suppressed = 0

        for i, (ev1, score1) in enumerate(detected_evs):
            if i in suppressed_idxs: continue

            for ji, (ev2, score2) in enumerate(detected_evs[i+1:]):
                j = i+ji+1
                if j in suppressed_idxs: continue

                if np.abs(ev2.time - ev1.time) > time_threshold_s: continue
                if dist_km((ev1.lon, ev1.lat), (ev2.lon, ev2.lat)) > dist_threshold_km: continue
                if score2 > score1:
                    # suppress ev1
                    suppressed_idxs.add(i)
                else:
                    # suppress ev2
                    suppressed_idxs.add(j)
                events_suppressed += 1
                break

    return [detected_evs[i] for i in range(len(detected_evs)) if i not in suppressed_idxs]

def scan_for_correlated_events(sta, phase, band, threshold, stime, etime):
    
    n_blocks = np.ceil((etime-stime)/7200.0)    
    boundaries = np.linspace(stime, etime, n_blocks+1)
    block_stimes = boundaries[:-1]
    block_etimes = boundaries[1:]

    base_dir = os.path.join(os.getenv("HOME"), "ctbt_data/correlation/extracted/")
    evs, signals = load_arrivals(sta, phase, band, base_dir, label="training")

    all_scored_evs = []
    for block_stime, block_etime in zip(block_stimes, block_etimes):
        signal = fetch_waveform(sta, "auto", block_stime, block_etime, pad_seconds=20)
        filtered = signal.filter(band)

        pad_idx = 20 * filtered['srate']
        filtered_data = filtered.data.data[pad_idx:-pad_idx]
        filtered_data[np.isnan(filtered_data)] = 0.0
        print "correlating %s %s %s from %.1f to %.1f" % (sta, phase, band, block_stime, block_etime)
        scored_evs = correlate(filtered_data, threshold=threshold, 
                               stime=block_stime, srate=filtered["srate"], 
                               train_signals=signals, train_evs=evs)
        print "got", len(scored_evs), "events, suppressing duplicates"
        filtered_evs = suppress_duplicate_events(scored_evs)
        print "filtering yielded %d unique events" % (len(filtered_evs))
        all_scored_evs += filtered_evs
    
    return all_scored_evs

def save_event_list(fname, evlist):

    evdicts = []
    for (ev, score) in evlist:
        evd = ev.to_dict()
        evd['score'] = score
        evdicts.append(evd)

    with open(fname, 'w') as f:
        f.write(repr(evdicts))

def main():
    parser = OptionParser()


    parser.add_option("--stime", dest="stime", default=None, type=float, help="")
    parser.add_option("--etime", dest="etime", default=None, type=float, help="")
    parser.add_option("--threshold", dest="threshold", default=0.3, type=float, help="")
    parser.add_option("--stas", dest="stas", default="ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA", type="str", help="")
    parser.add_option("--phases", dest="phases", default="P,Pn,Pg,Sn,Lg", type="str", help="")
    parser.add_option("--band", dest="band", default="freq_0.8_4.5", type="str", help="")

    (options, args) = parser.parse_args()
    
    stas = options.stas.split(",")
    phases = options.phases.split(",")

    base_dir = os.getenv("HOME")
    mkdir_p(base_dir)

    for sta in stas:
        for phase in phases:
            fname = os.path.join(base_dir, 'corr_evs_%s_%s_%s.txt' % (sta, phase, options.band))
            try:
                evlist = scan_for_correlated_events(sta, phase, 
                                                    options.band, options.threshold, 
                                                    options.stime, options.etime)
            except Exception as e:
                print "EXCEPTION scanning for %s: %s" % (fname, e)
                continue


            save_event_list(fname, evlist)
            print "saved %d events to %s" % (len(evlist), fname)

if __name__ == "__main__":
    main()
