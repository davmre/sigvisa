import numpy as np
from sigvisa import Sigvisa
import os

from sigvisa.signals.io import fetch_waveform
from sigvisa.source.event import get_event

from sigvisa.models.ttime import tt_predict
from sigvisa.infer.template_xc import fastxc
from sigvisa.utils.fileutils import mkdir_p

import cPickle as pickle

def get_training_arrivals(sta, phase, evids, band):
    s = Sigvisa()
    
    evs = []
    signals = []
    for i, evid in enumerate(evids):
        ev = get_event(evid=evid)
        try:
            atime = ev.time + tt_predict(ev, sta, phase)
        except:
            continue

        try:
            padding = 20
            signal = fetch_waveform(sta, "auto", atime-5.0, atime+15.0, pad_seconds=padding)
            filtered = signal.filter(band)
            srate = filtered["srate"]
            body = filtered.data.data[padding*srate:-padding*srate]
        except Exception as e:
            print e
            continue
            
        evs.append(ev)
        signals.append(body)
        
        if i % 20 == 0:
            print "%s %s arrival %d of %d" % (sta, phase, i, len(evids))
        
    return evs, signals

def save_training_arrivals(sta, phase, evids, band, base_dir):
    mkdir_p(base_dir)
    fname = "%s_%s_%s_training_signals.pkl" % (sta, phase, band)
    evs, signals = get_training_arrivals(sta, phase, evids, band)

    with open(os.path.join(base_dir, fname), "wb") as f:
        pickle.dump((evs, signals), f)

    print "dumped training signals to", fname


def load_arrivals(sta, phase, band, base_dir, label="training"):
    fname = os.path.join(base_dir, "%s_%s_%s_%s_signals.pkl" % (sta, phase, band, label))
    print fname
    with open(fname, "rb") as f:
        evs, signals = pickle.load(f)
    return evs, signals   


def batch_save_training():

    evids = [int(evid) for evid in np.loadtxt("/home/dmoore/python/sigvisa/notebooks/thesis/train_evids.txt")]
    band = "freq_0.8_4.5"
    stas = "ANMO,ELK,IL31,KDAK,NEW,NV01,PD31,PFO,TX01,ULM,YBH,YKR8".split(",")
    phases = "P,Pn,Pg,Sn,Lg".split(",")

    base_dir = "/home/dmoore/ctbt_data/correlation/extracted/"

    for sta in stas:
        for phase in phases:
            try:
                save_training_arrivals(sta, phase, evids, band, base_dir)
            except Exception as e:
                print e
                pass

if __name__ == "__main__":   
    batch_save_training()
