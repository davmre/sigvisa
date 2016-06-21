import numpy as np
from sigvisa import Sigvisa
import os

from sigvisa.signals.io import fetch_waveform
from sigvisa.source.event import get_event

from sigvisa.models.ttime import tt_predict
from sigvisa.infer.template_xc import fastxc


import cPickle as pickle

def get_training_arrivals(sta, phase, runid):
    s = Sigvisa()
    query = "select f.evid, fp.arrival_time from sigvisa_coda_fit f, sigvisa_coda_fit_phase fp where f.fitid=fp.fitid and fp.phase='%s' and f.sta='%s' and f.runid=%d" % (phase, sta, runid)
    r = s.sql(query)
    
    evs = []
    signals = []
    for i, (evid, atime) in enumerate(r):
        ev = get_event(evid=evid)
        try:
            signal = fetch_waveform(sta, "auto", atime, atime+20.0, pad_seconds=20)
            filtered = signal.filter("freq_0.8_4.5;hz_10")
            body = filtered.data.data[200:-200]
        except Exception as e:
            print e
            continue
            
        evs.append(ev)
        signals.append(body)
        
        if i % 20 == 0:
            print "%s %s %d arrival %d of %d" % (sta, phase, runid, i, len(r))
        
    return evs, signals

def save_training_arrivals(sta, phase, runid):
    fname = "%s_%s_%d_training_signals.pkl" % (sta, phase, runid)
    evs, signals = get_training_arrivals(sta, phase, runid)

    with open(fname, "wb") as f:
        pickle.dump((evs, signals), f)

    print "dumped training signals to", fname


def get_validation_arrivals(sta, phase, runid):
    s = Sigvisa()
    validation_evid_file = os.path.join(s.homedir, "notebooks", "thesis", "validation_evids.txt")
    validation_evids = np.loadtxt(validation_evid_file)

    signals = []
    evs = []

    for i, evid in enumerate(validation_evids):
        try:
            ev = get_event(evid=int(evid))
            
            atime = ev.time + tt_predict(ev, sta, phase)
            signal = fetch_waveform(sta, "auto", atime-20.0, atime+60.0, pad_seconds=20)
            filtered = signal.filter("freq_0.8_4.5;hz_10")
            body = filtered.data.data[200:-200]
        except Exception as e:
            print e
            continue

        signals.append(body)
        evs.append(ev)

        if i % 20 == 0:
            print "validation %s %s %d arrival %d of %d" % (sta, phase, runid, i, len(validation_evids))


    return evs, signals

def save_validation_arrivals(sta, phase, runid):
    fname = "%s_%s_%d_validation_signals.pkl" % (sta, phase, runid)
    evs, signals = get_validation_arrivals(sta, phase, runid)

    with open(fname, "wb") as f:
        pickle.dump((evs, signals), f)

    print "dumped training signals to", fname

def load_arrivals(sta, phase, runid, label="training"):
    fname = "/home/dmoore/python/sigvisa/%s_%s_%d_%s_signals.pkl" % (sta, phase, runid, label)
    print fname
    with open(fname, "rb") as f:
        evs, signals = pickle.load(f)
    return evs, signals   


def batch_save_training():

    runid = 14
    stas = "ANMO,ELK,IL31,KDAK,NEW,NV01,PD31,PFO,TX01,ULM,YBH,YKR8".split(",")

    phases = "P,Pn,Pg,Sn,Lg".split(",")

    for sta in stas:
        for phase in phases:
            try:
                save_training_arrivals(sta, phase, runid)
            except Exception as e:
                print e
                pass

def batch_save_validation():

    runid = 14
    stas = "ANMO,ELK,IL31,KDAK,NEW,NV01,PD31,PFO,TX01,ULM,YBH,YKR8".split(",")

    phases = "P,Pn,Pg,Sn,Lg".split(",")

    #stas=["ANMO"]
    #phases=["P",]

    for sta in stas:
        for phase in phases:
            try:
                save_validation_arrivals(sta, phase, runid)
            except Exception as e:
                print e
                pass


 
if __name__ == "__main__":   
    batch_save_validation()
    batch_save_training()
