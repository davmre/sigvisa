import numpy as np
import sys, os, itertools

from optparse import OptionParser

from signals.template_models.paired_exp import PairedExpTemplateModel
from source.event import Event
from sigvisa import *
from signals.io import fetch_waveform
from database.signal_data import ensure_dir_exists

def extracted_wave_fname(sta, chan, phase, window_len, filter_str, evid):

    fdir = os.path.join("waves", sta, chan, filter_str.replace(';', '_'))
    fname = str(int(evid)).strip() +  "_%s_%ds" % (phase, window_len) + ".dat"

    return fdir, fname

def extract_phase_window(sta, chan, phase, atime, window_len, filter_str, evid, cache=False):
    PAD = 10

    ev = Event(int(evid))
    load_from_db = not cache

    if cache:
        fdir, fname = extracted_wave_fname(sta, chan, phase, window_len, filter_str, evid)
        fullpath = os.path.join(fdir, fname)
        try:
            d = np.loadtxt(fullpath)
        except Exception as e:
#            print e
            load_from_db = True

    if load_from_db:
        wave = fetch_waveform(sta, chan, atime - 1, atime + window_len, pad_seconds = PAD)

        pad_samples = wave['srate']*PAD
        filtered = wave.filter(filter_str)

        d = filtered.data.filled(float('nan'))[pad_samples:-pad_samples]

        if cache:
            if not os.path.exists(fullpath):
                ensure_dir_exists(fdir)
                print "saved to", fullpath
                np.savetxt(fullpath, d)

    return d

def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station")
    parser.add_option("-c", "--chan", dest="chan", default="BHZ", type="str", help="channel to correlate")
    parser.add_option("-f", "--filter_str", dest="filter_str", default="freq_0.8_3.5", type="str", help="filter string to process waveforms")
    parser.add_option("-i", "--pairsfile", dest="pairsfile", default="", type="str", help="load potential doublets from this file")
    parser.add_option("-o", "--outfile", dest="outfile", default="", type="str", help="save doublets to this file")
#    parser.add_option("-p", "--phase", dest="phase", default="P", type="str", help="phase to extract / cross-correlate")
    parser.add_option("--window_len", dest="window_len", default=30.0, type=float, help="length of window to extract / cross-correlate")

    (options, args) = parser.parse_args()

    f = open(options.pairsfile)
    pairs = [line.split(',') for line in f]
    f.close()

    s = Sigvisa()

    f = open(options.outfile, 'w')

    for (evid1, evid2, dist, atime1, atime2, phase1, phase2) in pairs:
        evid1 = int(evid1)
        evid2 = int(evid2)
        dist = float(dist)
        atime1 = float(atime1)
        atime2 = float(atime2)
        phase1 = phase1.strip()
        phase2 = phase2.strip()

        try:
            wave1 = extract_phase_window(options.sta, options.chan, phase1, atime1, options.window_len, options.filter_str, evid1)
            wave2 = extract_phase_window(options.sta, options.chan, phase2, atime2, options.window_len, options.filter_str, evid2)
        except Exception as e:
            print "exception:", e
            continue
        
        if abs(atime1-atime2)<30:
            print "skipping simultaneous events", evid1, evid2
            continue

        xcmax = xcorr(wave1, wave2)

        print "evid1", evid1, "evid2", evid2, "dist", dist, "time_diff %.1f" % (atime1- atime2), "xc peak %.3f" % (xcmax)
        f.write("%s,%s, %.1f, %.1f, %.1f, %f\n" % (evid1, evid2, dist, atime1, atime2,xcmax))
    f.close()


def xcorr(a, b):

    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(b)))

    xc = np.correlate(a, b, 'full')
    N = len(a)
    unbiased = np.array([float(N)/(N- np.abs(N-i)) for i in range(1, 2*N)])
    xc *= unbiased

    shifted = xc[N - 200 : N+200]
    xcmax = np.max(shifted)
    offset = np.argmax(shifted) - 200

    return xcmax, offset

#    np.savetxt("xc.txt", xc)


if __name__ == "__main__":
    main()
