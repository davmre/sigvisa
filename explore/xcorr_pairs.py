import numpy as np
import sys, itertools

from optparse import OptionParser

from signals.template_models.paired_exp import PairedExpTemplateModel
from source.event import Event
from sigvisa import *
from signals.io import fetch_waveform

def extracted_wave_fname(sta, chan, phase, window_len, filter_str, evid):

    fdir = os.path.join("waves", sta, chan, "%s_%d" (phase, window_len), filter_str.replace(';', '_'))
    fname = str(evid) +  ".dat"
    return fdir, fname

def extract_phase_window(sta, chan, phase, window_len, filter_str, evid, tm):
    PAD = 10

    ev = Event(int(evid))
    atime = ev.time + tm.travel_time(ev, sta, phase)
    print ev
    print sta, chan, atime, atime+window_len, PAD
    wave = fetch_waveform(sta, chan, atime - 5, atime + window_len, pad_seconds = PAD)
    pad_samples = wave['srate']*PAD
    filtered = wave.filter(filter_str)
    d = filtered.data.filled(float('nan'))[pad_samples:-pad_samples]

    fdir, fname = extracted_wave_fname(sta, chan, phase, window_len, filter_str, evid)
    fullpath = os.path.join(fdir, fname)

    if not os.path.exists(fullpath):
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
    parser.add_option("-p", "--phase", dest="phase", default="P", type="str", help="phase to extract / cross-correlate")
    parser.add_option("--window_len", dest="window_len", default=30.0, type=float, help="length of window to extract / cross-correlate")

    (options, args) = parser.parse_args()

    f = open(options.pairsfile)
    pairs = [line.split(',') for line in f]
    f.close()
    d = zip(*pairs)
    evids = set(d[0] + d[1])

    s = Sigvisa()
    tm = PairedExpTemplateModel("fake_run_name", model_type="dummy")


    f = open(options.outfile, 'w')

    for (evid1, evid2, dist) in pairs:
        wave1 = extract_phase_window(options.sta, options.chan, options.phase, options.window_len, options.filter_str, evid1, tm)
        wave2 = extract_phase_window(options.sta, options.chan, options.phase, options.window_len, options.filter_str, evid1, tm)
        xcmax = xcorr(wave1, wave2)
        print "evid1", evid1, "evid2", evid2, "dist", dist, "xc peak %.3f" % (xcmax)
        f.write("%s,%s,%s, %f\n" % (evid1, evid2, dist, xcmax))
    f.close()


def xcorr(a, b):

    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(b)))

    xc = np.correlate(a, b, 'full')
    N = len(a)
    unbiased = np.array([float(N)/(N- np.abs(N-i)) for i in range(1, 2*N)])
    xc *= unbiased

    xcmax = np.max(xc[N - 300 : N+300])
    return xcmax

#    np.savetxt("xc.txt", xc)


if __name__ == "__main__":
    main()
