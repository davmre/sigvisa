import numpy as np
import sys
import os
import itertools

from optparse import OptionParser

from sigvisa.source.event import get_event
from sigvisa import *
from sigvisa.signals.io import fetch_waveform,MissingWaveform
from sigvisa.database.signal_data import ensure_dir_exists


def extracted_wave_fname(sta, chan, phase, window_len, filter_str, evid):

    fdir = os.path.join("waves", sta, chan, filter_str.replace(';', '_'))
    fname = str(int(evid)).strip() + "_%s_%ds" % (phase, window_len) + ".dat"

    return fdir, fname


def extract_phase_window(sta, chan, phase, atime, window_len, filter_str, evid, leadin_s=5, cache=False):
    PAD = 10

    ev = get_event(int(evid))
    load_from_db = not cache

    if cache:
        fdir, fname = extracted_wave_fname(sta, chan, phase, window_len, filter_str, evid)
        fullpath = os.path.join(fdir, fname)
        try:
            d = np.loadtxt(fullpath)
            srate = 40
        except Exception as e:
#            print e
            load_from_db = True

    if load_from_db:
        wave = fetch_waveform(sta, chan, atime - leadin_s, atime + window_len, pad_seconds=PAD)

        filtered = wave.filter(filter_str)
        pad_samples = filtered['srate'] * PAD
        d = filtered.data.filled(float('nan'))[pad_samples:-pad_samples]

        if cache:
            if not os.path.exists(fullpath):
                ensure_dir_exists(fdir)
                print "saved to", fullpath
                np.savetxt(fullpath, d)
        srate = filtered['srate']

    return d, srate


def main():
    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="name of station")
    parser.add_option("-c", "--chan", dest="chan", default="vertical", type="str", help="channel to correlate")
    parser.add_option(
        "-f", "--filter_str", dest="filter_str", default="freq_0.8_4.5", type="str", help="filter string to process waveforms")
    parser.add_option(
        "-i", "--pairsfile", dest="pairsfile", default="", type="str", help="load potential doublets from this file")
    parser.add_option("-o", "--outfile", dest="outfile", default="", type="str", help="save doublets to this file")
#    parser.add_option("-p", "--phase", dest="phase", default="P", type="str", help="phase to extract / cross-correlate")
    parser.add_option(
        "--window_lengths", dest="window_lengths", default="10,30.0", type="str", help="comma-separated lengths of window to extract / cross-correlate")

    (options, args) = parser.parse_args()

    f = open(options.pairsfile)
    pairs = [line.split(',') for line in f]
    f.close()

    s = Sigvisa()

    window_lengths = [float(wl) for wl in options.window_lengths.split(',')]
    max_len = np.max(window_lengths)

    chan=options.chan
    if chan=="vertical":
        cursor = s.dbconn.cursor()
        if s.earthmodel.site_info(options.sta, 0.0)[3] == 1:
            cursor.execute("select refsta from static_site where sta='%s'" % options.sta)
            selection = cursor.fetchone()[0]
        else:
            selection = options.sta
        cursor.close()
        print selection
        chan = s.default_vertical_channel[selection]


    f = open(options.outfile, 'w')


    for (evid1, evid2, dist, atime1, atime2, phase1, phase2) in pairs:
        evid1 = int(evid1)
        evid2 = int(evid2)
        dist = float(dist)
        atime1 = float(atime1)
        atime2 = float(atime2)
        phase1 = phase1.strip()
        phase2 = phase2.strip()

        
        leadin_s = 5.0
        try:
            wave1,srate = extract_phase_window(options.sta, chan, phase1, atime1, max_len+5, options.filter_str, evid1, leadin_s=leadin_s)
            wave2,srate = extract_phase_window(options.sta, chan, phase2, atime2, max_len+5, options.filter_str, evid2, leadin_s=leadin_s)
            env1,srate = extract_phase_window(options.sta, chan, phase1, atime1, max_len+5, options.filter_str+";env", evid1, leadin_s=leadin_s)
            env2,srate = extract_phase_window(options.sta, chan, phase2, atime2, max_len+5, options.filter_str+";env", evid2, leadin_s=leadin_s)
        except MissingWaveform as e:
            print e
            continue

        if abs(atime1 - atime2) < 60:
            print "skipping simultaneous events", evid1, evid2
            continue

        f.write("%s,%s, %.1f, %.1f, %.1f" % (evid1, evid2, dist, atime1, atime2))
        leadin_samples = int(leadin_s*srate)
        for len_s in window_lengths:
            len_samples  = int(len_s*srate)

            wave_xcmax, wave_offset = twoway_xcorr(wave1, wave2, window_len=len_samples, offset_max=leadin_samples)
            env_xcmax, env_offset = twoway_xcorr(env1, env2, window_len=len_samples, offset_max=leadin_samples)

            f.write(', %.4f, %d, %.4f, %d' % (wave_xcmax, wave_offset, env_xcmax, env_offset))
        f.write('\n')
        print "evid1", evid1, "evid2", evid2, "dist", dist

    f.close()

def twoway_xcorr(a, b, window_len, offset_max):

    # correlate a window of 'a' against 'b'
    a_window = a[offset_max:offset_max+window_len]
    b_window = b[offset_max:offset_max+window_len]
    a_large = a[:2*offset_max+window_len]
    b_large = b[:2*offset_max+window_len]

    xcmax_a, offset_a = xcorr_valid(a_window, b_large)
    xcmax_b, offset_b = xcorr_valid(b_window, a_large)

    if xcmax_a > xcmax_b:
        return xcmax_a, offset_a - offset_max
    else:
        return xcmax_b, offset_max - offset_b

def xcorr_valid(a,b):
    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(a)))

    xc = np.correlate(a, b, 'valid')
    xcmax = np.max(xc)
    offset = np.argmax(xc)
    return xcmax, offset


def xcorr(a, b, window=200):

    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(a)))

    xc = np.correlate(a, b, 'full')
    N = len(a)
    unbiased = np.array([float(N) / (N - np.abs(N - i)) for i in range(1, 2 * N)])
    xc *= unbiased


    midpoint = N-1
    shifted = xc[midpoint - window: midpoint + window]
    xcmax = np.max(shifted)
    offset = np.argmax(shifted) - window

    return xcmax, offset

#    np.savetxt("xc.txt", xc)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        import pdb, sys, traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
