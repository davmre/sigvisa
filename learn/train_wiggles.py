import os, errno, sys, time, traceback
import numpy as np
from scipy import stats


from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser

import plot
import sigvisa
import utils.geog
import obspy.signal.util
import itertools

from signals.armodel.learner import ARLearner
from signals.armodel.model import ARModel

def save_wiggles(wave, tm, run_name, template_params=None, snr_threshold=1, length=0):
    """
    Extract wiggles from a wave (under a given set of template
    params), and save to a file. Also save a record of the extracted
    wiggles to the database.
    """

    (phases, vals) = template_params


    s = Sigvisa()
    band = wave['band']
    sta = wave['sta']
    siteid = s.name_to_siteid_minus1[sta]+1
    chan = wave['chan']
    evid = wave['evid']

    arrivals = wave['event_arrivals']

    if template_params is not None:
        tag=""
        tmpl = tm.generate_template_waveform(template_params, model_waveform=wave)
        wiggles, l = extract_wiggles(wave, tmpl, threshold=snr_threshold)
    else:
        tag="_raw"
        wiggles, l = extract_wiggles(wave, None, length=length)

    for (pidx, phase) in enumerate(phases):
        if wiggles[pidx] is None or len(wiggles[pidx]) == 0:
            continue
        else:
            print "saving wiggles for phase", phase
            dirname = os.path.join("wiggles", run_name, sta, phase, )
            fname = os.path.join(dirname, "%d_%s%s.dat" % (evid, chan, tag))
            ensure_dir_exists(dirname)
            print "saving phase %s len %d" % (phase, len(wiggles[pidx]))
            np.savetxt(fname, np.array(wiggles[pidx]))

            try:
                sql_query = "INSERT INTO sigvisa_wiggle_wfdisc (run_name, arid, siteid, phaseid, band, chan, evid, fname, snr) VALUES (%d, %d, %d, %d, '%s', '%s', %d, '%s', %f)" % (run_name, arrivals[pidx, AR_ARID_COL], siteid, phaseid, band, chan, evid, fname, snr_threshold)
                cursor.execute(sql_query)
            except:
                print "DB error inserting wiggle description (probably duplicate key), continuing..."


def extract_wiggles(tr, tmpl, arrs, threshold=2.5, length=None):

    srate = tr.stats.sampling_rate
    st = tr.stats.starttime_unix

    if tmpl is not None:
        nf = tmpl.stats.noise_floor

    wiggles = []
    for (phase_idx, phase) in enumerate(arrs["arrival_phases"]):
        start_wiggle = arrs["arrivals"][phase_idx]-5
        start_idx = np.ceil((start_wiggle - st)*srate)

        if tmpl is not None:
            snr_test = lambda end_idx : tmpl[end_idx]/nf < np.exp(threshold)
        else:
            snr_test = lambda end_idx : (end_idx - start_idx)/srate > length
        snr_test = lambda end_idx : False

        try:
            next_phase_idx = np.ceil((arrs["arrivals"][phase_idx+1] - st)*srate)
        except:
            next_phase_idx = np.float('inf')
        for t in range(45):
            end_idx = start_idx + np.ceil(srate*t)
            if (end_idx >= next_phase_idx) or (snr_test(end_idx)):
                break

        if tmpl is not None:
            wiggle = tr[start_idx:end_idx] / tmpl[start_idx:end_idx]
        else:
            wiggle = tr[start_idx:end_idx]
        wiggles.append(wiggle)

    return wiggles, (end_idx - start_idx)/srate

def learn_wiggle_params(sigmodel, env, smoothed, phaseids, params):

    tmpl = get_template(sigmodel, env, phaseids, params)
    diff = subtract_traces(env, tmpl)

    # we don't want wiggles from the first (say) 25 secs, since some of that is onset and some of that is peak
    # we also only trust our wiggles if the envelope is, say, 10 times greater than the noise floor (so a height of 2.5 in natural-log-space)
    models = []
    for (phase_idx, phase) in enumerate(phaseids):
        start_wiggle = params[phase_idx, ARR_TIME_PARAM] + 20
        start_idx = np.ceil((start_wiggle - st)*srate)
        for t in range(100):
            end_idx = start_idx + np.ceil(srate*t)
            if tmpl[end_idx] - nf < 3:
                break
        l = (end_idx - start_idx)/srate
        if l < 8:
            import pdb
            pdb.set_trace()
            print "couldn't learn wiggle params for phase %d!" % (phase,)
            models.append(None)
            continue
        wiggle_train = diff.data[start_idx:end_idx]

        f = open("wiggles%d" % (phase,), 'w')
        for d in wiggle_train:
            f.write(str(d) + "\n")
        f.close()

        ar_learner = ARLearner(wiggle_train, srate)
        arparams, std = ar_learner.yulewalker(17)
        em = ErrorModel(0, std)
        print "learned wiggle params %s std %f mean %f from %f seconds of data" % (arparams, std, ar_learner.c, (end_idx - start_idx)/srate)
        wiggle_model = ARModel(arparams, em, c = ar_learner.c)
        models.append(wiggle_model)
    return models




def demo_get_wiggles():

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()
    tr, smoothed, tmpl, phases, wiggles, wiggles_smooth = get_wiggles(cursor, sigmodel, 5301405, 2)
    print tr
    print smoothed
    print tmpl
    print phases
    print wiggles
    print wiggles_smooth

def get_wiggles(cursor, sigmodel, evid, siteid, chan='BHZ', band='narrow_envelope_2.00_3.00', wiggle_threshold=2):
    """

    Arguments:
    cursor, sigmodel: objects initialized with sigvisa_util.init_sigmodel()
    evid, siteid: event ID (from leb_origin DB table) and site id (from static_siteid table)
    chan, band: channel and band strings
    wiggle_threshold: log-height above the noise level at which we cut off wiggle extraction (too close to the noise level and the fluctuations we see might be from noise rather than from wiggles). (TODO: determine this automatically using the learned noise variance)

    Returns:
    tr: A Trace object containing the log-envelope for the given band/channel, beginning 30 seconds before the first phase arrival associated with the given event, and continuing for 170 seconds after the final phase arrival.
    smoothed: The same as tr, but smoothed using a moving average (currently a Hamming window of length approximately 7.5 seconds)
    tmpl: A Trace object covering the same time period as tr and smoothed, but containing an empirically-fit log-envelope template.
    phases: a list of strings, giving the phase names for which wiggles were extracted
    wiggles: a list of wiggles (each in the form of an np.array object) extracted from the (unsmoothed) log-envelope.
    wiggles_smooth: a list of wiggles extracted from the smoothed log-envelope.
    """


    # load the relevant traces

    arrival_segment, smoothed_segment, arrs = load_segments(cursor, evid, siteid, ar_noise=False, chans=[chan,], bands=[band,])
    tr = arrival_segment[chan][band]
    smoothed = smoothed_segment[chan][band]

    # fit an envelope template
    start_params, phaseids, bounds, bounds_fp = find_starting_params(arrs, smoothed)
    start_params = remove_peak(start_params)
    start_params = start_params.flatten()
    bounds = bounds_fp

    c = sigvisa.canonical_channel_num(chan)
    b = sigvisa.canonical_band_num(band)
    sigmodel.set_noise_process(siteid, b, c, smoothed.stats.noise_floor, 1, np.array((.8,)))
    sigmodel.set_wiggle_process(siteid, b, 1, 1, np.array((.8,)))

    narrs = len(arrs["arrivals"])
    arr_times = np.reshape(np.array(arrs["arrivals"]), (narrs, -1))
    assem_params = lambda params: np.hstack([arr_times, restore_peak(np.reshape(params, (narrs, -1)))])
    f = lambda params : c_cost(sigmodel, smoothed, phaseids, assem_params(params), iid=True)
    best_params, best_cost = optimize(f, start_params, bounds, phaseids, method="simplex", by_phase=False)

    print "start params"
    print_params(assem_params(start_params))
    print "found params"
    print_params(assem_params(best_params))

    tmpl = get_template(sigmodel, tr, phaseids, assem_params(best_params))
    tmpls = get_template(sigmodel, tr, phaseids, assem_params(start_params))
    diff = subtract_traces(tr, tmpl)
    diff_smooth = subtract_traces(smoothed, tmpl)

    # p/s wiggles
    wiggles = extract_wiggles(tr, tmpl, arrs, threshold=1)
    wiggles_smooth = extract_wiggles(smoothed, tmpl, arrs, threshold=1)

    return tr, smoothed, tmpl, arrs["arrival_phases"], wiggles, wiggles_smooth


def main():

    cursor = db.connect().cursor()

    parser = OptionParser()
    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str", help="siteid of station for which to learn wiggle model (default: all)")
    parser.add_option("-r", "--run_names", dest="run_names", default=None, type="str", help="run_name of the extracted wiggles to use")
    parser.add_option("-p", "--phaseids", dest="phaseids", default=None, type="str", help="phaseids (P_PHASES)")
    parser.add_option("-c", "--channels", dest="channels", default=None, type="str", help="channels (all)")
    parser.add_option("-o", "--outfile", dest="outfile", default="parameters/signal_wiggles.txt", type="str", help="filename to save output (parameters/signal_wiggles.txt)")
    (options, args) = parser.parse_args()

    run_names = options.run_names.split(',')
    phaseids = P_PHASEIDS if options.phaseids is None else [int(r) for r in options.phaseids.split(',')]
    channels = chans if options.channels is None else [s for s in options.channels.split(',')]
    siteids = None if options.siteids is None else [int(s) for s in options.siteids.split(',')]
    run_name_cond = "(" + " or ".join(["run_name='%s'" % r for r in run_names])  + ")"
    print run_name_cond

    f = open(options.outfile, 'w')

    for (siteid, phaseid, channel, band) in itertools.product(siteids, phaseids, channels, bands):
        short_band = band[16:]

        sta = siteid_to_sta(siteid, cursor)
        phase = phaseid_to_name(phaseid)

        print sta, phase, channel, short_band

        sql_query = "select fname from sigvisa_wiggle_wfdisc where %s and siteid=%d and phaseid=%d and band='%s' and chan='%s'" % (run_name_cond, siteid, phaseid, short_band, channel)
        print sql_query
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        wiggles = []
        for row in rows:
            fname = row[0]
            print " loading %s..." % fname
            w = np.loadtxt(fname)

            if len(w) > 500:
                wiggles.append(w)
        print "loaded %d wiggles." % len(wiggles)

        if len(wiggles) > 5:
            ar_learner = ARLearner(wiggles)
            params, std = ar_learner.yulewalker(20)
            params_str = str(len(params)) + " " + " ".join([str(p) for p in params])
            line = "%s %s %s %s %f %f %s" % (sta, phase, channel, short_band, ar_learner.c, std, params_str)
        else:
            line = "%s %s %s %s 1 0.042897 2 1.15979568629 -0.162206492945" % (sta, phase, channel, short_band)
        print "writing line", line
        f.write(line + "\n")
    f.close()

if __name__ == "__main__":
    main()
