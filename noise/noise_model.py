import numpy as np
import numpy.ma as ma
import time, os

from sigvisa import Sigvisa
from obspy.signal.trigger import zDetect, recSTALTA

import sigvisa_c
from database import dataset
from database.signal_data import ensure_dir_exists

from signals.armodel.model import ARModel, ErrorModel, load_armodel_from_file
from signals.armodel.learner import ARLearner

from signals.io import fetch_waveform, MissingWaveform
from signals.common import filter_str_extract_band

import hashlib


NOISE_PAD_SECONDS = 20


# two modes:
# NOISE_HOURLY computes a single noise model for each hour at each station, and uses this for the entirety of the next hour
# NOISE_IMMEDIATE computes a new noise model for each requested starting time, based on the signal immediately prior to that time. This is more accurate but also more work.
NOISE_MODE_HOURLY, NOISE_MODE_IMMEDIATE = range(2)
NOISE_MODE = NOISE_MODE_IMMEDIATE


class NoNoiseException(Exception):
    pass




def classicSTALTAPy(a, nsta, nlta):
    """
Computes the standard STA/LTA from a given input array a. The length of
the STA is given by nsta in samples, respectively is the length of the
LTA given by nlta in samples. Written in Python.

.. note::

There exists a faster version of this trigger wrapped in C
called :func:`~obspy.signal.trigger.classicSTALTA` in this module!

:type a: NumPy ndarray
:param a: Seismic Trace
:type nsta: Int
:param nsta: Length of short time average window in samples
:type nlta: Int
:param nlta: Length of long time average window in samples
:rtype: NumPy ndarray
:return: Characteristic function of classic STA/LTA
"""
    #XXX From numpy 1.3 use numpy.lib.stride_tricks.as_strided
    # This should be faster then the for loops in this fct
    # Currently debian lenny ships 1.1.1
    m = len(a)
    # indexes start at 0, length must be subtracted by one
    nsta_1 = nsta - 1
    nlta_1 = nlta - 1
    # compute the short time average (STA)
    sta = np.zeros(len(a), dtype='float64')
    pad_sta = np.zeros(nsta_1)

#    import pdb; pdb.set_trace()

    # Tricky: Construct a big window of length len(a)-nsta. Now move this
    # window nsta points, i.e. the window "sees" every point in a at least
    # once.
    for i in xrange(nsta): # window size to smooth over
        sta = sta + np.concatenate((pad_sta, a[i:m - nsta_1 + i] ** 2))
    sta = sta / nsta
    #
    # compute the long time average (LTA)
    lta = np.zeros(len(a), dtype='float64')
    pad_lta = np.ones(nlta_1) # avoid for 0 division 0/1=0
    for i in xrange(nlta): # window size to smooth over
        lta = lta + np.concatenate((pad_lta, a[i:m - nlta_1 + i] ** 2))
    lta = lta / nlta
    #
    # pad zeros of length nlta to avoid overfit and
    # return STA/LTA ratio
    sta[0:nlta_1] = 0
    return sta / lta

def get_sta_lta_picks(wave):
    df = wave['srate']
    s = Sigvisa()

    fw = wave.filter('freq_0.5_5.0')
    stalta = recSTALTA(fw.data[df*NOISE_PAD_SECONDS+df:-df*NOISE_PAD_SECONDS-df].filled(float('nan')), int(5*df), int(10 * df))[10*df:]
#    stalta = classicSTALTAPy(fw.data[df*NOISE_PAD_SECONDS:-df*NOISE_PAD_SECONDS].filled(float('nan')), int(5 * df), int(20 * df))[20*df:]
    stalta = ma.masked_invalid(stalta)
    print "max stalta", np.max(stalta),
    print ", min stalta", np.min(stalta)



#    np.savetxt("fw.wave", fw.data)
#    np.savetxt("stalta.wave", stalta)
    if np.max(stalta) > 1.8:
        import pdb
        pdb.set_trace()

        return True
    elif np.min(stalta) < 0.4:
        return True
    else:
        return False


def model_path(sta, chan, filter_str, srate, order, hour_time=None, minute_time=None, anchor_time=None):

    if hour_time is not None:
        t = time.gmtime(hour_time)
        model_dir = "hour_%02d" % t.tm_hour
    elif anchor_time is not None:
        t = time.gmtime(anchor_time)
        model_dir = "preceding_%d" % int(anchor_time)
    elif minute_time is not None:
        t = time.gmtime(minute_time)
        model_dir = "%d" % int(minute_time)
    else:
        raise Exception("noise model path must specify either an hour or a specific minute")

    base_dir = os.path.join(os.getenv('SIGVISA_HOME'), "parameters", "noise_models", sta, str(t.tm_year), str(t.tm_mon), str(t.tm_mday))
    sanitized_filter_str = '_'.join([s for s in filter_str.split(';') if s != ""])
    model_fname = '.'.join([chan, sanitized_filter_str, "%.0f" % (srate) + "hz", str(order), 'armodel'])

    return os.path.join(base_dir, model_dir), model_fname


# load times of all arrivals at a station within a given period
def load_arrival_times(sta, stime, etime):
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    arrivals = dataset.read_station_detections(cursor, sta, stime, etime)
    if arrivals.shape[0] != 0:
        arrival_times = arrivals[:, dataset.DET_TIME_COL]
    else:
        arrival_times = np.array(())
    return arrival_times

# skip any block that includes a detected arrival, or is within some danger period  after one
def arrivals_intersect_block(block_start, block_end, arrival_times, danger_period_seconds = 400):
    for t in arrival_times:
        if t > block_start and t < block_end:
            return True
        buffer = block_start - t
        if buffer > 0 and buffer < danger_period_seconds:
            return True
    return False


def get_median_noise_wave(sta, chan, hour_start, hour_end, block_len_seconds):

    # load waveforms for two minutes within the hour
    waves = []
    blocks = []
    num_blocks = 3

    arrival_times = load_arrival_times(sta, hour_start, hour_end)

    for block_start in np.linspace(hour_start, hour_end-block_len_seconds, (3600.0/block_len_seconds)):
        # stop the search once we find enough good blocks
        if len(waves) >= num_blocks:
            break

        block_end = block_start + block_len_seconds

        # skip any block that includes a detected arrival, or is within ARRIVAL_BUFFER seconds after one
        if arrivals_intersect_block(block_start, block_end, arrival_times):
            continue
        try:
            wave = fetch_waveform(sta, chan, block_start, block_end, pad_seconds=NOISE_PAD_SECONDS)
        except Exception as e:
#            print "failed loading signal (%s, %s, %d, %d)." % (sta, chan, block_start, block_end)
            continue

        # also skip any minute for which we don't have much data
        if wave['fraction_valid'] < 0.5:
#            print (sta, chan, block_start, block_end, wave['fraction_valid'])
#            print "not enough datapoints for signal (%s, %s, %d, %d) (%.1f%% valid)." % (sta, chan, block_start, block_end, 100*wave['fraction_valid'])
            continue

        # finally, skip any block which seems to contain large spikes (even if not detected by IDC)
        if get_sta_lta_picks(wave):
#            fname = hashlib.sha1(wave.data).hexdigest()[0:6] + ".wave"
#            np.savetxt(fname, wave.data)
#            print "undetected spikes in signal (%s, %s, %d, %d)! skipping... wave dumped to %s" % (sta, chan, block_start, block_end, fname)
            continue

        waves.append(wave)
        blocks.append(block_start)

    if len(blocks) == 0:
        raise NoNoiseException("failed to load noise model for (%s, %s, %d)" % (sta, chan, hour_start))

    # choose the smallest block as our model
    waves.sort(key=lambda w : np.dot(w.data, w.data))
    idx = int(np.floor(len(waves)/2.0)) # median
#    idx = 0 # smallest
    model_wave = waves[idx]
    block_start = blocks[idx]

    print block_start - hour_start
    return model_wave, block_start

def construct_and_save_hourly_noise_models(hour, sta, chan, filter_str, srate, order, block_len_seconds=300):
    hour_start = hour*3600
    hour_end = (hour+1)*3600
    hour_dir, model_fname = model_path(sta, chan, filter_str, srate, order, hour_time=hour_start)

    # if we know which block to use for this hour, load that block
    if os.path.exists(hour_dir):
        minute_dir = os.path.realpath(hour_dir)
        minute = int(os.path.split(minute_dir)[-1])
        model_wave = fetch_waveform(sta, chan, minute, minute+block_len_seconds, pad_seconds=NOISE_PAD_SECONDS)
    # otherwise, find a block which is "safe" (no arrivals) and representative
    else:
        print "hour dir", hour_dir, "doesn't exist, computing new median minute"
        model_wave, minute = get_median_noise_wave(sta, chan, hour_start, hour_end, block_len_seconds = block_len_seconds)

    minute_dir = construct_and_save_noise_models(minute, block_len_seconds, sta, chan, filter_str, srate, order, model_wave)
    try:
        minute_dir_path = os.path.realpath(minute_dir)
        os.symlink(minute_dir_path, hour_dir)
        print "successfully created symlink"
    except OSError:
        # if symlink already exists, check to make sure it's the right thing
        current_link_target = os.path.realpath(hour_dir)

        if not current_link_target == minute_dir_path:
            raise BadParamTreeException("tried to symlink %s to %s, but symlink already exists and points to %s!" % (hour_dir, minute_dir_path, current_link_target))

def construct_and_save_immediate_noise_models(time, sta, chan, filter_str, srate, order):
    block_start, block_end = get_recent_safe_block(time, sta, chan)
    minute_dir = construct_and_save_noise_models(block_start, block_end-block_start, sta, chan, filter_str, srate, order)

    immediate_dir, model_fname = model_path(sta, chan, filter_str, srate, order, anchor_time=time)
    try:
        minute_dir_path = os.path.realpath(minute_dir)
        os.symlink(minute_dir_path, immediate_dir)
        print "successfully created symlink"
    except OSError:
        # if symlink already exists, check to make sure it's the right thing
        current_link_target = os.path.realpath(immediate_dir)
        if not current_link_target == minute_dir_path:
            raise BadParamTreeException("tried to symlink %s to %s, but symlink already exists and points to %s!" % (immediate_dir, minute_dir_path, current_link_target))


def construct_and_save_noise_models(minute, block_len_seconds, sta, chan, filter_str, srate, order, model_wave = None):

    s = Sigvisa()

    minute_dir, model_fname = model_path(sta, chan, filter_str, srate, order, minute_time=minute)
    if model_wave is None:
        model_wave = fetch_waveform(sta, chan, minute, minute+block_len_seconds, pad_seconds=NOISE_PAD_SECONDS)

    old_band = filter_str_extract_band(filter_str)
    for band in s.bands:
        tmp_filter_str = filter_str.replace(old_band, band)
        filtered_wave = model_wave.filter(tmp_filter_str)

        # train AR noise model
        ar_learner = ARLearner(filtered_wave.data.compressed(), filtered_wave['srate'])
        params, std = ar_learner.yulewalker(order)
        em = ErrorModel(0, std)
        armodel = ARModel(params, em, c = ar_learner.c)

        minute_dir, model_fname = model_path(sta, chan, tmp_filter_str, srate, order, minute_time=minute)
        ensure_dir_exists(minute_dir)
        print "saved model", model_fname
        armodel.dump_to_file(os.path.join(minute_dir, model_fname))

        wave_fname = model_fname.replace("armodel", "wave")
        np.savetxt(os.path.join(minute_dir, wave_fname), filtered_wave.data.filled(np.float('nan')))

    return minute_dir

def get_noise_model(waveform=None, sta=None, chan=None, filter_str=None, time=None, srate=40, order=17, noise_mode=NOISE_MODE):
    """
    Returns an ARModel noise model of the specified order for the
    given station/channel/filter, trained from the hour prior to
    the given time. Results are cached, so each noise model is
    learned only once.

    This method takes *either* a station, channel, filter string,
    and time, *or* a Waveform object from which these can all be
    read.

    """
    if waveform is not None:
        sta = waveform['sta']
        chan = waveform['chan']
        filter_str = waveform['filter_str']
        time = waveform['stime']
        srate = waveform['srate']
    else:
        if sta is None or chan is None or filter_str is None or time is None:
            raise Exception("missing argument to get_noise_model!")

    if noise_mode == NOISE_MODE_HOURLY:
        signal_hour = int(time/3600)
        noise_hour = signal_hour - 1
        model_dir, model_fname = model_path(sta, chan, filter_str, srate, order, hour_time=noise_hour*3600)
    elif noise_mode == NOISE_MODE_IMMEDIATE:
        model_dir, model_fname = model_path(sta, chan, filter_str, srate, order, anchor_time=time)
    else:
        raise Exception("unknown noise_mode (neither HOURLY nor IMMEDIATE)")


    try:
        armodel = load_armodel_from_file(os.path.join(model_dir, model_fname))
    except IOError:

        if noise_mode == NOISE_MODE_HOURLY:
            print "building noise models for hour %d at %s" % (noise_hour, sta)
            construct_and_save_hourly_noise_models(noise_hour, sta, chan, filter_str, srate, order)
        elif noise_mode == NOISE_MODE_IMMEDIATE:
            # block_start and block_end are computed above
            construct_and_save_immediate_noise_models(time, sta, chan, filter_str, srate, order)

    armodel = load_armodel_from_file(os.path.join(model_dir, model_fname))
    return armodel

def block_waveform_exists(sta, chan, stime, etime):
    try:
        model_wave = fetch_waveform(sta, chan, stime, etime, pad_seconds=NOISE_PAD_SECONDS)
    except MissingWaveform as e:
        return False

    return model_wave['fraction_valid'] > 0.6


def get_recent_safe_block(time, sta, chan, margin_seconds = 10, preferred_len_seconds = 120, min_len_seconds = 60, arrival_window_seconds=1200):
    """ Get a block of time preceding the specified time, and ending at least margin_seconds before that time, for which waveform data exists, and during which there are no recorded arrivals at the station."""
    arrival_times = load_arrival_times(sta, time, time - arrival_window_seconds)

    block_end = time - margin_seconds
    block_start = block_end - preferred_len_seconds

    # try a full-length block
    bad_block = arrivals_intersect_block(block_start, block_end, arrival_times)

    if not bad_block and block_waveform_exists(sta, chan, block_start, block_end):
        return block_start, block_end

    # if that didn't work, try a shorter block
    block_start = block_end - min_len_seconds
    bad_block = arrivals_intersect_block(block_start, block_end, arrival_times)

    if not bad_block and block_waveform_exists(sta, chan, block_start, block_end):
        return block_start, block_end

    # if that didn't work, slide backwards until in time until we hit a good block
    bad_block=True
    while bad_block:
        block_start -= 10
        block_end -= 10
        bad_block = arrivals_intersect_block(block_start, block_end, arrival_times)
        if not bad_block:
            bad_block =  not block_waveform_exists(sta, chan, block_start, block_end)

        # if we get all the way back to before we've loaded arrivals, try again with a longer window
        if block_start < time - arrival_window_seconds:
            block_start, block_end = get_recent_safe_block(time, sta, chan, margin_seconds = margin_seconds, preferred_len_seconds = preferred_len_seconds, min_len_seconds = min_len_seconds, arrival_window_seconds=arrival_window_seconds*2)
            break

    return block_start, block_end

def set_noise_process(wave):
    s = Sigvisa()
    arm = get_noise_model(waveform=wave)
    c = sigvisa_c.canonical_channel_num(wave['chan'])
    b = sigvisa_c.canonical_band_num(wave['band'])
    s.sigmodel.set_noise_process(wave['siteid'], b, c, arm.c, arm.em.std**2, np.array(arm.params))

def main():
    print "tests are now in test.py"



if __name__ == "__main__":
    main()
