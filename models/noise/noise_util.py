import numpy as np
import numpy.ma as ma
import time
import os

from sigvisa import Sigvisa, BadParamTreeException
from obspy.signal.trigger import zDetect, recSTALTA

from sigvisa.database import dataset
from sigvisa.database.signal_data import ensure_dir_exists

from sigvisa.models.noise.noise_model import NoiseModel
from sigvisa.models.noise.armodel.model import ARModel, ErrorModel
from sigvisa.models.noise.armodel.learner import ARLearner
from sigvisa.models.noise.iid import L1IIDModel, train_l1_model


from sigvisa.signals.io import fetch_waveform, MissingWaveform
from sigvisa.signals.common import filter_str_extract_band, filter_str_extract_smoothing






import hashlib


NOISE_PAD_SECONDS = 20




class NoNoiseException(Exception):
    pass


def get_sta_lta_picks(wave):
    df = wave['srate']
    s = Sigvisa()

    fw = wave.filter('freq_0.5_5.0')
    stalta = recSTALTA(
        fw.data[df * NOISE_PAD_SECONDS + df:-df * NOISE_PAD_SECONDS - df].filled(float('nan')), int(5 * df), int(10 * df))[10 * df:]
# stalta =
# classicSTALTAPy(fw.data[df*NOISE_PAD_SECONDS:-df*NOISE_PAD_SECONDS].filled(float('nan')),
# int(5 * df), int(20 * df))[20*df:]
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


def model_path(sta, chan, filter_str, srate, order, window_stime, model_type):
    t = time.gmtime(window_stime)
    model_dir = "%d" % int(window_stime)
    base_dir = os.path.join(
        "parameters", "noise_models", sta, str(t.tm_year), str(t.tm_mon), str(t.tm_mday))
    sanitized_filter_str = '_'.join([s for s in filter_str.split(';') if s != ""])
    model_fname = '.'.join([chan, sanitized_filter_str, "%.0f" % (srate) + "hz", str(order), '%smodel' % model_type])

    return os.path.join(base_dir, model_dir, model_fname)


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


def arrivals_intersect_block(block_start, block_end, arrival_times, danger_period_seconds=400):
    for t in arrival_times:
        if t > block_start and t < block_end:
            return True
        buffer = block_start - t
        if buffer > 0 and buffer < danger_period_seconds:
            return True
    return False

def construct_and_save_noise_models(hour, dbconn, window_stime, window_len, sta, chan, filter_str, srate, env, order, model_wave=None, model_type="ar", save_models=True):
    s = Sigvisa()

    if model_wave is None:
        model_wave = fetch_waveform(sta, chan, window_stime, window_stime + window_len, pad_seconds=NOISE_PAD_SECONDS)

    smooth = filter_str_extract_smoothing(filter_str)
    requested_band = filter_str_extract_band(filter_str)
    bandlist = set((requested_band,))
    if save_models:
        bandlist = bandlist.union(s.bands)

    for band in bandlist:
        tmp_filter_str = filter_str.replace(requested_band, band)
        filtered_wave = model_wave.filter(tmp_filter_str)

        # train AR noise model
        if model_type == "ar":
            ar_learner = ARLearner(filtered_wave.data.compressed(), sf=srate)
            params, std = ar_learner.yulewalker(order)
            em = ErrorModel(0, std)
            model = ARModel(params, em, c=ar_learner.c, sf=srate)
        elif model_type == "l1":
            model = train_l1_model(filtered_wave.data.compressed())
        else:
            raise Exception('unrecognized noise model type %s!' % model_type)

        if save_models:
            nm_fname = model_path(sta, chan, tmp_filter_str, srate, order, window_stime=window_stime, model_type=model_type)
            full_fname = os.path.join(os.getenv('SIGVISA_HOME'), nm_fname)
            ensure_dir_exists(os.path.dirname(full_fname))
            model.dump_to_file(full_fname)

            nmid = model.save_to_db(dbconn=dbconn, sta=sta, chan=chan,
                                    band=band, hz=srate, env=env, smooth=smooth,
                                    window_stime=window_stime, window_len=window_len,
                                    fname=nm_fname, hour=hour)

        if band == requested_band:
            requested_model = model

            if save_models:
                requested_nmid = nmid
                requested_fname = nm_fname

    if save_models:
        return requested_model, requested_nmid, requested_fname
    else:
        return requested_model

def get_noise_model(waveform=None, sta=None, chan=None, filter_str=None, time=None, env=True, srate=40, order="auto", return_details=False, model_type="ar", force_train=False):
    """
    Returns an ARModel noise model of the specified order for the
    given station/channel/filter, trained from the hour prior to
    the given time. Results are cached, so each noise model is
    learned only once.

    This method takes *either* a station, channel, filter string,
    and time, *or* a Waveform object from which these can all be
    read.

    """

    # without loading/saving to the DB, there are no details to return
    assert(not (return_details and force_train))

    if waveform is not None:
        sta = waveform['sta']
        chan = waveform['chan']
        filter_str = waveform['filter_str']
        time = waveform['stime']
        srate = waveform['srate']
        env = "env" in filter_str
    else:
        if sta is None or chan is None or filter_str is None or time is None or env is None:
            raise Exception("missing argument to get_noise_model!")

    if order=="auto":
        if model_type == "ar":
            order=int(np.ceil(srate/2))
        elif model_type == "l1":
            order = 0

    band = filter_str_extract_band(filter_str)
    smooth = filter_str_extract_smoothing(filter_str)
    s = Sigvisa()

    hour = int(time) / 3600

    # first see if we already have an applicable noise model
    model = None
    if not force_train:
        model, details = NoiseModel.load_from_db(dbconn=s.dbconn, sta=sta, chan=chan, band=band, smooth=smooth, hz=srate, env=env, order=order, model_type=model_type, hour=hour, return_extra=True)

    if model is None:
        # otherwise, train a new model
        window_stime, window_etime = get_recent_safe_block(time, sta, chan)
        window_len = window_etime - window_stime
        results = construct_and_save_noise_models(window_stime=window_stime, window_len=window_len, sta=sta, chan=chan, env=env, filter_str=filter_str, srate=srate, order=order, model_type=model_type, hour=hour, dbconn=s.dbconn, save_models=not force_train)
        if not force_train:
            model, nmid, nm_fname = results
        else:
            model = results
            nmid = None
            nm_fname = None

        # and store a record of it in the DB
        s.dbconn.commit()
    else:
        nmid, window_stime, window_len, mean, std, nm_fname = details

    if return_details:
        return model, nmid, nm_fname
    else:
        return model


def block_waveform_exists(sta, chan, stime, etime):
    try:
        model_wave = fetch_waveform(sta, chan, stime, etime, pad_seconds=NOISE_PAD_SECONDS)
    except MissingWaveform as e:
        return False

    return model_wave['fraction_valid'] > 0.4

def get_recent_safe_block(time, sta, chan, margin_seconds=10, preferred_len_seconds=120, min_len_seconds=60, arrival_window_seconds=1200):
    """ Get a block of time preceding the specified time, and ending at least margin_seconds before that time, for which waveform data exists, and during which there are no recorded arrivals at the station."""

    print "getting safe block at", sta, chan, "in", time - arrival_window_seconds, time
    arrival_times = load_arrival_times(sta, time - arrival_window_seconds, time)

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
    bad_block = True
    while bad_block:
        block_start -= 10
        block_end -= 10
        bad_block = arrivals_intersect_block(block_start, block_end, arrival_times)

        # if the block is otherwise fine, make sure the waveform exists
        if not bad_block:
            bad_block = not block_waveform_exists(sta, chan, block_start, block_end)

            # if bad_block:
            #    print "block", block_start, block_end, "was otherwise fine but no waveform :-("
            # else:
            #    print "found good block!", block_start, block_end
        # if we get all the way back to before we've loaded arrivals, try again with a longer window
        if block_start < time - arrival_window_seconds:
            block_start, block_end = get_recent_safe_block(
                time - arrival_window_seconds, sta, chan, margin_seconds=margin_seconds, preferred_len_seconds=preferred_len_seconds,
                min_len_seconds=min_len_seconds, arrival_window_seconds=arrival_window_seconds * 2)
            break

    return block_start, block_end


def main():
    print "tests are now in test.py"


if __name__ == "__main__":
    main()
