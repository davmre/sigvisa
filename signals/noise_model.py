import numpy as np
import time, os

from sigvisa import Sigvisa
import sigvisa_c
from database import dataset
from database.signal_data import ensure_dir_exists

from signals.armodel.model import ARModel, ErrorModel, load_armodel_from_file
from signals.armodel.learner import ARLearner

from signals.io import fetch_waveform
from signals.common import filter_str_extract_band

NOISE_PAD_SECONDS = 20

def model_path(sta, chan, filter_str, srate, order, hour_time=None, minute_time=None):

    if hour_time is not None:
        t = time.gmtime(hour_time)
        model_dir = "hour_%02d" % t.tm_hour
    elif minute_time is not None:
        t = time.gmtime(minute_time)
        model_dir = "%d" % int(minute_time)
    else:
        raise Exception("noise model path must specify either an hour or a specific minute")

    base_dir = os.path.join("parameters/noise_models/", sta, str(t.tm_year), str(t.tm_mon), str(t.tm_mday))
    sanitized_filter_str = '_'.join([s for s in filter_str.split(';') if s != ""])
    model_fname = '.'.join([chan, sanitized_filter_str, "%.0f" % (srate) + "hz", str(order), 'armodel'])

    return os.path.join(base_dir, model_dir), model_fname


def get_median_noise_wave(sta, chan, hour_start, hour_end):

    s = Sigvisa()

    arrivals = dataset.read_station_detections(s.cursor, sta, hour_start, hour_end)
    if arrivals.shape[0] != 0:
        arrival_times = arrivals[:, dataset.DET_TIME_COL]
    else:
        arrival_times = np.array(())


    # load waveforms for five minutes within the hour
    waves = []
    minutes = []
    failures = []
    max_failures=30
    while len(waves) < 5 and len(failures) < max_failures:

        while True:
            minute = np.random.randint(60)*60+hour_start
            if minute not in minutes and minute not in failures:
                break

        # skip any minute that includes a detected arrival
        for t in arrival_times:
            if t > minute and t < minute+60:
                continue
        try:
            wave = fetch_waveform(sta, chan, minute, minute+60, pad_seconds=NOISE_PAD_SECONDS)
        except Exception as e:
            failures.append(minute)
            print "failed loading signal (%s, %s, %d, %d)." % (sta, chan, minute, minute+60)
            continue

        # also skip any minute for which we don't have much data
        if wave['fraction_valid'] < 0.5:
            failures.append(minute)
            print "not enough datapoints for signal (%s, %s, %d, %d) (%.1f\% valid)." % (sta, chan, minute, minute+60, wave['fraction_valid'])
            continue

        waves.append(wave)
        minutes.append(minute)

    if failures == max_failures:
        raise Exception("failed to load noise model for (%s, %s, %d)" % (sta, chan, hour_start))

    # choose the median minute (index 2 of 0,1,2,3,4) as our model
    waves.sort(key=lambda w : np.dot(w.data, w.data))
    model_wave = waves[2]
    minute = minutes[2]

    return model_wave, minute

def construct_and_save_hourly_noise_models(hour, sta, chan, filter_str, srate, order):

    hour_start = hour*3600
    hour_end = (hour+1)*3600

    s = Sigvisa()
    # wave_fname = "%s.%s.%.0f.wave" % (sta, chan, srate)

    # if we've built a noise model here before, for a different channel or filter band, reload the same training segment
    hour_dir, model_fname = model_path(sta, chan, filter_str, srate, order, hour_time=hour_start)
    if os.path.exists(hour_dir):
        minute_dir = os.path.realpath(hour_dir)
        minute = int(os.path.split(minute_dir)[-1])
        model_wave = fetch_waveform(sta, chan, minute, minute+60, pad_seconds=NOISE_PAD_SECONDS)

    # otherwise, load a training segment from scratch
    else:
        print "hour dir", hour_dir, "doesn't exist, computing new median minute"
        model_wave, minute = get_median_noise_wave(sta, chan, hour_start, hour_end)

    old_band = filter_str_extract_band(filter_str)

    for band in s.bands:
        tmp_filter_str = filter_str.replace(old_band, band)
        filtered_wave = model_wave.filter(tmp_filter_str)

        # train AR noise model
        ar_learner = ARLearner(filtered_wave.data, filtered_wave['srate'])
        params, std = ar_learner.yulewalker(order)
        em = ErrorModel(0, std)
        armodel = ARModel(params, em, c = ar_learner.c)

        minute_dir, model_fname = model_path(sta, chan, tmp_filter_str, srate, order, minute_time=minute)
        ensure_dir_exists(minute_dir)
        print "saved model", model_fname
        armodel.dump_to_file(os.path.join(minute_dir, model_fname))

        wave_fname = model_fname.replace("armodel", "wave")
        np.savetxt(os.path.join(minute_dir, wave_fname), filtered_wave.data.filled(np.float('nan')))

    try:
        minute_dir_path = os.path.realpath(minute_dir)
        os.symlink(minute_dir_path, hour_dir)
        print "successfully created symlink"
    except OSError:
        # if symlink already exists, check to make sure it's the right thing
        current_link_target = os.path.realpath(hour_dir)

        if not current_link_target == minute_dir_path:
            raise Exception("tried to symlink %s to %s, but symlink already exists and points to %s!" % (hour_dir, minute_dir_path, current_link_target))

def get_noise_model(waveform=None, sta=None, chan=None, filter_str=None, time=None, srate=40, order=17):
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

    signal_hour = int(time/3600)
    noise_hour = signal_hour - 1

    model_dir, model_fname = model_path(sta, chan, filter_str, srate, order, hour_time=noise_hour*3600)
    try:
        armodel = load_armodel_from_file(os.path.join(model_dir, model_fname))
    except IOError:
        print "building noise models for hour %d at %s" % (noise_hour, sta)
        construct_and_save_hourly_noise_models(noise_hour, sta, chan, filter_str, srate, order)
        armodel = load_armodel_from_file(os.path.join(model_dir, model_fname))

    return armodel

def set_noise_process(wave):
    s = Sigvisa()
    arm = get_noise_model(waveform=wave)
    c = sigvisa_c.canonical_channel_num(wave['chan'])
    b = sigvisa_c.canonical_band_num(wave['band'])
    s.sigmodel.set_noise_process(wave['siteid'], b, c, arm.c, arm.em.std**2, np.array(arm.params))
