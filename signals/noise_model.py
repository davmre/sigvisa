import numpy as np
import numpy.ma as ma
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


def get_masked_hour_wave(sta, chan, hour_start, hour_end):
    s = Sigvisa()

    MASK_SECONDS_BEFORE_ARRIVAL = 10
    MASK_SECONDS_AFTER_ARRIVAL = 400

    arrivals = dataset.read_station_detections(s.cursor, sta, hour_start, hour_end)
    if arrivals.shape[0] != 0:
        arrival_times = arrivals[:, dataset.DET_TIME_COL]
    else:
        arrival_times = np.array(())

    wave = fetch_waveform(sta, chan, hour_start, hour_end)

    for arrival_time in arrival_times:
        srate = wave['srate']
        stime = wave['stime']
        mask_start_idx = max(0, int((arrival_time - MASK_SECONDS_BEFORE_ARRIVAL -stime)*srate))
        mask_end_idx = min(wave['npts'], int((arrival_time + MASK_SECONDS_AFTER_ARRIVAL  -stime)*srate))
        wave.data[mask_start_idx:mask_end_idx] = ma.masked

    return wave

def construct_and_save_hourly_noise_models(hour, sta, chan, filter_str, srate, order):

    hour_start = hour*3600
    hour_end = (hour+1)*3600

    s = Sigvisa()
    # wave_fname = "%s.%s.%.0f.wave" % (sta, chan, srate)

    # if we've built a noise model here before, for a different channel or filter band, reload the same training segment
    hour_dir, model_fname = model_path(sta, chan, filter_str, srate, order, hour_time=hour_start)
    ensure_dir_exists(hour_dir)

    model_wave = get_masked_hour_wave(sta, chan, hour_start, hour_end)

    old_band = filter_str_extract_band(filter_str)

    for band in s.bands:
        tmp_filter_str = filter_str.replace(old_band, band)
        filtered_wave = model_wave.filter(tmp_filter_str)

        # train AR noise model
        ar_learner = ARLearner(filtered_wave.data, filtered_wave['srate'])
        params, std = ar_learner.yulewalker(order)
        em = ErrorModel(0, std)
        armodel = ARModel(params, em, c = ar_learner.c)

        hour_dir, model_fname = model_path(sta, chan, tmp_filter_str, srate, order, hour_time=hour_start)

        print "saved model", model_fname
        armodel.dump_to_file(os.path.join(hour_dir, model_fname))

        wave_fname = model_fname.replace("armodel", "wave")
        np.savetxt(os.path.join(hour_dir, wave_fname), filtered_wave.data.filled(np.float('nan')))


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

def main():

    t_start = 1238917955 - 6*3600
    t_max = t_start + 14*24*3600

    t=t_start
    f = open('means.txt', 'w')
    fs = open('stds.txt', 'w')
    for d in range(14):
        for h in range(24):
            t = t_start + d*24*3600 + h*3600
            try:
                nm = get_noise_model(sta="URZ", chan="BHZ", time=t, filter_str="freq_2.0_3.0;env")
                print "nm for time", t," has mean", nm.c
                f.write("%f, " % nm.c)
                fs.write("%f, " % nm.em.std)
            except Exception as e:
                print e
                f.write(", ")
                fs.write(", ")
                continue

        f.write("\n")
        fs.write("\n")
    f.close()
    fs.close()



if __name__ == "__main__":
    main()
