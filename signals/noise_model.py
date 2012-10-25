import numpy as np

from sigvisa import Sigvisa

from database import dataset

from signals.armodel.model import ARModel, ErrorModel
from signals.armodel.learner import ARLearner

from signals.io import fetch_waveform

def get_noise_model(waveform=None, sta=None, chan=None, filter_str=None, time=None, order=17):
    """
    Returns an ARModel noise model of the specified order for the
    given station/channel/filter, trained from the hour prior to
    the given time. Results are cached, so each noise model is
    learned only once.

    This method takes *either* a station, channel, filter string,
    and time, *or* a Waveform object from which these can all be
    read.

    """

    s = Sigvisa()

    if waveform is not None:
        sta = waveform['sta']
        chan = waveform['chan']
        filter_str = waveform['filter_str']
        time = waveform['stime']
    else:
        if sta is None or chan is None or filter_str is None or time is None:
            raise Exception("missing argument to get_noise_model!")


    hour = int(time/3600)

    # check to see if we have this model in the cache
    armodel = s.noise_models[sta][chan][filter_str][order][hour]
    if isinstance(armodel, ARModel):
        return armodel

    noise_hour_start = (hour-1)*3600
    noise_hour_end = hour*3600

    arrivals = dataset.read_station_detections(s.cursor, sta, noise_hour_start, noise_hour_end)
    arrival_times = arrivals[:, dataset.DET_TIME_COL]

    # load waveforms for five minutes within the previous hour
    waves = []
    while len(waves) < 5:
        minute = np.random.randint(60)*60+noise_hour_start

        # skip any minute that includes a detected arrival
        for t in arrival_times:
            if t > minute and t < minute+60:
                continue

        try:
            wave = fetch_waveform(sta, chan, minute, minute+60)
        except:
            continue

        # also skip any minute for which we don't have much data
        if wave['fraction_valid'] < 0.5:
            continue

        waves.append(wave)

    # choose the median minute as our model
    waves.sort(key=lambda w : np.mean(w.data))
    model_wave = waves[2].filter(filter_str)

    # train AR noise model
    ar_learner = ARLearner(model_wave.data, model_wave['srate'])
    params, std = ar_learner.yulewalker(order)
    em = ErrorModel(0, std)
    armodel = ARModel(params, em, c = ar_learner.c)
    s.noise_models[sta][chan][filter_str][order][hour] = armodel
    return armodel
