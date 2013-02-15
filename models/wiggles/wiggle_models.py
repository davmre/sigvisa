import numpy as np
from sigvisa.models.noise.noise_model import get_noise_model


def wiggle_model_by_name(name, **kwargs):
    if name == "base":
        return WiggleModel(**kwargs)
    elif name == "stupidiid":
        return StupidL1WiggleModel(**kwargs)
    elif name == "plain":
        return PlainWiggleModel(**kwargs)
    elif name.startswith("sampling_"):
        featurizer_name = name.split("_")[1:]
        featurizer = featurizer_by_name(featurizer_name)
        return SamplingWiggleModel(featurizer=featurizer, **kwargs)

class WiggleModel(object):

    def __init__(self, tm):
        self.tm = tm

    def template_ncost(wave, phases, params):
        raise Exception("not implemented")

    def summary_str(self):
        return "base"


class SamplingWiggleModel(WiggleModel):

    def __init__(self, tm, featurizer):
        super(SamplingWiggleModel, self).__init__(self, tm)
        self.featurizer = featurizer

    def template_ncost(self, wave, phases, params):
        pass

    def summary_str(self):
        return "sampling_" + self.featurizer.summary_str()


class StupidL1WiggleModel(WiggleModel):

    def template_ncost(self, wave, phases, params):
        generated = self.tm.generate_template_waveform((phases, params), model_waveform=wave)
        diff = (wave.data - generated.data)
        return np.sum(np.abs(diff))

    def summary_str(self):
        return "stupidiid"


class PlainWiggleModel(WiggleModel):

    """

    Model signal as AR noise + plain template (no wiggles)

    """

    def template_ncost(self, wave, phases, params):
        generated = self.tm.generate_template_waveform((phases, params), model_waveform=wave)
        nm = get_noise_model(wave)
        ll = nm.lklhood(data=(wave.data - generated.data), zero_mean=True)
        return ll

    def summary_str(self):
        return "plain"
