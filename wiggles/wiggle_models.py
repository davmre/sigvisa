from noise.noise_model import get_noise_model


def wiggle_model_by_name(name, **kwargs):
    if name == "base":
        return WiggleModel(**kwargs)
    elif name == "stupidiid":
        return StupidL1WiggleModel(**kwargs)
    elif name == "plain":
        return PlainWiggleModel(**kwargs)

class WiggleModel(object):

    def __init__(self, tm):
        self.tm = tm

    def template_ncost(wave, phases, params):
        raise Exception("not implemented")

    def summary_str(self):
        return "base"


class StupidL1WiggleModel(WiggleModel):

    def template_ncost(self, wave, phases, params):
        return self.tm.waveform_log_likelihood_iid(wave, (phases, params))

    def summary_str(self):
        return "stupidiid"


class PlainWiggleModel(WiggleModel):

    """

    Model signal as AR noise + plain template (no wiggles)

    """

    def template_ncost(self, wave, phases, params):
        generated = self.tm.generate_template_waveform((phases, params), model_waveform=wave)
        nm = get_noise_model(wave)
        ll = nm.lklhood(data = (wave.data - generated.data), zero_mean=True)
        return ll

    def summary_str(self):
        return "plain"
