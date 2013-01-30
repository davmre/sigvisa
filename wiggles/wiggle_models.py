from noise.noise_model import get_noise_model

class WiggleModel(object):

    def __init__(self, tm):
        self.tm = tm

    def template_cost(wave, phases, params):
        raise Exception("not implemented")

    def summary_str(self):
        return "base"


class StupidL1WiggleModel(WiggleModel):

    def template_cost(self, wave, phases, params):
        return self.tm.waveform_log_likelihood_iid(wave, (phases, params))

    def summary_str(self):
        return "stupidiid"


class PlainWiggleModel(WiggleModel):

    """

    Model signal as AR noise + plain template (no wiggles)

    """

    def template_cost(self, wave, phases, params):
        generated = self.tm.generate_template_waveform((phases, params), model_waveform=wave)
        nm = get_noise_model(wave)
        return nm.lklhood(data = (wave.data - generated.data), zero_mean=True)

    def summary_str(self):
        return "plain"
