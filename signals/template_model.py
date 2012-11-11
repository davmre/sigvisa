import numpy as np
import sys, os
from sigvisa import *

class TemplateModel(object):
    """
    Abstract class defining a signal template model.

    A phase template is defined by some number of parameters
    (e.g. onset period, height, and decay rate for a
    paired-exponential template). A signal consists of a number of
    phase arrivals.

    The methods in the class deal with matrices, each row of which
    gives the parameters for a specific phase arrival. That is, we
    allow modeling the joint distribution of template parameters over
    multiple phases, though it's also possible for a particular
    implementation to treat them independently.

    Currently we assume that each channel and frequency band are
    independent. This should probably change.

    """

    # return the name of the template model as a string
    def model_name(self):
        raise Exception("abstract class: method not implemented")

    # return a tuple of strings representing parameter names
    def params(self):
        raise Exception("abstract class: method not implemented")

    def param_str(self, template_params):
        phases, vals= template_params
        pstr = ""
        for (i, phase) in enumerate(phases):
            pstr += "%s: " % phase
            for (j, param) in enumerate(self.params()):
                pstr += "%s: %.3f " % (param, vals[i,j])
            pstr += "\n"
        return pstr

    def generate_template_waveform(self, template_params, model_waveform=None, logscale=False, sample=False):
        raise Exception("abstract class: method not implemented")


    # p(waveform | params)
    def waveform_log_likelihood(self, wave, template_params):
        raise Exception("abstract class: method not implemented")

    # this is intended as a simple default option that gives decent,
    # fast fits.  since we don't actually learn the variance of the L1
    # noise, the interpretation as a "likelihood" is basically
    # meaningless.
    def waveform_log_likelihood_iid(self, wave, template_params):
        (phases, vals) = template_params
        if (vals < self.low_bounds(phases)).any() or (vals > self.high_bounds(phases)).any():
            return -np.inf

        env = self.generate_template_waveform(template_params, model_waveform=wave, logscale=True)
        return -wave.l1_cost(env)

    def low_bounds(self, phases):
        raise Exception("abstract class: method not implemented")

    def high_bounds(self, phases):
        raise Exception("abstract class: method not implemented")

    def __init__(self, run_name, model_type = "gp_dad"):
        self.sigvisa = Sigvisa()

        # load models
        self.models = NestedDict()

        if model_type == "dummy":
            self.dummy=True
            return
        else:
            self.dummy=False

        for param in self.params():
            if param == "arrival_time":
                continue

            basedir = os.path.join("parameters", self.model_name(), model_type, param)
            print basedir
            for sta in os.listdir(basedir):
                sta_dir = os.path.join(basedir, sta)
                if not os.path.isdir(sta_dir):
                    continue
                for phase in os.listdir(sta_dir):
                    phase_dir = os.path.join(sta_dir, phase)
                    if not os.path.isdir(phase_dir):
                        continue
                    for chan in os.listdir(phase_dir):
                        chan_dir = os.path.join(phase_dir, chan)
                        if not os.path.isdir(chan_dir):
                            continue
                        for band_model in os.listdir(chan_dir):
                            band_file = os.path.join(chan_dir, band_model)
                            band_run, ext = os.path.splitext(band_model)
                            band, run = os.path.splitext(band_run)

                            if run == run_name:
                                self.models[param][sta][phase][chan][band] = SpatialModel(fname=band_file)

    def predictTemplate(self, event, sta, chan, band, phases=None):
        if phases is None:
            phases = Sigvisa().arriving_phases(event, sta)

        params = self.params()

        predictions = np.zeros((len(phases),  len(params)))
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(params):
                model = self.models[param][sta][phase][chan][band]
                if isinstance(model, NestedDict):
                    raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                if param == "amplitude":
                    source_logamp = event.source_logamp(band)
                    predictions[i,j] = source_logamp + model.predict(event)
                elif param == "arrival_time":
                    predictions[i,j] = event.time + self.travel_time(event, sta)
                else:
                    predictions[i,j] = model.predict(event)
        return (phases, predictions)


    def sample(self, event, sta, chan, band, phases=None):

        if phases is None:
            phases = Sigvisa().arriving_phases(event, sta)

        params = self.params()

        samples = np.zeros((len(phases),  len(params)))
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(params):
                model = self.models[param][sta][phase][chan][band]
                if isinstance(model, NestedDict):
                    raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                if param == "amplitude":
                    source_logamp = event.source_logamp(band)
                    samples[i,j] =  source_logamp + model.predict(event)
                elif param == "arrival_time":
                    samples = event.time + self.sample_travel_time(event, sta)
                else:
                    samples[i,j] = model.sample(event)
        return (phases, predictions)

    def log_likelihood(self, template_params, event, sta, chan, band):

        if self.dummy:
            return 1.0

        phases = template_params[0]
        param_vals = template_params[1]

        log_likelihood = 0
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(self.params()):
                model = self.models[param][sta][phase][chan][band]
                if isinstance(model, NestedDict):
                    raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                if param == "amplitude":
                    source_logamp = event.source_logamp(band)
                    log_likelihood += model.log_likelihood(event, param_vals[i,j] - source_logamp)
                elif param == "arrival_time":
                    log_likelihood = self.travel_time_log_likelihood(event, sta, param_vals[i,j])
                else:
                    log_likelihood += model.log_likelihood(event, param_vals[i,j])

        return log_likelihood

    def travel_time(self, event, sta, phase):
        siteid = self.sigvisa.siteids[sta]
        phaseid = self.sigvisa.phaseids[phase]
        meantt = self.sigvisa.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid-1, phaseid-1)
        return meantt

    def sample_travel_time(self, event, sta, phase):
        meantt = self.mean_travel_time(event, sta, phase)

        # peak of a laplace distribution is 1/2b, where b is the
        # scale param, so (HACK ALERT) we can recover b by
        # evaluating the density at the peak
        siteid = sigvisa.siteids[sta]
        phaseid = sigvisa.phaseids[phase]
        ttscale = 2.0 / np.exp(self.sigvisa.sigmodel.arrtime_logprob(0, 0, 0, siteid-1, phaseid-1))

        # sample from a Laplace distribution:
        U = np.random.random() - .5
        tt = meantt - ttscale * np.sign(U) * np.log(1 - 2*np.abs(U))
        return tt

    def travel_time_log_likelihood(self, tt, event, sta, phase):
        meantt = self.mean_travel_time(event, sta, phase)
        siteid = sigvisa.siteids[sta]
        phaseid = sigvisa.phaseids[phase]
        ll = self.sigvisa.arrtime_logprob(tt, meantt, 0, siteid-1, phaseid-1)
        return ll



