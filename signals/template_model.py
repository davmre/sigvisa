import numpy as np
import sys, os
from sigvisa import *

import noise.noise_model as noise_model
from learn.optimize import BoundsViolation
from learn.train_coda_models import load_model
from signals.common import *


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

    def generate_template_waveform(self, template_params, model_waveform=None, sample=False):
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

        lb = np.nan_to_num(self.low_bounds(phases))
        hb = np.nan_to_num(self.high_bounds(phases))
        bounds_violations = np.abs((lb-vals) * (vals < lb)) + np.abs((vals-hb) * (vals > hb))       
        bound_penalty = 1000*np.exp(min(700,np.sum(bounds_violations))) - 1000
#        if (vals < lb).any() or (vals > hb).any():
#            raise BoundsViolation("params: %s\n\n low bounds: %s\n\n high bounds: %s" % (str(vals), str(lb), str(hb)))

        # assume the provided wave is in original (linear) scale, but we do the comparison in logscale
        env = self.generate_template_waveform(template_params, model_waveform=wave)
        logwave =  wave.filter("log")
        logenv = env.filter("log")
        cost = logwave.l1_cost(logenv)

#        env = self.generate_template_waveform(template_params, model_waveform=wave, logscale=False)
#        cost = wave.l1_cost(env)

        a = -cost - bound_penalty
        return a

    def low_bounds(self, phases):
        raise Exception("abstract class: method not implemented")

    def high_bounds(self, phases):
        raise Exception("abstract class: method not implemented")

    def __init__(self, run_name, run_iter, model_type = "gp_dad"):
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

            basedir = os.path.join("parameters", "runs", run_name, "iter_%02d" % run_iter, self.model_name(), param)
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
                        for band in os.listdir(chan_dir):
                            band_dir = os.path.join(chan_dir, band)
                            for fname in os.listdir(band_dir):

                                fullname = os.path.join(band_dir, fname)
                                evidhash, ext = os.path.splitext(fname)

                                if ext != model_type:
                                    continue

                                self.models[param][sta][phase][chan][band] = load_model(fullname)


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
        siteid = self.sigvisa.name_to_siteid_minus1[sta] + 1
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


    def generate_trace_python(self, model_waveform, template_params):
        nm = noise_model.get_noise_model(model_waveform)

        srate = model_waveform['srate']
        st = model_waveform['stime']
        et = model_waveform['etime']
        npts = model_waveform['npts']
        
        data = np.ones((npts,)) * nm.c
        phases, vals = template_params


        for (i, phase) in enumerate(phases):
            v = vals[i,:]
            arr_time = v[0]
            start = (arr_time - st) * srate
            start_idx = int(start)
            offset = start - start_idx
            phase_env = self.abstract_logenv_raw(v, idx_offset = offset, srate=srate)
            end_idx = start_idx + len(phase_env)

            try:
                overshoot = max(0, end_idx - len(data))
                data[start_idx:end_idx-overshoot] += np.exp(phase_env[:len(phase_env)-overshoot])
            except Exception as e:
                print e
                raise
        return data

    def generate_template_waveform(self, template_params, model_waveform, sample=False):
        s = self.sigvisa

        siteid = model_waveform['siteid']
        srate = model_waveform['srate']
        st = model_waveform['stime']
        et = model_waveform['etime']
        c = sigvisa_c.canonical_channel_num(model_waveform['chan'])
        b = sigvisa_c.canonical_band_num(model_waveform['band'])

        phases, vals = template_params
        phaseids = [s.phaseids[phase] for phase in phases]



        if not sample:
            env = self.generate_trace_python(model_waveform, template_params) #env = s.sigmodel.generate_trace(st, et, int(siteid), int(b), int(c), srate, phaseids, vals)
        else:
            env = s.sigmodel.sample_trace(st, et, int(siteid), int(b), int(c), srate, phaseids, vals)


        if len(env) == len(model_waveform.data)-1:
            le = len(env)
            new_env = np.ones(le+1)
            new_env[0:le] = env
            new_env[le] = env[-1]
            env = new_env
        assert len(env) == len(model_waveform.data)

        wave = Waveform(data = env, segment_stats=model_waveform.segment_stats.copy(), my_stats=model_waveform.my_stats.copy())

        try:
            del wave.segment_stats['evid']
            del wave.segment_stats['event_arrivals']
        except KeyError:
            pass
        return wave
