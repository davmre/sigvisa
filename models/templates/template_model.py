import numpy as np
import sys, os
from sigvisa import *


# from
from sigvisa.database.signal_data import get_fitting_runid
import sigvisa.models.noise.noise_model as noise_model
from sigvisa.infer.optimize.optim_utils import BoundsViolation
from sigvisa.learn.train_coda_models import load_model
from sigvisa.signals.common import *


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

    def low_bounds(self, phases):
        raise Exception("abstract class: method not implemented")

    def high_bounds(self, phases):
        raise Exception("abstract class: method not implemented")

    def __init__(self, run_name=None, run_iter=None, model_type = None, verbose=True, sites=None, modelids=None):
        s = Sigvisa()
        cursor = s.dbconn.cursor()

        # load models

        self.models = NestedDict()

        if model_type == "dummy":
            self.dummy=True
            return
        else:
            self.dummy=False
            basedir = os.getenv('SIGVISA_HOME')

            if modelids is not None:
                models = []
                for modelid in modelids:
                    sql_query = "select param, site, phase, chan, band, model_type, model_fname, modelid from sigvisa_template_param_model where modelid=%d" % (modelid)
                    cursor.execute(sql_query)
                    models.append(cursor.fetchone())
            elif run_name is not None and run_iter is not None and model_type is not None:
                runid = get_fitting_runid(cursor, run_name=run_name, iteration=run_iter, create_if_new=False)

                if isinstance(model_type, str):
                    model_type_cond = "model_type = '%s'" % model_type
                elif isinstance(model_type, dict):
                    model_type_cond = "(" + " or ".join(["(model_type = '%s' and param = '%s')" % (v, k) for (k, v) in model_type.items()]) + ")"
                else:
                    raise Exception("model_type must be either a string, or a dict of param->model_type mappings")

                if sites is not None:
                    site_cond = " and (" + " or ".join(["site='%s'" % site for site in sites]) + ")"
                else:
                    site_cond = ""

                sql_query = "select param, site, phase, chan, band, model_type, model_fname, modelid from sigvisa_template_param_model where %s %s and fitting_runid=%d" % (model_type_cond, site_cond, runid)
                cursor.execute(sql_query)
                models = cursor.fetchall()
            else:
                raise Exception("you must specify either a fitting run or a list of template model ids!")

            for (param, sta, phase, chan, band, db_model_type, fname, modelid) in models:
                if param == "amp_transfer":
                    param = "coda_height"
                self.models[param][sta][phase][chan][band] = load_model(os.path.join(basedir, fname), db_model_type)
                self.models[param][sta][phase][chan][band].modelid = modelid
                if verbose:
                    print "loaded model type '%s' for param %s at %s:%s:%s phase %s (modelid %d)" % ( db_model_type, param, sta, chan, band, phase, modelid)

    def predictTemplate(self, event, sta, chan, band, phases):

        params = self.params()
        predictions = np.zeros((len(phases),  len(params)))
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(params):
                if param == "arrival_time":
                    predictions[i,j] = event.time + self.travel_time(event, sta, phase)
                else:
                    model = self.models[param][sta][phase][chan][band]
                    if isinstance(model, NestedDict):
                        raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                    if param == "coda_height":
                        source_logamp = event.source_logamp(band, phase)
                        predictions[i,j] = source_logamp + model.predict(event)
                    else:
                        predictions[i,j] = model.predict(event)
        return predictions


    def sample(self, event, sta, chan, band, phases):

        params = self.params()

        samples = np.zeros((len(phases),  len(params)))
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(params):

                if param == "arrival_time":
                    samples[i,j] = event.time + self.sample_travel_time(event, sta, phase)
                else:
                    model = self.models[param][sta][phase][chan][band]
                    if isinstance(model, NestedDict):
                        raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                    if param == "coda_height":
                        source_logamp = event.source_logamp(band, phase)
                        samples[i,j] =  source_logamp + model.predict(event)
                    else:
                        samples[i,j] = model.sample(event)
        return samples

    def log_likelihood(self, template_params, event, sta, chan, band):

        if self.dummy:
            return 1.0

        phases = template_params[0]
        param_vals = template_params[1]

        log_likelihood = 0
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(self.params()):
                if param == "arrival_time":
                    log_likelihood = self.travel_time_log_likelihood(tt=param_vals[i,j]-event.time, event=event, sta=sta, phase=phase)
                else:
                    model = self.models[param][sta][phase][chan][band]
                    if isinstance(model, NestedDict):
                        raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                    if param == "coda_height":
                        source_logamp = event.source_logamp(band, phase)
                        log_likelihood += model.posterior_log_likelihood(event, param_vals[i,j] - source_logamp)
                    else:
                        log_likelihood += model.posterior_log_likelihood(event, param_vals[i,j])

        return log_likelihood

    def travel_time(self, event, sta, phase):
        s = Sigvisa()
        siteid = s.name_to_siteid_minus1[sta] + 1
        phaseid = s.phaseids[phase]
        meantt = s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid-1, phaseid-1)
        return meantt

    def sample_travel_time(self, event, sta, phase):
        s = Sigvisa()
        meantt = self.travel_time(event, sta, phase)

        # peak of a laplace distribution is 1/2b, where b is the
        # scale param, so (HACK ALERT) we can recover b by
        # evaluating the density at the peak
        siteid = s.name_to_siteid_minus1[sta] + 1
        phaseid = s.phaseids[phase]
        ttscale = 2.0 / np.exp(s.sigmodel.arrtime_logprob(0, 0, 0, siteid-1, phaseid-1))

        # sample from a Laplace distribution:
        U = np.random.random() - .5
        tt = meantt - ttscale * np.sign(U) * np.log(1 - 2*np.abs(U))
        return tt

    def travel_time_log_likelihood(self, tt, event, sta, phase):
        s = Sigvisa()

        meantt = self.travel_time(event, sta, phase)
        siteid = s.name_to_siteid_minus1[sta] + 1
        phaseid = s.phaseids[phase]

        ll = s.sigmodel.arrtime_logprob(tt, meantt, 0, siteid-1, phaseid-1)
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
            start_idx = int(np.floor(start))
            if start_idx >= npts:
                continue

            offset = start - start_idx
            phase_env = self.abstract_logenv_raw(v, idx_offset = offset, srate=srate)
            end_idx = start_idx + len(phase_env)
            if end_idx <= 0:
                continue

            try:
                early = max(0, -start_idx)
                overshoot = max(0, end_idx - len(data))
                data[start_idx+early:end_idx-overshoot] += np.exp(phase_env[early:len(phase_env)-overshoot])
            except Exception as e:
                print e
                raise
        return data

    def generate_template_waveform(self, template_params, model_waveform, sample=False):
        s = Sigvisa()

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
            raise Exception("sampling is currently (somewhat) broken...")
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
