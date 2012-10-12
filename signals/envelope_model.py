

import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser

import plot
import sigvisa
import learn, sigvisa_util
import signals.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util


from signals.train_wiggles import *
from signals.coda_decay_common import *
from signals.source_spectrum import *
from signals.templates import *
from signals.train_coda_models import CodaModel


class EnvelopeModel:

"""
Compute the probability of a set of segments (signal envelopes), given a set of events. This is doen by either maximizing or integrating the probability over the template parameters describing the signals.
"""



    def __init__(self, template_model, bands=None, chans=None, phases=None):
        
        self.sigvisa = Sigvisa()
        self.template_model = template_model

        if bands is None:
            self.bands = self.sigvisa.bands
        else:
            self.bands = bands

        if chans is None:
            self.chans = self.sigvisa.chans
        else:
            self.chans = chans

        if phases is None:
            self.phases = self.sigvisa.phases
        else:
            self.phases = phases


    def log_likelihood_mode(self, segment, event, iid=False):

        total_ll = 0
        set_noise_processes(self.sigmodel, segment)
        all_params = NestedDict()
        
        for chan in self.chans:
            for band in self.bands:
                wave = segment.filter(band)[chan]

                f = lambda params: -1 * c_cost(wave, self.phases, params, iid=iid) + self.template_model.log_likelihood(params, event, sta, chan, band, self.phases)

                #just use the mean parameters
                params = self.template_model.predictTemplate(event, sta, chan, band, phases=self.phases)
                ll = f(params)
                all_params[chan][band] = params
                total_ll += ll

        return total_ll, all_params

    def log_likelihood_optimize(self, segment, event, iid=False):

        total_ll = 0
        set_noise_processes(self.sigmodel, segment)
        all_params = NestedDict()
        
        for chan in self.chans:
            for band in self.bands:
                wave = segment.filter(band)[chan]

                f = lambda params: -1 * c_cost(wave, self.phases, params, iid=iid) + self.template_model.log_likelihood(params, event, sta, chan, band, self.phases)

                #optimize over parameters
                params = self.template_model.predictTemplate(event, sta, chan, band, phases=self.phases)

                sf = lambda flat_params : -1 * f(np.reshape(flat_params, (len(self.phases), -1)))
                params = scipy.optimize.fmin(sf, params.flatten(), maxfun=30)
                ll = f(params)
                all_params[chan][band] = params
                print "found best value", ll

                total_ll += ll

        return total_ll, all_params

    def log_likelihood_montecarlo(self, segment, event, iid=False, n=50):

        total_ll = 0
        set_noise_processes(self.sigmodel, segment)
        all_params = NestedDict()
        
        for chan in self.chans:
            for band in self.bands:
                wave = segment.filter(band)[chan]

                f = lambda params: -1 * c_cost(wave, self.phases, params, iid=iid) + self.template_model.log_likelihood(params, event, sta, chan, band, self.phases)

                sum_ll = np.float("-inf")

                best_params = None
                best_param_ll = np.float("-inf")

                for i in range(n):
                    params = self.template_model.sample(event, sta, chan, band, self.phases)
                    ll = f(params)
                    sum_ll = np.logaddexp(sum_ll, ll) if not (ll < 1e-300) else sum_ll

                    if ll > best_param_ll:
                        best_params = params
                        best_param_ll = ll

                    if np.isnan(sum_ll):
                        print "sum_ll is nan!!"
                        import pdb
                        pdb.set_trace()

                all_params[chan][band] = best_params

                ll = sum_ll - np.log(n)
                print "got ll", ll
                total_ll += ll

        return total_ll, all_params


    def plot_predicted_signal(self, s, event, pp, iid=False, band='narrow_envelope_2.00_3.00', chan='BHZ'):

        tr = s[chan][band]
        siteid = tr.stats.siteid
        
        ll, pdict  = self.log_likelihood(s, event, pp=pp, marginalize_method="mode", iid=iid)
        params = pdict[chan][band]
        if params is not None and not isinstance(params, NestedDict):
            gen_tr = get_template(self.sigmodel, tr, [1, 5], params)
            fig = plot.plot_trace(gen_tr, title="siteid %d ll %f \n p_arr %f p_height %f decay %f" % (siteid, ll, params[0, ARR_TIME_PARAM], params[0, CODA_HEIGHT_PARAM], params[0, CODA_DECAY_PARAM]))
            pp.savefig()
            plt.close(fig)


