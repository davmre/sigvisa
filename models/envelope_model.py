
import os
import errno
import sys
import time
import traceback
import numpy as np
import scipy

from sigvisa.database.signal_data import filter_and_sort_template_params
from sigvisa.database.dataset import *
from sigvisa.database import db

from sigvisa.infer.optimize.optim_utils import minimize_matrix, construct_optim_params

from optparse import OptionParser

from sigvisa import Sigvisa, NestedDict
import sigvisa.utils.geog
import obspy.signal.util


class EnvelopeModel:

    """

    Compute the probability of a set of segments (signal
    envelopes), given a set of events. This is doen by either
    maximizing or integrating the probability over the template
    parameters describing the signals.

    """

    def __init__(self, template_model, wiggle_model, bands=None, chans=None, phases=None):

        self.sigvisa = Sigvisa()
        self.template_model = template_model
        self.wiggle_model = wiggle_model

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

    def get_method(self, method_str, segment=True):
        f_ll = None
        if method_str == "mode":
            f_ll = self.log_likelihood_mode if segment else self.wave_log_likelihood_mode
        elif method_str == "monte_carlo":
            f_ll = self.log_likelihood_montecarlo if segment else self.wave_log_likelihood_montecarlo
        elif method_str == "optimize":
            f_ll = self.log_likelihood_optimize if segment else self.wave_log_likelihood_optimize
        else:
            raise Exception("unrecognized marginalization method %s" % options.method)
        return f_ll

    def wave_log_likelihood_mode(self, wave, event):

        sta = wave['sta']
        chan = wave['chan']
        band = wave['band']

        # p(template | wave, event) ~ p(wave | template, event) * p(template | event)
        f = lambda params: self.wiggle_model.template_ncost(
            wave, self.phases, params) + self.template_model.log_likelihood((self.phases, params), event, sta, chan, band)

        # just use the mode parameters
        params = self.template_model.predictTemplate(event, sta, chan, band, phases=self.phases)
        ll = f(params)

        return ll, params

    def log_likelihood_mode(self, segment, event):

        total_ll = 0
        all_params = NestedDict()
        sta = segment['sta']

        for chan in self.chans:
            for band in self.bands:
                wave = segment.with_filter(band)[chan]
                ll, params = self.wave_log_likelihood_mode(wave, event)
                all_params[chan][band] = params
                total_ll += ll

        return total_ll, all_params

    def wave_log_likelihood_optimize(self, wave, event, use_leb_phases=False, optim_params=None):

        if optim_params is None:
            optim_params = construct_optim_params("")

        sta = wave['sta']
        chan = wave['chan']
        band = wave['band']

        if use_leb_phases:
            (phases, start_param_vals) = self.template_model.heuristic_starting_params(wave)
            (phases, start_param_vals) = filter_and_sort_template_params(phases, start_param_vals, Sigvisa().phases)
        else:
            phases = self.phases
            start_param_vals = self.template_model.predictTemplate(event, sta, chan, band, phases=phases)

        # p(template | wave, event) ~ p(wave | template, event) * p(template | event)
        f = lambda params: -self.wiggle_model.template_ncost(
            wave, phases, params) - self.template_model.log_likelihood((phases, params), event, sta, chan, band)

        low_bounds = None
        high_bounds = None
        if optim_params['method'] != "simplex":
            atimes = start_param_vals[:, 0]
            low_bounds = self.template_model.low_bounds(phases, default_atimes=atimes)
            high_bounds = self.template_model.high_bounds(phases, default_atimes=atimes)

        params, nll = minimize_matrix(
            f, start_param_vals, low_bounds=low_bounds, high_bounds=high_bounds, optim_params=optim_params)
        return nll, (phases, params)

    def log_likelihood_optimize(self, segment, event):

        total_ll = 0
        all_params = NestedDict()
        sta = segment['sta']

        for chan in self.chans:
            for band in self.bands:
                wave = segment.with_filter(band)[chan]
                ll, params = self.wave_log_likelihood_optimize(wave, event)
                all_params[chan][band] = params
                total_ll += ll

        return total_ll, all_params

    def wave_log_likelihood_montecarlo(self, wave, event, n=50):

        sta = wave['sta']
        chan = wave['chan']
        band = wave['band']

        # p(template | wave, event) ~ p(wave | template, event) * p(template | event)
        f = lambda params: self.wiggle_model.template_ncost(
            wave, self.phases, params) + self.template_model.log_likelihood((self.phases, params), event, sta, chan, band)

        sum_ll = np.float("-inf")

        samples = []

        for i in range(n):
            params = self.template_model.sample(event, sta, chan, band, self.phases)
            ll = f(params)
            samples.append((ll, params))
            sum_ll = np.logaddexp(sum_ll, ll) if not (ll < 1e-300) else sum_ll
            assert (not np.isnan(sum_ll))

        ll = sum_ll - np.log(n)

        return ll, samples

    def log_likelihood_montecarlo(self, segment, event):

        total_ll = 0
        all_samples = NestedDict()
        sta = segment['sta']

        for chan in self.chans:
            for band in self.bands:
                wave = segment.with_filter(band)[chan]
                ll, samples = self.wave_log_likelihood_montecarlo(wave, event)
                all_samples[chan][band] = samples
                total_ll += ll

        return total_ll, all_samples
