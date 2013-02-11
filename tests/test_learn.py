import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from sigvisa.source.event import get_event

from sigvisa.signals.common import Waveform, Segment
from sigvisa.signals.io import load_event_station
from sigvisa.models.templates.paired_exp import PairedExpTemplateModel

from sigvisa.models.spatial_regression.SpatialGP import distfns, SpatialGP, start_params, gp_extract_features

from sigvisa.infer.optimize.optim_utils import construct_optim_params

from sigvisa.models.wiggles.wiggle_models import StupidL1WiggleModel, PlainWiggleModel
from sigvisa.models.envelope_model import EnvelopeModel

import matplotlib

from sigvisa.plotting.plot_coda_decays import *

from sigvisa.learn.train_coda_models import learn_model, load_model, get_training_data, get_model_fname



class TestFit(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.event = get_event(evid=2781427) # Event(evid=5301405)
        self.sta = "FITZ"
        self.s = Sigvisa()
        cursor = self.s.dbconn.cursor()
        self.seg = load_event_station(self.event.evid, self.sta, cursor=cursor).with_filter("freq_2.0_3.0;env")
        self.tm = PairedExpTemplateModel(run_name="", run_iter=0, model_type="dummy")

    def test_plot(self):
        pvals = np.array([[1238917955.54000, 4.10006, 1.35953, -0.04931], \
                          [1238918128.71000, 10.70001, .9597, -0.03423]])
        params = (('P', 'S'), pvals)

        pp = PdfPages("testplot.pdf")
        plot_waveform_with_pred(wave=self.seg['BHZ'], tm=self.tm, template_params=params, logscale=True)
        pp.savefig()
        pp.close()

    def test_fit_template_iid(self):
        tm = self.tm
        wave = self.seg['BHZ']
        t = time.time()

        wm = StupidL1WiggleModel(tm)
        em = EnvelopeModel(template_model=tm, wiggle_model=wm, phases=None)
        ll, fit_params = em.wave_log_likelihood_optimize(wave=wave, event=self.event, use_leb_phases=True, optim_params=construct_optim_params())

        print "fit ev %d at %s in %f seconds." % (self.event.evid, self.sta, time.time() - t)
        print "got params", fit_params

    def test_fit_template_AR(self):
        tm = self.tm
        wave = self.seg['BHZ']
        t = time.time()

        wm = PlainWiggleModel(tm)
        em = EnvelopeModel(template_model=tm, wiggle_model=wm, phases=None)
        ll, fit_params = em.wave_log_likelihood_optimize(wave=wave, event=self.event, use_leb_phases=True, optim_params=construct_optim_params())

        print "fit ev %d at %s in %f seconds." % (self.event.evid, self.sta, time.time() - t)
        print "got params", fit_params




class TestLearnModel(unittest.TestCase):

    def test_learn_model(self):
        site="CTA"
        chan="BHZ"
        band="freq_2.0_3.0"
        phase="P"
        target="coda_decay"

        run_name = "run4"
        run_iter = 0

        model_type = "linear_distance"

        X, y, evids = get_training_data(run_name, run_iter, site, chan, band, [phase,], target)

        model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, model_type, evids, model_name="paired_exp")
        print model_fname
        distfn = model_type[3:]
        model = learn_model(X, y, model_type, sta=site, target=target)

        model.save_trained_model(model_fname)


if __name__ == '__main__':
    unittest.main()
