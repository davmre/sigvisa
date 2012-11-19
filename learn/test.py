import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from source.event import Event

from signals.common import Waveform, Segment
from signals.io import load_event_station
from signals.template_models.paired_exp import PairedExpTemplateModel

from learn.fit_shape_params import fit_event_segment, fit_template
import learn.optimize as optimize

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plotting.plot
from plotting.plot_coda_decays import *



class TestFit(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.event = Event(evid=5429382)
        self.s = Sigvisa()
        self.seg = load_event_station(self.event.evid, "URZ", cursor=self.s.cursor).with_filter("freq_2.0_3.0;env")
        self.tm = PairedExpTemplateModel(run_name="", model_type="dummy")

    def test_plot(self):
        pvals = np.array([[1238917955.54000, 4.10006, 1.35953, -0.04931], \
                          [1238918128.71000, 10.70001, .9597, -0.03423]])
        params = (('P', 'S'), pvals)

        pp = PdfPages("testplot.pdf")
        plot_waveform_with_pred(pp, self.seg['BHZ'], self.tm, params, logscale=True)
        pp.close()

    def test_fit_template_iid(self):
        tm = self.tm
        wave = self.seg['BHZ']
        pp = PdfPages("test_fit.pdf")
        fit_params, fit_cost = fit_template(wave, pp=pp, ev=self.event, tm=tm, method="simplex", wiggles=None, iid=True)
        pp.close()
#        fit_event_segment(event=event, sta='URZ', tm=tm, output_run_name="unittest", output_iteration=1, plot=False, wiggles=None, iid=True, extract_wiggles=False)



class TestOptimize(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_minimize(self):
        f = lambda x : (x-17)**2 + 4
        x0 = np.array([13])
        for method in ["bfgs", "tnc", "simplex", "anneal"]:
            x_min, f_min = optimize.minimize(f, x0, method=method, disp=False)
            self.assertAlmostEqual(x_min, 17, places=1)
            self.assertAlmostEqual(f_min, 4, places=1)

    def test_minimize_matrix(self):

        f = lambda X : np.dot(X.flatten(), X.flatten())
        x0 = np.array([[1, 2], [3, 4]])
        x_opt = np.array([[0, 0], [0, 0]])
        x_fixedcol_opt = np.array([[1, 0], [3, 0]])
        for method in ["bfgs", "tnc"]:
            x_min, f_min = optimize.minimize_matrix(f, start=x0, method=method, maxfun=1000)
            self.assertAlmostEqual(np.sum((x_min - x_opt).flatten()), 0, places=1)
            self.assertAlmostEqual(f_min, 0, places=1)

            x_min, f_min = optimize.minimize_matrix(f, start=x0, method=method, fix_first_col=True, maxfun=1000)
            self.assertAlmostEqual(np.sum((x_min - x_fixedcol_opt).flatten()), 0, places=1)
            self.assertAlmostEqual(f_min, 10, places=1)


if __name__ == '__main__':
    unittest.main()
