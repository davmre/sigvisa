import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from source.event import Event

from signals.common import Waveform, Segment
from signals.io import load_event_station
from signals.template_models.paired_exp import PairedExpTemplateModel

from learn.fit_shape_params import fit_event_segment
import learn.optimize as optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting.plot


class TestFit(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_fit_template_iid(self):
        s = Sigvisa()
        event = Event(evid=5301405)
        tm = PairedExpTemplateModel(run_name="", model_type="dummy")
        fit_event_segment(event=event, sta='URZ', tm=tm, output_run_name="unittest", plot=False, wiggles=None, iid=True, extract_wiggles=False)


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
