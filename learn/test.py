import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from source.event import Event

from signals.common import Waveform, Segment
from signals.io import load_event_station
from signals.template_models.paired_exp import PairedExpTemplateModel

from learn.fit_shape_params import fit_event_segment, fit_template
from learn.SpatialGP import distfns, SpatialGP, start_params, gp_extract_features

import learn.optimize as optimize

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plotting.plot
from plotting.plot_coda_decays import *

from train_coda_models import learn_model, load_model, get_training_data, get_model_fname

class TestFit(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.event = Event(evid=2781427) # Event(evid=5301405)
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
        plot_waveform_with_pred(pp, self.seg['BHZ'], self.tm, params, logscale=True)
        pp.close()

    def test_fit_template_iid(self):
        tm = self.tm
        wave = self.seg['BHZ']
        pp = PdfPages("test_fit.pdf")
        t = time.time()
        fit_params, fit_cost = fit_template(wave, pp=pp, ev=self.event, tm=tm, method="simplex", wiggles=None, iid=True)
        print "fit ev %d at %s in %f seconds." % (self.event.evid, self.sta, time.time() - t)
        print "got params", fit_params
        pp.close()



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


class TestModels(unittest.TestCase):

    def setUp(self):
        self.X = np.array([ \
            [120, 30, 0, 1000, 32], \
            [118, 31, 0, 1050, 32], \
            [120, 29, 40, 1000, 34] \
                ])
        self.y = np.array([ \
                -0.02, \
                -0.01, \
                -0.015])
        self.evids = np.array([ \
                1, \
                2, \
                3])
        self.testX1 = np.array([120, 30, 0, 1025, 32])
        self.testX2 = np.array([119, 31, 0, 1000, 33])


    def test_constant(self):
        model = learn_model(self.X, self.y, model_type="constant")
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)
        self.assertAlmostEqual(pred1, np.mean(self.y))
        self.assertAlmostEqual(pred1, pred2)

        fname = "test_constant_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "constant")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

    def test_linear_distance(self):
        model = learn_model(self.X, self.y, model_type="linear_distance")
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)
        self.assertAlmostEqual(pred1, -0.01375)
        self.assertAlmostEqual(pred2, -0.0175)

        fname = "test_linear_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "linear_distance")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

    def test_GP(self):
        model = learn_model(self.X, self.y, model_type="gp_dad_log", target="decay")
        pred1 = model.predict(self.testX1)
        pred2 = model.predict(self.testX2)
        self.assertAlmostEqual(pred1, -0.0149555023)
        self.assertAlmostEqual(pred2, -0.0150870250)

        fname = "test_gp_model"
        model.save_trained_model(fname)
        nmodel = load_model(fname, "gp_dad_log")
        pred3 = nmodel.predict(self.testX1)
        self.assertAlmostEqual(pred1, pred3)

class TestLearnModel(unittest.TestCase):

    def test_learn_model(self):
        site="CTA"
        chan="BHZ"
        band="freq_2.0_3.0"
        phase="P"
        target="decay"

        run_name = "run4"
        run_iter = 0

        model_type = "gp_dad_log"

        X, y, evids = get_training_data(run_name, run_iter, site, chan, band, [phase,], target)

        model_fname = get_model_fname(run_name, run_iter, site, chan, band, phase, target, model_type, evids, model_name="paired_exp")
        print model_fname
        distfn = model_type[3:]
        model = learn_model(X, y, model_type, target=target)

        model.save_trained_model(model_fname)


if __name__ == '__main__':
    unittest.main()
