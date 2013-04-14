import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from sigvisa.source.event import get_event

from sigvisa.signals.common import Waveform, Segment
from sigvisa.signals.io import load_event_station
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.models.spatial_regression.SpatialGP import distfns, SpatialGP, start_params, gp_extract_features
from sigvisa.models.spatial_regression.baseline_models import LinearBasisModel, poly_basisfns


from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.optimize.gradient_descent import approx_gradient


import matplotlib

from sigvisa.plotting.plot_coda_decays import *

from sigvisa.learn.train_param_common import learn_model, load_model, get_model_fname
from sigvisa.learn.train_coda_models import get_shape_training_data
from sigvisa.learn.train_wiggle_models import get_wiggle_training_data
from sigvisa.database.signal_data import RunNotFoundException

class TestFit(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.event = get_event(evid=5301405)
        self.sta = "FITZ"

        self.s = Sigvisa()
        cursor = self.s.dbconn.cursor()
        self.seg = load_event_station(self.event.evid, self.sta, cursor=cursor).with_filter("freq_2.0_3.0;env")
        cursor.close()
        self.wave = self.seg['BHZ']

        self.sg = SigvisaGraph(phases = ['P', 'S'])
        self.sg.add_event(self.event)
        self.sg.add_wave(self.wave)

        self.optim_params = construct_optim_params("'method': 'none'")

    def test_deriv(self):
        wave_node = self.sg.get_wave_node(wave=self.wave)
        wave_node.set_noise_model(nm_type='l1')

        node_list = list(self.sg.template_nodes)
        all_children = [child for node in node_list for child in node.children]
        relevant_nodes = set(node_list + all_children)

        vals = np.concatenate([node.get_mutable_values() for node in node_list])
        jp = lambda v: self.sg.joint_prob(values=v, relevant_nodes=relevant_nodes, node_list=node_list)

        grad1 = approx_gradient(jp, vals, eps=1e-4)
        grad2 = self.sg.log_p_grad(values=vals, node_list = list(self.sg.template_nodes), relevant_nodes=relevant_nodes)

        self.assertTrue( (np.abs(grad1-grad2) < 0.001 ).all()  )


    def test_fit_template_iid(self):
        wave_node = self.sg.get_wave_node(wave=self.wave)
        wave_node.set_noise_model(nm_type='l1')

        t = time.time()

        self.sg.joint_optimize_nodes(node_list = self.sg.template_nodes, optim_params=self.optim_params)

#        print "iid fit ev %d at %s in %f seconds." % (self.event.evid, self.sta, time.time() - t)
#        print "got params", fit_params

    def test_fit_template_AR(self):
        t = time.time()

        wave_node = self.sg.get_wave_node(wave=self.wave)
        wave_node.set_noise_model(nm_type='ar')


        self.sg.joint_optimize_nodes(node_list = self.sg.template_nodes, optim_params=self.optim_params)


class TestLearnModel(unittest.TestCase):

    def test_learn_shape_model(self):
        site = "CTA"
        chan = "BHZ"
        band = "freq_2.0_3.0"
        phase = "P"
        target = "coda_decay"

        run_name = "run4"
        run_iter = 0

        model_type = "linear_distance"

        try:
            X, y, evids = get_shape_training_data(run_name=run_name, run_iter=run_iter, site=site, chan=chan, band=band, phases=[phase, ], target=target)
        except RunNotFoundException:
            return

        model_fname = get_model_fname(
            run_name, run_iter, site, chan, band, phase, target, model_type, evids, model_name="paired_exp")
        print model_fname
        distfn = model_type[3:]
        model = learn_model(X, y, model_type, sta=site, target=target)

        model.save_trained_model(model_fname)

    def test_poly_regression(self):
        N = 10
        bfn = poly_basisfns(3)
        X = np.reshape(np.linspace(-5, 5, N), (-1, 1))
        H = np.array([[f(x) for f in bfn] for x in X])
        coeffs = np.array([50, 1, 3, -3])
        sigma_n = 1
        np.random.seed(0)
        y = np.dot(H, coeffs) + np.random.randn(N) * sigma_n

        b = np.zeros((4,))
        B = np.eye(4) * 100000
        B[0,0] = (1000000)**2

        model = LinearBasisModel(X=X, y=y, basisfns=bfn, param_mean=b, param_covar=B, noise_std=sigma_n, H=H, compute_ll=True)

        tol = np.array((1, .4, .05, .05))
        self.assertTrue( (np.abs(model.mean - coeffs) < tol ).all())

        # test that the variances are reasonable: we should be most
        # certain about the high-order terms and least certain about
        # the low-order terms.
        v = np.diag(np.dot(model.sqrt_covar.T, model.sqrt_covar))
        self.assertGreater(v[0], v[1])
        self.assertGreater(v[1], v[2])
        self.assertGreater(v[2], v[3])

        # test that loading and saving work
        p = model.predict(X)
        c = model.covariance(X, include_obs=True)

        model.save_trained_model('model.npz')
        m = LinearBasisModel(fname='model.npz')
        p1 = m.predict(X)
        c1 = m.covariance(X, include_obs=True)

        self.assertTrue( (np.abs(p-p1) < .001).all() )
        self.assertTrue( (np.abs(c-c1) < .001).flatten().all() )

if __name__ == '__main__':
    unittest.main()
