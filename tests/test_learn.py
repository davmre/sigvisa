import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from sigvisa.source.event import get_event

from sigvisa.signals.common import Waveform, Segment
from sigvisa.signals.io import load_event_station
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.dag import get_relevant_nodes

from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.optimize.gradient_descent import approx_gradient


import matplotlib

from sigvisa.plotting.plot_coda_decays import *

from sigvisa.learn.train_param_common import learn_model, load_model, get_model_fname
from sigvisa.learn.train_coda_models import get_shape_training_data
from sigvisa.learn.train_wiggle_models import get_wiggle_training_data
from sigvisa.database.signal_data import RunNotFoundException, get_fitting_runid

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
        self.sg.add_wave(self.wave)
        self.sg.add_event(self.event)


        self.optim_params = construct_optim_params("'method': 'none'")

    def test_deriv(self):
        wave_node = self.sg.get_wave_node(wave=self.wave)
        wave_node.set_noise_model(nm_type='l1')

        node_list, relevant_nodes = get_relevant_nodes(self.sg.template_nodes)

        vals = np.concatenate([node.get_mutable_values() for node in node_list])
        jp = lambda v: self.sg.joint_logprob(values=v, relevant_nodes=relevant_nodes, node_list=node_list)

        grad1 = approx_gradient(jp, vals, eps=1e-4)
        grad2 = self.sg.log_p_grad(values=vals, node_list = node_list, relevant_nodes=relevant_nodes)

        print grad1
        print grad2
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

# commenting out for now because this test requires loading data from
# the database, so it *really* ought to have a first piece that puts
# the relevant data *into* the database. otherwise it breaks constantly.
"""
class TestLearnModel(unittest.TestCase):

    def test_learn_shape_model(self):
        site = "CTA"
        chan = "BHZ"
        band = "freq_2.0_3.0"
        phase = "P"
        target = "coda_decay"

        run_name = "run4"
        run_iter = 1

        s = Sigvisa()
        cursor = s.dbconn.cursor()
        runid = get_fitting_runid(cursor, run_name, run_iter, create_if_new=False)
        cursor.close()

        model_type = "linear_distance"

        try:
            X, y, evids = get_shape_training_data(runid=runid, site=site, chan=chan, band=band, phases=[phase, ], target=target)
        except RunNotFoundException:
            return

        model_fname = get_model_fname(
            run_name, run_iter, site, chan, band, phase, target, model_type, evids, model_name="paired_exp")
        print model_fname
        distfn = model_type[3:]
        model = learn_model(X, y, model_type, sta=site, target=target)

        model.save_trained_model(model_fname)
"""

if __name__ == '__main__':
    unittest.main()
