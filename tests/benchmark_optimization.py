import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from sigvisa.source.event import get_event

from sigvisa.signals.common import Waveform, Segment
from sigvisa.signals.io import load_event_station
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.models.spatial_regression.SpatialGP import distfns, SpatialGP, start_params, gp_extract_features

from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.infer.optimize.gradient_descent import approx_gradient


import matplotlib

from sigvisa.plotting.plot_coda_decays import *

from sigvisa.learn.train_param_common import learn_model, load_model, get_model_fname
from sigvisa.learn.train_coda_models import get_shape_training_data
from sigvisa.learn.train_wiggle_models import get_wiggle_training_data
from sigvisa.database.signal_data import RunNotFoundException


def setup_graph(chans = ('BHZ',)):
    np.random.seed(0)
    event = get_event(evid=5301405)
    sta = "FITZ"

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    seg = load_event_station(event.evid, sta, cursor=cursor).with_filter("freq_2.0_3.0;env")
    cursor.close()


    sg = SigvisaGraph(phases = ['P', 'S'])
    sg.add_event(event)
    for c in chans:
        wn = sg.add_wave(seg[c])
        wn.set_noise_model(nm_type='l1')

    return sg

def benchmark_derivs():
    sg = setup_graph(('BHZ', 'BHN', 'BHE'))

    node_list = list(sg.template_nodes)
    all_children = [child for node in node_list for child in node.children]
    relevant_nodes = set(node_list + all_children)

    vals = np.concatenate([node.get_mutable_values() for node in node_list])
    jp = lambda v: sg.joint_logprob(values=v, relevant_nodes=relevant_nodes, node_list=node_list)

    N = 20
    st = time.time()
    for i in range(N):
        grad1 = approx_gradient(jp, vals, eps=1e-4)
    et = time.time()
    print "approx done"
    for i in range(N):
        grad2 = sg.log_p_grad(values=vals, node_list = list(sg.template_nodes), relevant_nodes=relevant_nodes)
    et2 = time.time()

    print grad1
    print "joint grad in %f s" % ( (et-st)/N)
    print grad2
    print "decomposed grad in %f s" % ( (et2-et)/N )

def benchmark_deriv_optimization():
    sg = setup_graph(('BHZ','BHE'))
    optim_params = construct_optim_params("'method': 'bfgs', 'disp': True, 'normalize': True")
    lp0 = sg.current_log_p()
    print "start logp", lp0

    t1 = time.time()

    sg.prior_predict_all()
    sg.joint_optimize_nodes(node_list = sg.template_nodes, optim_params=optim_params, use_grad=True)
    lp1 = sg.current_log_p()

    t2 = time.time()

    sg.prior_predict_all()
    sg.joint_optimize_nodes(node_list = sg.template_nodes, optim_params=optim_params, use_grad=False)
    lp2 = sg.current_log_p()

    t3 = time.time()

    print "start logp", lp0
    print "grad: %.4fs, ll = %.2f" % (t2-t1, lp1)
    print "no grad: %.4fs, ll = %.2f" % (t3-t2, lp2)

if __name__ == "__main__":
    benchmark_deriv_optimization()
