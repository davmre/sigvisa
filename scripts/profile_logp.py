from sigvisa.learn.fit_shape_params import setup_graph
from sigvisa.source.event import get_event
import time

ev = get_event(evid=5326226)
sg = setup_graph(event=ev, sta="FIA3", chan="SHZ", band="freq_2.0_3.0",
                            tm_shape="paired_exp", tm_type="dummy",
                            wm_family="fourier_0.8", wm_type="dummy", phases="leb",
                            fit_hz=5.0, nm_type="ar", output_run_name="profile", output_iteration=1)


node_list = list(sg.template_nodes) + list(sg.wiggle_nodes)
node_list = [node for node in node_list if not node.deterministic()]
all_stochastic_children = [child for node in node_list for (child, intermediates) in node.get_stochastic_children()]
relevant_nodes = set(node_list + all_stochastic_children)

values = sg.get_all(node_list=node_list)

t0 = time.time()
for i in range(200):
    #prob = sg.joint_prob(values=values, relevant_nodes=relevant_nodes, node_list=node_list, c=-1)
    grad = sg.log_p_grad(values=values, relevant_nodes=relevant_nodes, node_list=node_list, c=-1)
t1 = time.time()
print grad
print "total time", t1-t0
