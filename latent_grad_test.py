import numpy as np

from sigvisa import Sigvisa
from sigvisa.signals.io import load_event_station_chan
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.source.event import get_event
from sigvisa.infer.optimize.optim_utils import construct_optim_params

s = Sigvisa()

tmtype = {'amp_transfer': 'param_sin1', 'tt_residual': 'constant_laplacian', 'coda_decay': 'param_linear_distmb', 'peak_offset': 'param_linear_mb'}

sg = SigvisaGraph(template_model_type=tmtype, template_shape='paired_exp', wiggle_model_type='dummy', wiggle_family='dummy',
                  phases=['P'], nm_type = 'ar', run_name='everything', iteration=2)


ev = get_event(evid=5393637)

cursor = s.dbconn.cursor()
wave = load_event_station_chan(5393637, 'FINES', 'SHZ', cursor=cursor).filter("%s;env" % 'freq_2.0_3.0')
cursor.close()

sg.add_wave(wave=wave, fixed=True)
sg.add_event(ev=ev, fixed=True)
print 'adding event'
sg.parent_sample_all()

wn = sg.station_waves.values()[0][0]
latent = wn.parents['1;P;FIA0;SHZ;freq_2.0_3.0;latent_arrival']


analytic_grad = latent._debug_grad()

#sys.exit(1)

numeric_grad = np.zeros(analytic_grad.shape)

numeric_latent_grad = np.zeros(analytic_grad.shape)
numeric_obs_grad = np.zeros(analytic_grad.shape)

#base_lp = sg.current_log_p()
eps = 1e-2
latent_val = latent.get_value()
for i in range(30):#range(len(numeric_grad)):
    v = latent_val[i]
    latent.set_index(v + eps, i)
    lp1 = sg.current_log_p()

    latent_lp1 = latent.log_p()
    obs_lp1 = wn.log_p()

    latent.set_index(v-eps, i)
    lp2 = sg.current_log_p()

    latent_lp2 = latent.log_p()
    obs_lp2 = wn.log_p()

    latent.set_index(v, i)
    numeric_grad[i] = (lp1-lp2)/(2*eps)
    numeric_latent_grad[i] = (latent_lp1-latent_lp2)/(2*eps)
    numeric_obs_grad[i] = (obs_lp1-obs_lp2)/(2*eps)

#print 'numeric latent grad', numeric_latent_grad[:30]
print analytic_grad[:30]
print numeric_grad[:30]

#print analytic_grad[:30] - numeric_grad[:30]
