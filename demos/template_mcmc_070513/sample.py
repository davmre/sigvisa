import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform

from sigvisa.plotting.plot import plot_with_fit

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "demos", "template_mcmc_070513")

def sample_template(seed=None, name="", env=True):
    if len(name) > 0:
        name += "_"

    if env:
        wiggle_family = "dummy"
    else:
        wiggle_family = "fourier_2.5"
    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = "ar", assume_envelopes=env,
                      wiggle_len_s = 60.0)

    wave = Waveform(data = np.zeros(500), srate=5.0, stime=1239915900.0, sta="FIA3", chan="SHZ", filter_str="freq_2.0_3.0;%shz_5.0" % ('env;' if env else ''))
    wn = sg.add_wave(wave)

    if seed is not None:
        np.random.seed(seed)

    templates = sg.prior_sample_uatemplates(wn, wiggles=True)
    print "sampled %d templates!" % len(templates)
    for (i, tmpl) in enumerate(templates):
        for (param, node) in tmpl.items():
            print "template %d param %s: %.3f" % (i, param, node.get_value())

    plot_with_fit(os.path.join(BASE_DIR, "%ssampled.png" % name), wn)
    with open(os.path.join(BASE_DIR, "%ssampled_templates.pkl" % name), 'w') as f:
        pickle.dump(templates, f)
    np.save(os.path.join(BASE_DIR, "%ssampled_wave.npy" % name), wn.get_value().data)

    lp = sg.current_log_p()
    print "sampled lp", lp

    #wn.parent_predict()
    #lp = sg.current_log_p()
    #print "predicted lp", lp


    sys.exit(0)

if __name__ == "__main__":
    try:
        sample_template(name="nowiggle", seed=3, env=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
