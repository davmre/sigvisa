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

def sample_template(env=True):
    if env:
        wiggle_family = "fourier_0.8"
    else:
        wiggle_family = "fourier_2.5"
    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = "ar", assume_envelopes=env,
                      wiggle_len_s = 60.0)

    wave = Waveform(data = np.zeros(500), srate=5.0, stime=1239915900.0, sta="FIA3", chan="SHZ", filter_str="freq_2.0_3.0;%shz_5.0" % ('env;' if env else ''))
    wn = sg.add_wave(wave)

    templates = sg.prior_sample_uatemplates(wn, wiggles=True)
    print "sampled %d templates!" % len(templates)
    for (i, tmpl) in enumerate(templates):
        for (param, node) in tmpl.items():
            print "template %d param %s: %.3f" % (i, param, node.get_value())

    plot_with_fit("unass_sampled.png", wn)
    with open("sampled_templates.pkl", 'w') as f:
        pickle.dump(templates, f)
    np.save("sampled_wave.npy", wn.get_value().data)
    sys.exit(0)

if __name__ == "__main__":
    try:
        sample_template()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
