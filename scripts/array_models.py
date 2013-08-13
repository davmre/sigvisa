import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan
from sigvisa.source.event import get_event

from sigvisa.infer.optimize.optim_utils import construct_optim_params

def main():
    sg = SigvisaGraph(template_model_type="gp_lld", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family="dummy",
                      phases=["P",], nm_type = "ar", assume_envelopes=True,
                      wiggle_len_s = 60.0, arrays_joint=False, runid=1, iteration=1, base_srate=5.0)
    s = Sigvisa()
    elements = [el for el in s.get_array_elements("FINES")]

    cursor = s.dbconn.cursor()
    for element in elements:
        wave = load_event_station_chan(evid=5326226, sta=element, chan="SHZ", cursor=cursor).filter("%s;env" % "freq_2.0_3.0").filter('hz_5.0')
        sg.add_wave(wave)

    ev = get_event(evid=5326226)
    sg.add_event(ev)

    #sg.optimize_templates(construct_optim_params("'disp': True, 'method': 'bfgs'"))
    print "overall log prob", sg.current_log_p(verbose=True)

if __name__ == "__main__":
    try:
        #sample_template()
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
