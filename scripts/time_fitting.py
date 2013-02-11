import time

import numpy as np
import numpy.ma as ma


from sigvisa import Sigvisa
from sigvisa.models.templates.paired_exp import PairedExpTemplateModel
from sigvisa.source.event import get_event


from sigvisa.learn.fit_shape_params import fit_event_segment, fit_template


def main():
    event = get_event(evid=2781427)  # Event(evid=5301405)
    tm = PairedExpTemplateModel(run_name="", run_iter=0, model_type="dummy")

    for optim_method in ('bfgs',):
        t = time.time()
        fit_event_segment(event=event, sta='FITZ', tm=tm, output_run_name="fit_profiling", output_iteration=1,
                          plot=False, wiggles=None, iid=True, extract_wiggles=False, method=optim_method)
        print "fit ev %d at %s in %f seconds using method %s." % (event.evid, "FITZ", time.time() - t, optim_method)

if __name__ == "__main__":
    main()
