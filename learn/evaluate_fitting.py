import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database.signal_data import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plotting import histogram

from optparse import OptionParser
from utils import arrival_select as arrival_select

from sigvisa import *
from signals.template_models.load_by_name import load_template_model
from signals.io import *


def main():

    parser = OptionParser()

    arrival_select.register_arrival_options(parser)

    (options, args) = parser.parse_args()

    arrivals = arrival_select.arrivals_from_options(options)

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    bands = ['freq_2.0_3.0',]
    chans = ['BHZ',]

    fname = os.path.join("logs", "eval_fits.pdf")
    pp = PdfPages(fname)


    tm = load_template_model(template_shape = "paired_exp", run_name=None, run_iter=None, model_type="dummy")


    costs = []
    times = []
    for (sta, evid) in arrivals:
        event = get_event(evid)
        seg = load_event_station(event.evid, sta, cursor=cursor).with_filter("env")
        for band in bands:
            band_seg = seg.with_filter(band)
            for chan in chans:

                wave = band_seg[chan]
                st = time.time()
                fit_params, fit_cost = fit_template(wave, pp=pp, ev=event, tm=tm, method="simplex", wiggles=None, iid=True, init_run_name=None, init_iteration=None)
                et = time.time()
                times.append(et-st)
                print "fit took %.3fs" % (et-st)
                costs.append(fit_cost)

    plt.figure()
    print "Costs:\n", histogram.plot_histogram(costs)
    plt.title("Costs")
    pp.savefig()
    plt.figure()
    print "Times\n", histogram.plot_histogram(costs)
    plt.title("Times")
    pp.savefig()
    pp.close()

if __name__ == "__main__":
    main()


