
import numpy as np
import sys, traceback, pdb
from optparse import OptionParser

from plotting.plot_coda_decays import plot_waveform_with_pred


from database.dataset import *
from database.signal_data import *
from signals.io import *

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from signals.template_models.paired_exp import *
from sigvisa import *



def main():
    parser = OptionParser()

    s = Sigvisa()

    parser.add_option("-s", "--site", dest="site", default=None, type="str", help="site for which to train models")
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="run_name")
    parser.add_option("--run_iter", dest="run_iter", default="latest", type="str", help="run iteration (latest)")
    parser.add_option("-c", "--channel", dest="chan", default="BHZ", type="str", help="name of channel to examine (BHZ)")
    parser.add_option("-n", "--band", dest="band", default="freq_2.0_3.0", type="str", help="name of band to examine (freq_2.0_3.0)")
    parser.add_option("--max_cost", dest="max_cost", default=10.0, type="float", help="maximum cost")

    (options, args) = parser.parse_args()

    sta = options.site
    chan = options.chan
    band = options.band
    acost_threshold = options.max_cost
    run_name = options.run_name

    if options.run_iter == "latest":
        iters = read_fitting_run_iterations(s.cursor, run_name)
        run_iter = np.max(iters[:, 0])
    else:
        run_iter = int(options.run_iter)
    runid = get_fitting_runid(s.cursor, run_name, run_iter, create_if_new=False)


    pieces = band.split('_')
    lowband = float(pieces[1])
    highband = float(pieces[2])
    sql_query = "select distinct evid from sigvisa_coda_fits where runid=%d and sta='%s' and chan='%s' and lowband=%f and highband=%f and acost < %f" % (runid, sta, chan, lowband, highband, acost_threshold)
    s.cursor.execute(sql_query)
    evids = s.cursor.fetchall()

    print "loaded", len(evids), "evids"

    from signals.template_models.load_by_name import load_template_model
    tm = load_template_model("paired_exp", run_name=None, run_iter=0, model_type="dummy")

    for evid in evids:
        evid = evid[0]
        try:
            # todo: make fit_shape_params also save to these directories
            fname = os.path.join("logs", "template_fits", run_name, "%02d" % run_iter, sta, chan, band)
            ensure_dir_exists(fname)
            fname = os.path.join(fname, str(evid) + ".pdf")
            pp = PdfPages(fname)
            print "writing ev %d to %s" % (evid, fname)

            (phases, vals), cost = load_template_params(evid, sta, chan, band, run_name, run_iter)
            seg = load_event_station(evid, sta, cursor=s.cursor).with_filter("env;"+band)
            wave = seg[chan]
            plt.clf()
            plot_waveform_with_pred(pp, wave, tm, (phases, vals), logscale=True)
            plot_waveform_with_pred(pp, wave, tm, (phases, vals), logscale=False)
            pp.close()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print "Warning: exception", e
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            raise
            continue

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

