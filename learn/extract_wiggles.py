import os, errno, sys, time, traceback, pdb
import numpy as np
from scipy import stats


from sigvisa.database.dataset import *
from sigvisa.database import db
from sigvisa.database.signal_data import *

from optparse import OptionParser

from sigvisa import Sigvisa



from sigvisa.source.event import get_event
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan
from sigvisa.models.templates.load_by_name import load_template_model

from sigvisa.models.wiggles.wiggle import extract_phase_wiggle


def main():

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser = OptionParser()
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="")
    parser.add_option("-i", "--run_iter", dest="run_iter", default=None, type="int", help="")
    (options, args) = parser.parse_args()

    if options.run_name is None:
        raise Exception("must specify a run name!")
    else:
        run_name = options.run_name
    if options.run_iter is None:
        raise Exception("must specify a run iteration!")
    else:
        iteration = options.run_iter

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggle_data")
    run_wiggle_dir = os.path.join(wiggle_dir, options.run_name, str(options.run_iter))
    ensure_dir_exists(run_wiggle_dir)

    runid = get_fitting_runid(cursor, run_name, iteration, create_if_new=False)
    sql_query = "select fp.fpid, fp.phase, f.fitid, fp.template_model from sigvisa_coda_fit_phase fp, sigvisa_coda_fit f where f.runid=%d and fp.fitid=f.fitid and fp.wiggle_stime IS NULL" % (runid, )
    cursor.execute(sql_query)
    phases = cursor.fetchall()
    for (fpid, phase, fitid, template_shape) in phases:
        tm = load_template_model(template_shape = template_shape, model_type="dummy")

        # load the waveform for this fit
        sql_query = "select evid, sta, chan, band from sigvisa_coda_fit where fitid=%d" % fitid
        cursor.execute(sql_query)
        (evid, sta, chan, band) = cursor.fetchone()
        wave = load_event_station_chan(evid, sta, chan, cursor=cursor).filter(band + ";env")

        # extract the wiggle
        template_params = load_template_params_by_fitid(cursor, fitid, return_cost=False)
        wiggle, st, et = extract_phase_wiggle(wave, template_params, phase=phase, tm=tm)
        if len(wiggle) < wave['srate']:
            print "evid %d phase %s at %s (%s, %s) is not prominent enough to extract a wiggle, skipping..." % (evid, phase, sta, chan, band)
            fname = "NONE"
            st = -1
        else:
            wiggle_wave = Waveform(data=wiggle, srate=wave['srate'], stime=st, sta=sta, chan=chan, evid=evid)
            fname = os.path.join(options.run_name, str(options.run_iter), "%d.wave" % (fpid,))
            wiggle_wave.dump_to_file(os.path.join(wiggle_dir, fname))
            print "extracted wiggle for fpid %d." % fpid

        sql_query = "update sigvisa_coda_fit_phase set wiggle_fname='%s', wiggle_stime=%f where fpid=%d" % (fname, st, fpid,)
        cursor.execute(sql_query)
        s.dbconn.commit()

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
