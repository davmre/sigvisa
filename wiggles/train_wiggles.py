import os, errno, sys, time, traceback
import numpy as np
from scipy import stats


from database.dataset import *
from database import db
from database.signal_data import *

from optparse import OptionParser, OptionGroup

from sigvisa import Sigvisa



from source.event import get_event
from signals.common import Waveform, load_waveform_from_file
from signals.io import load_event_station_chan
from signals.template_models.load_by_name import load_template_model
from signals.waveform_matching.fourier_features import FourierFeatures


def wiggles(dbconn, run_name, iteration):
    cursor = dbconn.cursor()
    runid = get_fitting_runid(cursor, run_name, iteration, create_if_new=False)
    sql_query = "select fp.fpid, fp.phase, fp.wiggle_fname, f.fitid, fp.template_model from sigvisa_coda_fit_phase fp, sigvisa_coda_fit f where f.runid=%d and fp.fitid=f.fitid and fp.wiggle_fname IS NOT NULL and fp.wiggle_fname != 'NONE'" % (runid, )
    cursor.execute(sql_query)
    for row in cursor:
        yield row
    cursor.close()

def main():

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser = OptionParser()
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="")
    parser.add_option("-i", "--run_iter", dest="run_iter", default=None, type="int", help="")
    parser.add_option("--log", dest="log", default=False, action="store_true", help="use log wiggles")
    parser.add_option("-t", "--lead_time", dest="lead_time", default=np.float("inf"), type="float", help="use only the first n seconds of the extracted wiggle")
    parser.add_option("-p", "--parameterization", dest="parameterization", default="fourier", type="str", help="how to parameterize the extracted waveform (fourier)")
    
    fourier_group = OptionGroup(parser, "Fourier params")
    fourier_group.add_option("-f", "--fundamental", dest="fundamental", default="0.1", type="float", help="fundamental frequency (0.1Hz)")
    fourier_group.add_option("--min_freq", dest="min_freq", default=None, type="float", help="using fourier params, this is the min freq")
    fourier_group.add_option("--max_freq", dest="max_freq", default=None, type="float", help="using fourier params, this is the max freq")
    parser.add_option_group(fourier_group)
    
    (options, args) = parser.parse_args()

    if options.run_name is None:
        raise Exception("must specify a run name!")
    else:
        run_name = options.run_name
    if options.run_iter is None:
        raise Exception("must specify a run iteration!")
    else:
        iteration = options.run_iter

    wiggle_dir = os.path.join(os.getenv("SIGVISA_HOME"), "wiggles")
    run_wiggle_dir = os.path.join(wiggle_dir, options.run_name, str(options.run_iter))

    if options.parameterization == "fourier":
        min_freq = options.min_freq if options.min_freq else options.fundamental
        max_freq = options.max_freq
        f = FourierFeatures(fundamental = options.fundamental, min_freq=min_freq, max_freq=max_freq)
        meta0 = options.fundamental
        meta1 = min_freq
        meta2 = max_freq
        
    for (fpid, phase, wiggle_fname, fitid, template_shape) in wiggles(s.dbconn, run_name, iteration):
        tm = load_template_model(template_shape = template_shape, model_type="dummy")
        wave = load_waveform_from_file(os.path.join(wiggle_dir, wiggle_fname))
        if options.log:
            print "log with min value", np.min(wave.data)
            wave = wave.filter("log")
        extract_idx = min(wave['npts'], int(options.lead_time * wave['srate']))
        param_blob = f.encode_params_from_signal(wave.data[:extract_idx], srate=wave['srate'])

        p = {"fpid": fpid, "stime": wave['stime'], "etime": wave['stime'] + float(extract_idx) / wave['srate'], "srate": wave['srate'], 'timestamp': time.time(), "type": options.parameterization, "log": options.log, "meta0": meta0, "meta1": meta1, "meta2": meta2, "params": param_blob}
        wiggleid = insert_wiggle(s.dbconn, p)
        print "inserted wiggle %d for fpid %d." % (wiggleid, fpid)
        s.dbconn.commit()


        
if __name__ == "__main__":
    main()
