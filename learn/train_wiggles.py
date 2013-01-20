import os, errno, sys, time, traceback
import numpy as np
from scipy import stats


from database.dataset import *
from database import db
from database.signal_data import *

from optparse import OptionParser

from sigvisa import Sigvisa



from source.event import get_event
from signals.common import Waveform
from signals.io import load_event_station_chan
from signals.template_models.load_by_name import load_template_model
from signals.waveform_matching.fourier_features import FourierFeatures


def create_wiggled_phase(template_params, tm, phase, wiggle, st, npts, srate, chan, sta):
    
    (phases, vals) = template_params

    phase_idx = phases.index(phase)
    other_phases = phases[:phase_idx] + phases[phase_idx+1:]
    other_vals = np.vstack([vals[:phase_idx, :], vals[phase_idx+1:, :]])

    wave = Waveform(data = np.zeros((npts,)), srate=srate, stime=st, chan=chan, sta=sta)
    template_with_phase = tm.generate_template_waveform((phases, vals), wave, sample=False)
    template_without_phase = tm.generate_template_waveform((other_phases, other_vals), wave, sample=False)

    start_idx = int((vals[phase_idx, 0] - wave['stime']) * wave['srate']) + 1 # plus one to avoid div-by-zero
    peak_idx = int(start_idx + ( vals[phase_idx, 1] * wave['srate'])) - 1

    wiggled_phase_data = template_with_phase.data - template_without_phase.data
    wiggled_phase_data[start_idx:start_idx+len(wiggle)] *= wiggle
    wiggled_phase_data += template_without_phase.data
    return wiggled_phase_data
    
def extract_phase_wiggle(wave, template_params, tm, phase):
    (phases, vals) = template_params

    phase_idx = phases.index(phase)
    other_phases = phases[:phase_idx] + phases[phase_idx+1:]
    other_vals = np.vstack([vals[:phase_idx, :], vals[phase_idx+1:, :]])

    template_with_phase = tm.generate_template_waveform((phases, vals), wave, sample=False)
    template_without_phase = tm.generate_template_waveform((other_phases, other_vals), wave, sample=False)

    start_idx = int((vals[phase_idx, 0] - wave['stime']) * wave['srate']) + 1 # plus one to avoid div-by-zero
    peak_idx = int(start_idx + ( vals[phase_idx, 1] * wave['srate'])) - 1

    def wiggle_well_defined(with_phase, without_phase, idx, threshold=2):
        return np.log(with_phase.data[idx]) - np.log(without_phase.data[idx]) > threshold

    wiggle_data = []
    st = None
    et = None
    if wiggle_well_defined(template_with_phase, template_without_phase, peak_idx):
        i = peak_idx + 2
        while i < wave['npts'] and wiggle_well_defined(template_with_phase, template_without_phase, i):
            i += 1
        wiggle_data = (wave.data[start_idx:i] - template_without_phase.data[start_idx:i]) / (template_with_phase.data[start_idx:i] - template_without_phase.data[start_idx:i])

        st = vals[phase_idx, 0]
        et = st + i/wave['srate']
        
    return wiggle_data, st, et



def main():

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    parser = OptionParser()
    parser.add_option("-r", "--run_name", dest="run_name", default=None, type="str", help="")
    parser.add_option("-i", "--run_iter", dest="run_iter", default=None, type="int", help="")
    parser.add_option("-p", "--parameterization", dest="parameterization", default="fourier", type="str", help="how to parameterize the extracted waveform (fourier)")
    parser.add_option("-f", "--fundamental", dest="fundamental", default="0.1", type="float", help="fundamental frequency (0.1Hz)")
    parser.add_option("--min_freq", dest="min_freq", default=None, type="float", help="using fourier params, this is the min freq")
    parser.add_option("--max_freq", dest="max_freq", default=None, type="float", help="using fourier params, this is the max freq")
    parser.add_option("--start_at_fpid", dest="start_at_fpid", default=None, type="int", help="ignore all fpids smaller than this value")
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
    ensure_dir_exists(wiggle_dir)

    runid = get_fitting_runid(cursor, run_name, iteration, create_if_new=False)
    sql_query = "select fp.fpid, fp.phase, f.fitid, fp.template_model from sigvisa_coda_fit_phase fp, sigvisa_coda_fit f where f.runid=%d and fp.fitid=f.fitid %s" % (runid, "" if options.start_at_fpid is None else "and fp.fpid>%d" % options.start_at_fpid)
    cursor.execute(sql_query)
    phases = cursor.fetchall()
    for (fpid, phase, fitid, template_shape) in phases:
        tm = load_template_model(template_shape = template_shape, model_type="dummy")

        sql_query = "select wiggleid from sigvisa_coda_fit_phase_wiggle where fpid=%d" % (fpid,)
        cursor.execute(sql_query)
        w = cursor.fetchall()
        if len(w) != 0:
            print "skipping fpid %d because it already has a wiggle" % fpid
            continue

        sql_query = "select evid, sta, chan, band from sigvisa_coda_fit where fitid=%d" % fitid
        cursor.execute(sql_query)
        (evid, sta, chan, band) = cursor.fetchone()
        wave = load_event_station_chan(evid, sta, chan, cursor=cursor).filter(band + ";env")

        template_params = load_template_params_by_fitid(cursor, fitid, return_cost=False)
        
        wiggle, st, et = extract_phase_wiggle(wave, template_params, phase=phase, tm=tm)
        if len(wiggle) < wave['srate']:
            print "evid %d phase %s at %s (%s, %s) is not prominent enough to extract a wiggle, skipping..." % (evid, phase, sta, chan, band)
            continue

        fname = "%d_%d_%s.wave" % (fitid, fpid, phase)
        np.savetxt(os.path.join(wiggle_dir, fname), wiggle.filled(np.float("nan")))
        
        if options.parameterization == "fourier":
            if options.min_freq:
                min_freq = options.min_freq
            else:
                min_freq = float(band.split("_")[1])

            if options.max_freq:
                max_freq = options.max_freq
            else:
                max_freq = float(band.split("_")[2])

            f = FourierFeatures(fundamental = options.fundamental, min_freq=min_freq, max_freq=max_freq, srate = wave['srate'])            
            params = f.basis_decomposition(wiggle)
            n = len(params.flatten())
            
            meta1 = min_freq
            meta2 = max_freq
            params = repr(params.tolist())
        else:
            raise Exception("unrecognized wiggle type %s!" % options.parameterization)

        sql_query = "insert into sigvisa_coda_fit_phase_wiggle (fpid, stime, etime, srate, filename, timestamp, type, nparams, meta1, meta2, fundamental, params) values (%d, %f, %f, %f, '%s', %f, '%s', %d, %f, %f, %f, '%s')" % (fpid, st, et, wave['srate'], fname, time.time(), options.parameterization, n, meta1, meta2, options.fundamental, params.replace("'", "''"))
        #print sql_query
        cursor.execute(sql_query)
        print "extracted wiggle for fpid %d." % fpid
        s.dbconn.commit()

if __name__ == "__main__":
    main()
