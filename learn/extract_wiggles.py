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
    if start_idx + len(wiggle) > len(wiggled_phase_data):
        wiggle = wiggle[:len(wiggled_phase_data)-start_idx]
    wiggled_phase_data[start_idx:start_idx+len(wiggle)] *= wiggle
    wiggled_phase_data += template_without_phase.data
    template_with_phase.data = wiggled_phase_data
    return template_with_phase
    
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
        while i < wave['npts'] and not wave.data.mask[i] and wiggle_well_defined(template_with_phase, template_without_phase, i):
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
    ensure_dir_exists(run_wiggle_dir)

    runid = get_fitting_runid(cursor, run_name, iteration, create_if_new=False)
    sql_query = "select fp.fpid, fp.phase, f.fitid, fp.template_model from sigvisa_coda_fit_phase fp, sigvisa_coda_fit f where f.runid=%d and fp.fitid=f.fitid and fp.wiggle_fname IS NULL" % (runid, )
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
        else:
            wiggle_wave = Waveform(data=wiggle, srate=wave['srate'], stime=st, sta=sta, chan=chan, evid=evid)
            fname = os.path.join(options.run_name, str(options.run_iter), "%d.wave" % (fpid,))
            wiggle_wave.dump_to_file(os.path.join(wiggle_dir, fname))
            print "extracted wiggle for fpid %d." % fpid
        
        sql_query = "update sigvisa_coda_fit_phase set wiggle_fname='%s' where fpid=%d" % (fname, fpid,)
        cursor.execute(sql_query)
        s.dbconn.commit()

if __name__ == "__main__":
    main()
