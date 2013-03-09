import numpy as np
import numpy.ma as ma
from sigvisa.signals.common import Waveform

def create_wiggled_phase(tm_node, wave_node, wiggle_data):
    wiggle_data = wiggle_data.filled(1)

    other_tmnodes = [tm for tm in wave_node.parents.values() if (tm.label.startswith("template_") and not tm == tm_node)]
    noise = wave_node.nm.predict(n=wave_node.npts)
    template_with_phase = wave_node.assem_signal(include_wiggles=False) + noise
    template_without_phase = wave_node.assem_signal(include_wiggles=False, parent_templates=other_tmnodes) + noise
    wave = wave_node.get_wave()
    vals = tm_node.get_value()

    start_idx = int(np.floor((vals[0] -  wave['stime']) * wave['srate']))


    wiggled_phase_data = template_with_phase - template_without_phase
    if start_idx + len(wiggle_data) > len(wiggled_phase_data):
        wiggle_data = wiggle_data[:len(wiggled_phase_data) - start_idx]
    wiggled_phase_data[start_idx:start_idx + len(wiggle_data)] *= wiggle_data
    wiggled_phase_data += template_without_phase
    return wiggled_phase_data

def extract_phase_wiggle(tm_node, wave_node, skip_initial_s=1.0):

    other_tmnodes = [tm for tm in wave_node.parents.values() if (tm.label.startswith("template_") and not tm == tm_node)]
    noise = wave_node.nm.predict(n=wave_node.npts)
    template_with_phase = ma.masked_array(data = wave_node.assem_signal(include_wiggles=False) + noise)
    template_without_phase = wave_node.assem_signal(include_wiggles=False, parent_templates=other_tmnodes) + noise
    wave = wave_node.get_wave()

    vals = tm_node.get_value()

    start_idx = int(np.floor((vals[0] - wave['stime']) * wave['srate']))
    wiggle_stime = vals[0]
    peak_idx = max(start_idx, int(start_idx + (vals[1]  * wave['srate'])))
    skipped_idx = start_idx + int(skip_initial_s * wave['srate'])
    template_with_phase[start_idx: skipped_idx] = ma.masked

    def wiggle_well_defined(with_phase, without_phase, idx, threshold=1):
        return np.log(with_phase[idx]) - np.log(without_phase[idx]) > threshold

    wiggle_data = []
    st = None
    et = None
    i = max(skipped_idx, peak_idx + 2)
    if wiggle_well_defined(template_with_phase, template_without_phase, i):
        while i < wave['npts'] and not wave.data.mask[i] and wiggle_well_defined(template_with_phase, template_without_phase, i):
            i += 1
        wiggle_data = (wave.data[start_idx:i] - template_without_phase[start_idx:i]) / \
            (template_with_phase[start_idx:i] - template_without_phase[start_idx:i])
        #wiggle_data[: skipped_idx - start_idx] = ma.masked

        np.savetxt('wave_data.txt', wave.data[start_idx:i])
        np.savetxt('tmpl_without_phase.txt', template_without_phase[start_idx:i])
        np.savetxt('tmpl_with_phase.txt', template_with_phase[start_idx:i])
        np.savetxt('wiggle_data.txt', wiggle_data)


        st = wiggle_stime
        et = st + i / wave['srate']

    return wiggle_data, st, et
