import numpy as np

from sigvisa.signals.common import Waveform

def create_wiggled_phase(tm_node, wave_node, wiggle_data, skip_initial_s=0.5):

    other_tmnodes = [tm for tm in wave_node.parents.values() if (tm.label.startswith("template_") and not tm == tm_node)]
    noise = wave_node.nm.predict(n=wave_node.npts)
    template_with_phase = wave_node.assem_signal(include_wiggles=False) + noise
    template_without_phase = wave_node.assem_signal(include_wiggles=False, parent_templates=other_tmnodes) + noise
    wave = wave_node.get_wave()

    vals = tm_node.get_value()

    start_idx = int(np.floor((vals[0] + skip_initial_s - wave['stime']) * wave['srate']))
    wiggle_stime = wave['stime'] + float(start_idx) / wave['srate']
                                         # this will be close to 0.5s after the wave stime, but not exactly,
                                         # depending on the sampling rate.

    peak_idx = max(start_idx, int(start_idx + ((vals[1] - skip_initial_s) * wave['srate'])))

    wiggled_phase_data = template_with_phase - template_without_phase
    if start_idx + len(wiggle_data) > len(wiggled_phase_data):
        wiggle_data = wiggle_data[:len(wiggled_phase_data) - start_idx]
    wiggled_phase_data[start_idx:start_idx + len(wiggle_data)] *= wiggle_data
    wiggled_phase_data += template_without_phase
    template_with_phase = wiggled_phase_data
    return template_with_phase


def extract_phase_wiggle(tm_node, wave_node, skip_initial_s=0.5):

    other_tmnodes = [tm for tm in wave_node.parents.values() if (tm.label.startswith("template_") and not tm == tm_node)]
    noise = wave_node.nm.predict(n=wave_node.npts)
    template_with_phase = wave_node.assem_signal(include_wiggles=False) + noise
    template_without_phase = wave_node.assem_signal(include_wiggles=False, parent_templates=other_tmnodes) + noise
    wave = wave_node.get_wave()

    vals = tm_node.get_value()

    start_idx = int(np.floor((vals[0] + skip_initial_s - wave['stime']) * wave['srate']))
    wiggle_stime = wave['stime'] + float(start_idx) / wave['srate']
                                         # this will be close to 0.5s after the wave stime, but not exactly,
                                         # depending on the sampling rate.

    peak_idx = max(start_idx, int(start_idx + ((vals[1] - skip_initial_s) * wave['srate'])))

    def wiggle_well_defined(with_phase, without_phase, idx, threshold=2):
        return np.log(with_phase[idx]) - np.log(without_phase[idx]) > threshold


    wiggle_data = []
    st = None
    et = None
    if wiggle_well_defined(template_with_phase, template_without_phase, peak_idx):
        i = peak_idx + 2
        while i < wave['npts'] and not wave.data.mask[i] and wiggle_well_defined(template_with_phase, template_without_phase, i):
            i += 1
        wiggle_data = (wave.data[start_idx:i] - template_without_phase[start_idx:i]) / (template_with_phase[
                                                                                             start_idx:i] - template_without_phase[start_idx:i])

        st = wiggle_stime
        et = st + i / wave['srate']

    return wiggle_data, st, et
