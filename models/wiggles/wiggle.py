import numpy as np

from sigvisa.signals.common import Waveform

def create_wiggled_phase(template_params, tm, phase, wiggle, wiggle_stime, st, npts, srate, chan, sta, band):

    (phases, vals) = template_params

    phase_idx = phases.index(phase)
    other_phases = phases[:phase_idx] + phases[phase_idx + 1:]
    other_vals = np.vstack([vals[:phase_idx, :], vals[phase_idx + 1:, :]])

    wave = Waveform(data=np.zeros((npts,)), srate=srate, stime=st, chan=chan, sta=sta, filter_str='%s;env' % band)
    template_with_phase = tm.generate_template_waveform((phases, vals), wave, sample=False)
    template_without_phase = tm.generate_template_waveform((other_phases, other_vals), wave, sample=False)

    start_idx = int((vals[phase_idx, 0] - wave['stime']) * wave['srate'])
    peak_idx = int(start_idx + (vals[phase_idx, 1] * wave['srate'])) - 1

    wiggled_phase_data = template_with_phase.data - template_without_phase.data
    if start_idx + len(wiggle) > len(wiggled_phase_data):
        wiggle = wiggle[:len(wiggled_phase_data) - start_idx]
    wiggled_phase_data[start_idx:start_idx + len(wiggle)] *= wiggle
    wiggled_phase_data += template_without_phase.data
    template_with_phase.data = wiggled_phase_data
    return template_with_phase


def extract_phase_wiggle(wave, template_params, tm, phase):
    (phases, vals) = template_params

    phase_idx = phases.index(phase)
    other_phases = phases[:phase_idx] + phases[phase_idx + 1:]
    other_vals = np.vstack([vals[:phase_idx, :], vals[phase_idx + 1:, :]])

    template_with_phase = tm.generate_template_waveform((phases, vals), wave, sample=False)
    template_without_phase = tm.generate_template_waveform((other_phases, other_vals), wave, sample=False)

    # ignore the first half-second of arrival since the template is so
    # small that the wiggles are basically meaningless
    skip_initial_s = 0.5
    start_idx = int(np.floor((vals[phase_idx, 0] + skip_initial_s - wave['stime']) * wave['srate']))
    wiggle_stime = wave['stime'] + float(start_idx) / wave['srate']
                                         # this will be close to 0.5s after the wave stime, but not exactly,
                                         # depending on the sampling rate.

    peak_idx = max(start_idx, int(start_idx + ((vals[phase_idx, 1] - skip_initial_s) * wave['srate'])))

    def wiggle_well_defined(with_phase, without_phase, idx, threshold=2):
        return np.log(with_phase.data[idx]) - np.log(without_phase.data[idx]) > threshold

    wiggle_data = []
    st = None
    et = None
    if wiggle_well_defined(template_with_phase, template_without_phase, peak_idx):
        i = peak_idx + 2
        while i < wave['npts'] and not wave.data.mask[i] and wiggle_well_defined(template_with_phase, template_without_phase, i):
            i += 1
        wiggle_data = (wave.data[start_idx:i] - template_without_phase.data[start_idx:i]) / (template_with_phase.data[
                                                                                             start_idx:i] - template_without_phase.data[start_idx:i])

        st = wiggle_stime
        et = st + i / wave['srate']

    return wiggle_data, st, et
