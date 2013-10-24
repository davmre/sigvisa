import numpy as np
import numpy.ma as ma
from sigvisa.signals.common import Waveform

def create_wiggled_phase(arrival, wave_node, wiggle_data):
    wiggle_data = wiggle_data.filled(1)

    other_tmnodes = [tm for tm in wave_node.parents.values() if (tm.label.startswith("template_") and not tm == tm_node)]
    noise = wave_node.nm.predict(n=wave_node.npts)
    template_with_phase = wave_node.assem_signal(include_wiggles=False, arrivals=[arrival,])
    #template_without_phase = wave_node.assem_signal(include_wiggles=False, parent_templates=other_tmnodes) + noise
    wave = wave_node.get_wave()
    vals, tg = wave_node.get_template_params_for_arrival(eid=arrival[0], phase=arrival[1])

    start_idx = int(np.floor((vals['arrival_time'] -  wave['stime']) * wave['srate']))

    wiggled_phase_data = template_with_phase # - template_without_phase
    if start_idx + len(wiggle_data) > len(wiggled_phase_data):
        wiggle_data = wiggle_data[:len(wiggled_phase_data) - start_idx]
    wiggled_phase_data[start_idx:start_idx + len(wiggle_data)] *= wiggle_data
    wiggled_phase_data += noise
    return wiggled_phase_data

def extract_phase_wiggle(arrival, arrivals, wave_node, skip_initial_s=1.0):

    other_arrivals = [a for a in arrivals if not a == arrival]
    noise = wave_node.nm.predict(n=wave_node.npts)
    template_with_phase = ma.masked_array(data = wave_node.assem_signal(arrivals=arrivals, include_wiggles=False) + noise)
    template_without_phase = ma.masked_array(data = wave_node.assem_signal(arrivals=other_arrivals, include_wiggles=False) + noise)
    wave = wave_node.get_wave()

    vals, tg = wave_node.get_template_params_for_arrival(eid=arrival[0], phase=arrival[1])

    start_idx = int(np.floor((vals['arrival_time'] - wave['stime']) * wave['srate']))
    wiggle_stime = vals['arrival_time']
    peak_idx = max(start_idx, int(start_idx + (vals['peak_offset']  * wave['srate'])))
    skipped_idx = start_idx + int(skip_initial_s * wave['srate'])
    template_with_phase[start_idx: skipped_idx] = ma.masked

    def wiggle_well_defined(with_phase, without_phase, idx, threshold=1):
        return np.log(with_phase[idx]) - np.log(without_phase[idx]) > threshold

    wiggle_data = []
    st = None
    et = None
    i = max(skipped_idx, peak_idx + 2)
    if wiggle_well_defined(template_with_phase, template_without_phase, i):
        try:
            wave.data.mask[0]
            mask = wave.data.mask
        except IndexError:
            mask = np.array([False,] * len(wave.data))
        while i < wave['npts'] and not mask[i] and wiggle_well_defined(template_with_phase, template_without_phase, i):
            i += 1
        wiggle_data = (wave.data[start_idx:i] - template_without_phase[start_idx:i]) / \
            (template_with_phase[start_idx:i] - template_without_phase[start_idx:i])
        #wiggle_data[: skipped_idx - start_idx] = ma.masked

        #np.savetxt('wave_data.txt', wave.data[start_idx:i])
        #np.savetxt('tmpl_without_phase.txt', template_without_phase[start_idx:i])
        #np.savetxt('tmpl_with_phase.txt', template_with_phase[start_idx:i])
        #np.savetxt('wiggle_data.txt', wiggle_data)
        st = wiggle_stime
        et = st + i / wave['srate']

    return wiggle_data, st, et

def extract_phase_wiggle_for_proposal(arrival, arrivals, wave_node, wg,skip_initial_s=2.0):

    """
    Extract a wiggle for the purpose of an MCMC proposal, rather than for learning wiggle params.
    Differences:
       - always extracts the full time period required by the wiggle generator, even if no signal is avaialble
       - also subtracts out the wiggles of other, oevrlapping signals

    Probably this should eventually totally replace the other method.
    """

    other_arrivals = [a for a in arrivals if not a == arrival]
    noise = wave_node.nm.predict(n=wave_node.npts)
    #template_with_phase = ma.masked_array(data = wave_node.assem_signal(arrivals=arrivals, include_wiggles=False) + noise)
    #template_without_phase = ma.masked_array(data = wave_node.assem_signal(arrivals=other_arrivals, include_wiggles=False) + noise)
    phase_template =  ma.masked_array(data = wave_node.assem_signal(arrivals=(arrival,), include_wiggles=False), copy=True)
    explained_signal = ma.masked_array(data = wave_node.assem_signal(arrivals=other_arrivals, include_wiggles=True) + noise, copy=True)


    wave = wave_node.get_wave()

    vals, tg = wave_node.get_template_params_for_arrival(eid=arrival[0], phase=arrival[1])

    # indices are into the waveform at the wave node (as opposed to into the wiggle data that we'll return)
    start_idx = int(np.floor((vals['arrival_time'] - wave['stime']) * wave['srate']))
    start_idx_practical = max(0, start_idx) # if the theoretical start
                                            # time is before the wave
                                            # starts, we'll start
                                            # extracting data whenever
                                            # the wave does start
    start_idx_skip = start_idx_practical - start_idx #the difference
                                                     #between the
                                                     #theoretical and
                                                     #practical start
                                                     #times for wiggle
                                                     #extraction
    skip_npts = int(skip_initial_s * wave_node.srate) # skip the first
                                                      # few points to
                                                      # avoid
                                                      # divide-by-zero
                                                      # weirdness in
                                                      # the onset
    skip_npts_practical = max(0,
                              skip_npts - start_idx_skip) # but if the
                                                          # onset is
                                                          # happening
                                                          # "offscreen",
                                                          # we don't
                                                          # need to
                                                          # skip it
                                                          # since we
                                                          # can't see
                                                          # it anyway.
    npts = wg.npts
    wiggle_data = np.ones((npts,))

    end_idx = min(start_idx + npts, wave_node.npts)
    wiggle_npts = end_idx-start_idx_practical

    residual = wave.data[start_idx_practical:end_idx] - explained_signal[start_idx_practical:end_idx]
    residual[0:skip_npts_practical] = ma.masked
    wiggle = residual / phase_template[start_idx_practical:end_idx]
    #wiggle -= (ma.mean(wiggle) - 1) # make sure the wiggle has mean 1

    wiggle_data[start_idx_skip:start_idx_skip+wiggle_npts] = wiggle.filled(1.0)

    """
    from sigvisa.plotting.plot import basic_plot_to_file
    basic_plot_to_file(data=wiggle_data, fname='wiggle_data.png')
    basic_plot_to_file(data=phase_template, fname='phase_template.png')
    basic_plot_to_file(data=explained_signal, fname='explained_signal.png')
    basic_plot_to_file(data=residual, fname='residual.png')
    basic_plot_to_file(data=residual, data2=phase_template[start_idx:end_idx], fname='residual.png')
    """

    return wiggle_data
