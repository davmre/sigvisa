import time
import numpy as np
import os
from sigvisa import Sigvisa

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.source.event import get_event
from sigvisa.signals.io import load_event_station_chan

def load_sg_from_db_fit(fitid, load_wiggles=True):

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    fit_sql_query = "select f.runid, f.evid, f.sta, f.chan, f.band, f.hz, f.stime, f.etime, nm.model_type from sigvisa_coda_fit f, sigvisa_noise_model nm where f.fitid=%d and f.nmid=nm.nmid" % (fitid)
    cursor.execute(fit_sql_query)
    fit = cursor.fetchone()
    ev = get_event(evid=fit[1])
    wave = load_event_station_chan(fit[1], fit[2], fit[3], cursor=cursor).filter('%s;env;hz_%.2f' % (fit[4], fit[5]))
    nm_type = fit[8]
    runid = fit[0]

    phase_sql_query = "select fpid, phase, template_model, arrival_time, peak_offset, coda_height, coda_decay from sigvisa_coda_fit_phase where fitid=%d" % fitid
    cursor.execute(phase_sql_query)
    phase_details = cursor.fetchall()
    phases = [p[1] for p in phase_details]
    templates = {}
    tmshapes = {}
    for (phase, p) in zip(phases, phase_details):
        tmshapes[phase] = p[2]
        templates[phase] = {'arrival_time': p[3], 'peak_offset': p[4], 'coda_height': p[5], 'coda_decay': p[6]}

    if load_wiggles:
        wiggle_family = None
        wiggles = {}
        basisids = {}
        wiggle_family = None
        for (phase, phase_detail) in zip(phases, phase_details):
            wiggle_sql_query = "select w.wiggleid, w.params, w.basisid from sigvisa_wiggle w where w.fpid=%d " % (phase_detail[0])
            cursor.execute(wiggle_sql_query)
            w = cursor.fetchall()
            if len(w) < 1:
                raise KeyError("no wiggle found for phase fit ID %d" % phase_detail[0])

            assert(len(w) == 1) # if there's more than one wiggle
                                # parameterization of a phase, we'd need
                                # some way to disambiguate.

            basisids[phase] = w[0][2]
            wiggles[phase] = str(w[0][1])
    else:
        wiggle_family = "fourier_0.1"
        basisids = None

    sg = SigvisaGraph(template_model_type="dummy", wiggle_model_type="dummy",
                      template_shape=tmshapes, wiggle_family=wiggle_family,
                      nm_type = nm_type, runid=runid, phases=phases,
                      wiggle_basisids=basisids, base_srate=wave['srate'])
    wave_node = sg.add_wave(wave)
    sg.add_event(ev)


    for phase in phases:
        sg.set_template(eid=ev.eid, sta=wave['sta'], band=wave['band'],
                        chan=wave['chan'], phase=phase,
                        values = templates[phase])

        if load_wiggles:
            wg = sg.wiggle_generator(phase=phase, srate=wave['srate'])
            param_array = wg.decode_params(wiggles[phase])
            sg.set_template(eid=ev.eid, sta=wave['sta'], band=wave['band'],
                            chan=wave['chan'], phase=phase,
                            values = wg.array_to_param_dict(param_array))


    return sg
