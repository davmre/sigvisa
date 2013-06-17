import numpy as np
import sys
import os
from sigvisa import Sigvisa
import sigvisa.models.noise.noise_model as noise_model
from sigvisa.source.event import get_event
from sigvisa.signals.common import *
from sigvisa.models.templates.template_model import TemplateGenerator
from sigvisa.models.templates.coda_height import CodaHeightNode
from sigvisa.graph.graph_utils import create_key, extract_sta_node

ARR_TIME_PARAM, PEAK_OFFSET_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(4 + 1)

class PairedExpTemplateGenerator(TemplateGenerator):

    @staticmethod
    def params():
        return ("peak_offset", "coda_height", "coda_decay")

    @staticmethod
    def default_param_vals():
        d = dict()
        d['peak_offset'] = 1
        d['coda_height'] = 1
        d['coda_decay'] = -0.03
        return d

    @staticmethod
    def model_name():
        return "paired_exp"


    def create_param_node(self, graph, site, phase, band, chan, param,
                          event_node, tt_node=None, amp_transfer_node=None, **kwargs):
        nodes = dict()
        if param == "coda_height":
            # we want to create a coda height node for each station under the current site.
            # each of these nodes will depend on the approprite amp_transfer parent
            nodes = dict()
            for sta in graph.site_elements[site]:
                label = create_key(param=param, sta=sta, phase=phase,
                                          eid=event_node.eid, chan=chan, band=band)
                atn = extract_sta_node(amp_transfer_node, sta)
                nodes[sta] = CodaHeightNode(eid=event_node.eid, sta=sta, band=band,
                                            chan=chan, phase=phase,
                                            label = label, parents=[event_node, atn])
                graph.add_node(nodes[sta])

        else:
            return graph.setup_site_param_node(param=param, site=site, phase=phase, parent=event_node,
                                               chan=chan, band=band, **kwargs)


    @staticmethod
    def abstract_logenv_raw(vals, min_logenv=-7, idx_offset=0, srate=40):
        arr_time, peak_offset, coda_height, coda_decay = \
            vals['arrival_time'], vals['peak_offset'], vals['coda_height'], vals['coda_decay']
        assert(idx_offset >= 0 and idx_offset < 1)

        if np.isnan(vals.values()).any() or coda_decay > 0:
            return np.empty((0,))

        if coda_decay > -0.001:
            l = 1200 * srate
        else:
            # minimum length is 2, so that even very small arrivals
            # can create a small bump (not doing this confuses the
            # approx-gradient routine; it tries making the bump
            # slightly bigger but with no effect since it's too small
            # to create a nonzero-length envelope).
            l = int(max(2, min(1200, peak_offset + (min_logenv - coda_height) / coda_decay) * srate))
        d = np.empty((l,))

        peak_idx = max(0, peak_offset * srate)
        if peak_idx != 0:
            onset_slope = np.exp(coda_height) / peak_idx
        else:
            onset_slope = 0

        try:
            intro_len = min(len(d), int(idx_offset + peak_idx) + 1)
            if intro_len > 0 and onset_slope > 0:
                intro_env = (np.arange(intro_len) - idx_offset) * onset_slope + np.exp(min_logenv)

                # avoid taking log(0)
                intro_env[0] = 0.1
                d[0:intro_len] = np.log(intro_env)

                # say peak_idx is 9.9 and idx_offset is 0.2, so we have intro_len=11
                # then at t=1, we have onset[0.8]
                # up to t=10, where we have onset[9.8], which is what we want.
                # now imagine peak_idx is 10 and idx_offset is 0; again we have intro_len=11
                # now at t=10 we have onset[10], which is fine.

            # now for case a, at t=11 we are 10.8 into the signal, so we want decay[0.9]
            # for case b, at t=11 we are 11 into the signal, so we want decay[1]
            # in general at t=intro_len we are intro_len - idx_offset into the signal,
            # so we want decay[intro_len - idx_offset - peak_idx]
            initial_decay = intro_len - idx_offset - peak_idx
            d[intro_len:] = (np.arange(len(d) - intro_len) + initial_decay) / srate * coda_decay + coda_height
            if len(d) > 0:
                d[0] = -999
        except Exception as e:
            print e
            raise

        return d

    def low_bounds(self):

        bounds = { k: -np.inf for k in self.keys() }

        bounds['coda_height'] = -7
        bounds['peak_offset'] = 0.5
        bounds['coda_decay'] = -.2

        # only return bounds for the mutable params, since these are what we're optimizing over
        bounds = np.array([bounds[k] for k in self.keys() if self._mutable[k]])

        return bounds

    def high_bounds(self):

        bounds = { k: np.inf for k in self.keys() }

        bounds['coda_height'] = 10
        bounds['peak_offset'] = 25
        bounds['coda_decay'] = -.0001

        # only return bounds for the mutable params, since these are what we're optimizing over
        bounds = np.array([bounds[k] for k in self.keys() if self._mutable[k]])

        return bounds
