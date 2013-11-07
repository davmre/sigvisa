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
from sigvisa.models import DummyModel
from sigvisa.models.distributions import Gamma, Negate, Gaussian

import scipy.weave as weave
from scipy.weave import converters


ARR_TIME_PARAM, PEAK_OFFSET_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(4 + 1)


class PairedExpTemplateGenerator(TemplateGenerator):

    def __init__(self, *args, **kwargs):
        super(PairedExpTemplateGenerator, self).__init__(*args, **kwargs)

        self.uamodels = {"peak_offset": Gamma(2.0, 0.4),
                         "coda_height": Gaussian(-.5, 1),
                         "coda_decay": Negate(Gamma(4.0, 160.0)),}

    @staticmethod
    def params():
        return ("peak_offset", "coda_height", "coda_decay")

    @staticmethod
    def default_param_vals():
        d = dict()
        d['peak_offset'] = 1.0
        d['coda_height'] = 1.0
        d['coda_decay'] = -0.03
        return d

    @staticmethod
    def model_name():
        return "paired_exp"


    def create_param_node(self, graph, site, phase, band, chan, param,
                          evnodes, atime_node=None, amp_transfer_node=None, children=(), **kwargs):
        nodes = dict()
        if param == "coda_height":
            # we want to create a coda height node for each station under the current site.
            # each of these nodes will depend on the approprite amp_transfer parent
            nodes = dict()
            eid = evnodes['mb'].eid
            for sta in graph.site_elements[site]:
                label = create_key(param=param, sta=sta, phase=phase,
                                          eid=eid, chan=chan, band=band)
                atn = extract_sta_node(amp_transfer_node, sta)
                my_children = [wn for wn in children if wn.sta==sta]
                nodes[sta] = CodaHeightNode(eid=eid, sta=sta, band=band,
                                            chan=chan, phase=phase,
                                            label = label, parents=[evnodes['mb'], evnodes['natural_source'], atn],
                                            children=my_children,
                                            initial_value=self.default_param_vals()['coda_height'])
                graph.add_node(nodes[sta], template=True)
            return nodes
        else:
            return graph.setup_site_param_node(param=param, site=site, phase=phase, parent=evnodes['loc'],
                                               chan=chan, band=band, template=True, children=children, **kwargs)


    @staticmethod
    def abstract_logenv_raw(vals, min_logenv=-7.0, idx_offset=0.0, srate=40.0):
        arr_time, peak_offset, coda_height, coda_decay = \
            vals['arrival_time'], vals['peak_offset'], vals['coda_height'], vals['coda_decay']

        if np.isnan(peak_offset) or np.isnan(coda_height) or np.isnan(coda_decay) or coda_decay > 0:
            return np.empty((0,))

        if coda_decay > -0.001:
            l = int(1200 * srate)
        else:
                # minimum length is 2, so that even very small arrivals
                # can create a small bump (not doing this confuses the
                # approx-gradient routine; it tries making the bump
                # slightly bigger but with no effect since it's too small
                # to create a nonzero-length envelope).
            l = int(max(2.0, min(1200.0, peak_offset + (min_logenv - coda_height) / coda_decay) * srate))
        d = np.empty((l,))
        code = """
double onset_slope;
double peak_idx = std::max(0.0, peak_offset * srate);
if (peak_idx != 0) {
    onset_slope = exp(coda_height) / peak_idx;
} else {
    onset_slope = 0;
}

int intro_len = std::min(l, int(idx_offset + peak_idx) + 1);
double min_env = exp(min_logenv);

npy_intp dims[1] = {l};

for (int i=1; i < intro_len; ++i) {
  d(i) = log((i - idx_offset) * onset_slope + min_env);
}
double initial_decay = intro_len - idx_offset - peak_idx;
for (int i=0; i < l-intro_len; ++i) {
  d(i + intro_len) = (i + initial_decay) / srate * coda_decay + coda_height;
}
if (l > 0) {
  d(0) = -999;
}
"""
        weave.inline(code,['l', 'd', 'peak_offset', 'coda_height', 'coda_decay', 'min_logenv', 'idx_offset', 'srate'],type_converters = converters.blitz,verbose=2,compiler='gcc')
        return d


    def low_bounds(self):

        bounds = { k: -np.inf for k in self.params() }

        bounds['coda_height'] = -4
        bounds['peak_offset'] = 0.2
        bounds['coda_decay'] = -2

        return bounds

    def high_bounds(self):

        bounds = { k: np.inf for k in self.params() }

        bounds['coda_height'] = 10
        bounds['peak_offset'] = 25
        bounds['coda_decay'] = -.0001

        return bounds

    def unassociated_model(self, param, nm=None):
        if nm is not None and param=="coda_height":
            mu = np.log(nm.c * 10)
            std = 1.0
            return Gaussian(mu, std)
        else:
            return self.uamodels[param]
