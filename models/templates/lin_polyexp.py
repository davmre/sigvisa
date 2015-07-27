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
from sigvisa.models.distributions import Gamma, Negate, Gaussian, TruncatedGaussian, Beta

import scipy.weave as weave
from scipy.weave import converters


ARR_TIME_PARAM, PEAK_OFFSET_PARAM, PEAK_DECAY_PARAM, CODA_HEIGHT_PARAM, CODA_DECAY_PARAM, NUM_PARAMS = range(5 + 1)

MAX_PEAK_OFFSET_S = 20.0

class LinPolyExpTemplateGenerator(TemplateGenerator):

    def __init__(self, *args, **kwargs):
        super(LinPolyExpTemplateGenerator, self).__init__(*args, **kwargs)

        self.uamodels = {"peak_offset": Gaussian(.3, 1.1),
                         "peak_decay": Gaussian(-2.5, 1.),
                         "coda_height": Gaussian(0.5, 3),
                         "coda_decay": Gaussian(-2.5, 1.5),
                         "mult_wiggle_std": Beta(4.23342488227, 1.41958698073)}

        self.hack_force_mean = None

    @staticmethod
    def params():
        return ("peak_offset", "peak_decay", "coda_height", "coda_decay", "mult_wiggle_std")

    @staticmethod
    def default_param_vals():
        d = dict()
        d['peak_offset'] = 0.0
        d['peak_decay'] = 0.0
        d['coda_height'] = 1.0
        d['coda_decay'] = np.log(0.03)
        d['mult_wiggle_std'] = 0.5
        return d

    @staticmethod
    def model_name():
        return "lin_polyexp"


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
            return graph.setup_site_param_node(param=param, site=site, phase=phase,
                                               parents=[evnodes['loc'], evnodes['mb']],
                                               chan=chan, band=band, template=True, children=children, **kwargs)


    @staticmethod
    def abstract_logenv_length(vals, min_logenv=-7.0, srate=40.0):
        peak_offset, peak_decay, coda_height, coda_decay = \
            float(np.exp(vals['peak_offset'])), float(-np.exp(vals['peak_decay'])), float(vals['coda_height']), float(-np.exp(vals['coda_decay']))

        if coda_decay > -0.001:
            l = int(1200 * srate)
        else:
                # minimum length is 2, so that even very small arrivals
                # can create a small bump (not doing this confuses the
                # approx-gradient routine; it tries making the bump
                # slightly bigger but with no effect since it's too small
                # to create a nonzero-length envelope).

            # also, these lengths are pessimistic since they don't account for peak_decay
            l = int(max(2.0, min(1200.0, peak_offset + (min_logenv - coda_height) / coda_decay) * srate))
        return l

    @staticmethod
    def abstract_logenv_raw(vals, min_logenv=-7.0, idx_offset=0.0, srate=40.0, return_jac_exp=False, fixedlen=None):
        peak_offset, peak_decay, coda_height, coda_decay = \
            float(np.exp(vals['peak_offset'])), float(-np.exp(vals['peak_decay'])), float(vals['coda_height']), float(-np.exp(vals['coda_decay']))
        return_jac_exp = int(return_jac_exp)
        idx_offset = float(idx_offset)

        if np.isnan(peak_offset) or np.isnan(coda_height) or np.isnan(coda_decay) or np.isnan(peak_decay):
            return np.empty((0,))

        if fixedlen:
            l = int(fixedlen)
        elif coda_decay > -0.001:
            # warning: next two cases duplicate the code in
            # abstract_logenv_length above, but we don't call that
            # directly to avoid the overhead of unpacking the 'vals'
            # dict a second time.
            l = int(1200 * srate)
        else:
                # minimum length is 2, so that even very small arrivals
                # can create a small bump (not doing this confuses the
                # approx-gradient routine; it tries making the bump
                # slightly bigger but with no effect since it's too small
                # to create a nonzero-length envelope).
            l = int(max(2.0, min(1200.0, peak_offset + (min_logenv - coda_height) / coda_decay) * srate))

        if return_jac_exp:
            jacobian = np.empty((l, 5))
        else:
            jacobian = np.empty((0,0))

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
  double si = (i - idx_offset) * onset_slope + min_env;
  d(i) = log(si);

  if (return_jac_exp) {
      jacobian(i, 0) = -onset_slope*srate;
      jacobian(i, 1) = -si+min_env;
      jacobian(i, 2) = si - min_env;
      jacobian(i, 3) = 0;
      jacobian(i, 4) = 0;
  }
}

double initial_decay = intro_len - idx_offset - peak_idx;



for (int i=0; i < l-intro_len; ++i) {
  double t = (i + initial_decay) / srate;
  d(i + intro_len) = t * coda_decay + peak_decay * log(t+1) + coda_height;

  if (return_jac_exp) {
      double si = exp(d(i+intro_len));
      double dsi_dt = -(si * coda_decay + si * peak_decay/(t+1));
      jacobian(i+intro_len, 0) = dsi_dt;
      jacobian(i+intro_len, 1) = dsi_dt * peak_offset;
      jacobian(i+intro_len, 2) = si;
      jacobian(i+intro_len, 3) = log(t+1) * peak_decay * si;
      jacobian(i+intro_len, 4) = t * coda_decay * si;
  }
}
if (l > 0) {
  d(0) = min_logenv;
  if (return_jac_exp) {
     for (int d=0; d < 5; ++d) {
        jacobian(0, d) = 0;
     }
  }
}
"""
        weave.inline(code,['l', 'd', 'peak_offset', 'coda_height', 'peak_decay', 'coda_decay', 'min_logenv', 'idx_offset', 'srate', 'return_jac_exp', 'jacobian'],type_converters = converters.blitz,verbose=2,compiler='gcc')

        #if np.isfinite(vals.values()).all() and (~np.isfinite(d[1:])).any():
        # finite parameters should never produce an infinite signal...
        #    import pdb; pdb.set_trace()

        if return_jac_exp:
            return d, jacobian
        else:
            return d

    @staticmethod
    def low_bounds():

        bounds = { k: -np.inf for k in LinPolyExpTemplateGenerator.params() }

        bounds['coda_height'] = -4
        bounds['peak_offset'] = -2
        bounds['coda_decay'] = -7
        bounds['peak_decay'] = -7
        bounds['mult_wiggle_std'] = 0

        return bounds

    @staticmethod
    def high_bounds():

        bounds = { k: np.inf for k in LinPolyExpTemplateGenerator.params() }

        bounds['coda_height'] = 10
        bounds['peak_offset'] = 4
        bounds['coda_decay'] = 1
        bounds['peak_decay'] = 4
        bounds['mult_wiggle_std'] = 1

        return bounds

    def unassociated_model(self, param, nm=None):

        if nm is not None and param=="coda_height":
            return self.uamodels[param]
            """mu = np.log(nm.c * .3)
            if self.hack_force_mean is not None:
                mu = self.hack_force_mean
            std = 0.5

            return Gaussian(mu, std)
            """
        else:
            return self.uamodels[param]
