
import unittest
import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa import Sigvisa

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.infer.run_mcmc import run_open_world_MH
from sigvisa.infer.template_mcmc import birth_move, death_move, split_move, merge_move
from sigvisa.plotting.plot import plot_with_fit
from sigvisa.signals.common import Waveform
from sigvisa.utils.fileutils import clear_directory

BASE_DIR = os.path.join(os.getenv("SIGVISA_HOME"), "logs", "mcmctest")

def sample_template(seed=None, wiggles = True):
    clear_directory(BASE_DIR)

    if wiggles:
        wiggle_family = "fourier_0.8"
    else:
        wiggle_family = "dummy"
    sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                      wiggle_model_type="dummy", wiggle_family=wiggle_family,
                      phases="leb", nm_type = "ar", wiggle_len_s = 60.0)

    wave = Waveform(data = np.zeros(500), srate=5.0, stime=1239915900.0, sta="FIA3", chan="SHZ", filter_str="freq_2.0_3.0;env;hz_5.0")
    wn = sg.add_wave(wave)
    sg.uatemplate_rate = .01

    if seed is not None:
        np.random.seed(seed)

    templates = sg.prior_sample_uatemplates(wn, wiggles=True)
    print "sampled %d templates!" % len(templates)
    for (i, tmpl) in enumerate(templates):
        for (param, node) in tmpl.items():
            print "template %d param %s: %.3f" % (i, param, node.get_value())

    wave = wn.get_wave()

    plot_with_fit(os.path.join(BASE_DIR, "sampled.png"), wn)
    lp = sg.current_log_p()
    with open(os.path.join(BASE_DIR, "sampled_logp.txt"), 'w') as f:
        f.write("%f\n" % lp)
    with open(os.path.join(BASE_DIR, "sampled_templates.pkl"), 'w') as f:
        pickle.dump(templates, f)

    print "sampled lp", lp
    return wave, templates

class TestUAMCMC(unittest.TestCase):

    def setUp(self):
        wave, templates = sample_template(seed=3, wiggles=False)
        sg = SigvisaGraph(template_model_type="dummy", template_shape="paired_exp",
                          wiggle_model_type="dummy", wiggle_family="fourier_0.8",
                          phases="leb", nm_type = "ar", wiggle_len_s = 60.0)
        sg.uatemplate_rate = .01
        self.wn = sg.add_wave(wave)

        self.sg = sg
        self.templates = templates

    def test_splitmerge(self):
        tvals = dict([(p, n.get_value()) for (p, n) in self.templates[0].items()])

        # initialize with a true good template
        self.sg.create_unassociated_template(self.wn, tvals['arrival_time'], initial_vals=tvals)
        lp1 =  self.sg.current_log_p()

        self.sg.debug_dump('testuamcmc-pre')

        # force a terrible split
        split_accepted, split_lpnew, split_lpold, split_log_qforward, split_log_qbackward, split_jd = split_move(self.sg, self.wn, return_probs=True, force_accept=True)
        print split_accepted, split_lpnew, split_lpold, split_log_qforward, split_log_qbackward, split_jd
        lp2 =  self.sg.current_log_p()
        self.assertAlmostEqual(split_lpnew-split_lpold, lp2-lp1)

        self.sg.debug_dump('testuamcmc')

        # check that the merge fixes the split and restores the original state.
        # note: this will only work for half of random seeds
        np.random.seed(2)
        merge_accepted, merge_lpnew, merge_lpold, merge_log_qforward, merge_log_qbackward, merge_jd = merge_move(self.sg, self.wn, return_probs=True)
        print merge_accepted, merge_lpnew, merge_lpold, merge_log_qforward, merge_log_qbackward, merge_jd
        self.assertTrue(merge_accepted)
        lp3 =  self.sg.current_log_p()
        self.sg.debug_dump('testuamcmc-post')
        self.assertAlmostEqual(lp1,lp3)

        self.assertAlmostEqual(split_lpnew-split_lpold, merge_lpold-merge_lpnew)
        self.assertAlmostEqual(split_log_qforward, merge_log_qbackward)
        self.assertAlmostEqual(merge_log_qforward, split_log_qbackward)

    def test_birthdeath(self):
        lp1 = self.sg.current_log_p()
        birth_accepted, birth_lp_new, birth_lp_old, birth_log_qforward, birth_log_qbackward, logu = birth_move(self.sg, self.wn, return_probs=True)
        lp2 = self.sg.current_log_p()
        print birth_accepted, birth_lp_new, birth_lp_old, birth_log_qforward, birth_log_qbackward

        self.sg.debug_dump("testuamcmc")
        self.assertTrue(birth_accepted)
        self.assertAlmostEqual(birth_lp_new-birth_lp_old, lp2-lp1)

        death_accepted, death_lp_new, death_lp_old, death_log_qforward, death_log_qbackward, logu = death_move(self.sg, self.wn, return_probs=True)
        print death_accepted, death_lp_new, death_lp_old, death_log_qforward, death_log_qbackward
        self.assertFalse(death_accepted)

        lp3 = self.sg.current_log_p()
        self.assertAlmostEqual(lp3, lp2)


        self.assertAlmostEqual(birth_lp_new-birth_lp_old, death_lp_old-death_lp_new)
        self.assertAlmostEqual(birth_log_qforward, death_log_qbackward)
        self.assertAlmostEqual(death_log_qforward, birth_log_qbackward)


if __name__ == '__main__':
    unittest.main()
