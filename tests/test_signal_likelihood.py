import numpy as np
import numpy.ma as ma
import unittest

from sigvisa.signals.io import load_event_station
from sigvisa.source.event import get_event
from sigvisa.models.templates.paired_exp import PairedExpTemplateModel
from sigvisa.models.wiggles.wiggle_models import PlainWiggleModel
from sigvisa.plotting import plot

from sigvisa.tests.test_signals import savefig


class TestSignalLikelihood(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.event = get_event(evid=5301405)
        self.tm = PairedExpTemplateModel(run_name="", model_type="dummy")
        self.wm = PlainWiggleModel(self.tm)

    def test_generate(self):
        st = self.seg['stime']
        param_vals = np.array(((st + 10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04)))
        bhz_23_template = (('P', 'S'), param_vals)
        template = self.tm.generate_template_waveform(template_params=bhz_23_template, model_waveform=self.seg['BHZ'])

# sampled =
# self.tm.generate_template_waveform(template_params=bhz_23_template,
# model_waveform = self.seg['BHZ'], sample=True)

        f = plot.plot_waveform(template, logscale=True)
        savefig('template.png', f)

#        plotting.plot.plot_waveform(sampled, logscale=True)
#        plt.savefig('sampled.png')

    def test_likelihood(self):
        st = self.seg['stime']
        param_vals = np.array(((st + 10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04)))

        ll = self.wm.template_ncost(wave=self.seg['BHZ'], phases=('P', 'S'), params=param_vals)
        print ll

        missing_bhz = self.seg['BHZ']
        missing_bhz.data[50:5000] = ma.masked
        ll_missing = self.wm.template_ncost(missing_bhz, phases=('P', 'S'), params=param_vals)
        print "missing ll", ll_missing

        self.assertGreater(ll_missing, ll)

if __name__ == '__main__':
    unittest.main()
