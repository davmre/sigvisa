import numpy as np
import numpy.ma as ma
import unittest

from sigvisa.signals.io import load_event_station
from sigvisa.source.event import get_event
from sigvisa.models.sigvisa_graph import SigvisaGraph
from sigvisa.plotting import plot

from sigvisa.tests.test_signals import savefig


class TestSignalLikelihood(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.wave = self.seg['BHZ']
        self.event = get_event(evid=5301405)

        self.sg = SigvisaGraph(phases = ['P', 'S'])
        self.sg.add_event(self.event)
        self.sg.add_wave(self.wave)

        st = self.seg['stime']

        tm_P_node = self.sg.get_template_node(ev=self.event, wave=self.wave, phase='P')
        tm_P_node.set_value(np.array((st + 10.0, 15.0, 10.0, -.01)))
        tm_S_node = self.sg.get_template_node(ev=self.event, wave=self.wave, phase='S')
        tm_S_node.set_value(np.array((st + 50.0, 15.0, 15.0, -.04)))


    def test_generate(self):

        wave_node = self.sg.get_wave_node(wave=self.wave)
        wave_node.prior_predict()

        template_wave = wave_node.get_wave()

        f = plot.plot_waveform(template_wave, logscale=True)
        savefig('template_new.png', f)


    def test_likelihood(self):

        wave_node = self.sg.get_wave_node(wave=self.wave)
        ll = wave_node.log_p()
        print ll

        wave_data = wave_node.get_value()
        wave_data[50:5000] = ma.masked
        wave_node.set_value(wave_data)
        ll_missing = wave_node.log_p()
        print "missing ll", ll_missing

        self.assertGreater(ll_missing, ll)

if __name__ == '__main__':
    unittest.main()
