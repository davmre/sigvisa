import numpy as np
import numpy.ma as ma
import unittest

from sigvisa.signals.io import load_event_station
from sigvisa.source.event import get_event
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.graph_utils import create_key
from sigvisa.plotting import plot

from sigvisa.plotting.plot import savefig


class TestSignalLikelihood(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.wave = self.seg['BHZ']
        self.event = get_event(evid=5301405)

        sg = SigvisaGraph(phases = ['P', 'S'])
        sg.add_wave(self.wave)
        sg.add_event(self.event)
        self.sg = sg


        st = self.seg['stime']

        p_params = {'arrival_time': st + 10.0, 'peak_offset': 15.0, 'coda_height': 10.0, 'coda_decay': -.01}
        s_params = {'arrival_time': st + 50.0, 'peak_offset': 15.0, 'coda_height': 15.0, 'coda_decay': -.04}
        sg.set_template(eid=self.event.eid, sta=self.wave['sta'],
                        phase='P', band='freq_2.0_3.0', chan='BHZ',
                        values=p_params)
        sg.set_template(eid=self.event.eid, sta=self.wave['sta'],
                        phase='S', band='freq_2.0_3.0', chan='BHZ',
                        values=s_params)

    def test_generate(self):
        wave_node = self.sg.get_wave_node(wave=self.wave)
        wave_node.unfix_value()
        wave_node.parent_predict()

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
