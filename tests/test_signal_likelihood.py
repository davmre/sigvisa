import numpy as np
import numpy.ma as ma
import unittest

from sigvisa.signals.io import load_event_station
from sigvisa.source.event import get_event
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.graph.graph_utils import create_key
from sigvisa.plotting import plot

from sigvisa.tests.test_signals import savefig


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

        sg.set_value(key=create_key(param="arrival_time", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P'),
                     value=st+10.0)
        sg.set_value(key=create_key(param="peak_offset", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P',
                                    band='freq_2.0_3.0', chan='BHZ'),
                     value=15.0)
        sg.set_value(key=create_key(param="coda_height", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P',
                                    band='freq_2.0_3.0', chan='BHZ'),
                     value=10.0)
        sg.set_value(key=create_key(param="coda_decay", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P',
                                    band='freq_2.0_3.0', chan='BHZ'),
                     value=-.01)

        sg.set_value(key=create_key(param="arrival_time", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P'),
                     value=st+50.0)
        sg.set_value(key=create_key(param="peak_offset", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P',
                                    band='freq_2.0_3.0', chan='BHZ'),
                     value=15.0)
        sg.set_value(key=create_key(param="coda_height", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P',
                                    band='freq_2.0_3.0', chan='BHZ'),
                     value=15.0)
        sg.set_value(key=create_key(param="coda_decay", eid=self.event.eid,
                                    sta=self.wave['sta'], phase='P',
                                    band='freq_2.0_3.0', chan='BHZ'),
                     value=-.04)


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
