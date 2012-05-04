import unittest

import numpy as np
from obspy.core import Trace, Stream, UTCDateTime

import netvisa, sigvisa, sigvisa_util, learn
import priors.SignalPrior
from database.dataset import *
from database import db


def gen_random_segments(siteids, length):
    segments = []
    for siteid in siteids:
        segments.append(gen_random_segment(siteid, length))
    return segments

def gen_random_segment(siteid, length):
    channels = dict()
    for (chan, chanid) in (('BHE', 0), ('BHN', 1), ('BHZ', 2)):
        data = np.random.random((length, 1))
        stats = {'network': 's' + str(siteid), 'station': 's' + str(siteid), 'location': '',
                 'channel': chan, 'npts': len(data), 'sampling_rate': 5,
                 'starttime_unix': 10000, 'siteid': siteid, 'chanid': chanid, 'starttime': UTCDateTime(10000)}
        trace = Trace(data = data, header=stats)        
        channels[chan] = {"broadband_envelope": trace}
    return channels

class TestPurePythonFunctions(unittest.TestCase):
    def setUp(self):
        pass

 

class TestLoadFunctions(unittest.TestCase):
    def setUp(self):
        self.cursor = db.connect().cursor()
        self.start_time = 1237680000
        self.end_time = self.start_time + 900
        self.stalist = [2, 5]
        
    # Load a couple of brief waveforms from disk.
    def test_load_signals(self):
        segments = sigvisa_util.load_and_process_traces(self.cursor, self.start_time, self.end_time, stalist = self.stalist)

        self.assertTrue(len(segments) > 0)
        self.assertEqual(len(segments[0]), 3)

        self.assertTrue(isinstance(segments[0]["BHZ"]["broadband_envelope"], Trace))
        s = segments[0]["BHZ"]["broadband_envelope"]


        self.assertEqual(len(s.data), s.stats['npts'])
        self.assertTrue(len(s.data) > 0)

    def tearDown(self):
        self.cursor.close()

class TestCFunctions(unittest.TestCase):

    def setUp(self):
        self.hours = 1
        start_time, end_time, detections, leb_events, leb_evlist, sel3_events, \
            sel3_evlist, site_up, sites, phasenames, phasetimedef, arid2num \
            = read_data(hours=self.hours)

        self.sigmodel = learn.load_sigvisa("parameters", start_time, end_time, "envelope", site_up, sites, phasenames, phasetimedef, load_signal_params = False)
        self.test_siteids = (2,5)
        self.set_default_params()

    def set_default_params(self):
        site_params = {"chan_mean_BHZ": 0, "chan_var_BHZ": 1, "chan_mean_BHE": 0, "chan_var_BHE": 1, "chan_mean_BHN": 0, "chan_var_BHN": 1, "env_p_height": 1, "env_p_onset": 0.8, "env_p_decay": 0.04, "env_s_height": 1, "env_s_onset": 0.8, "env_s_decay": 0.04, "ar_noise_sigma2": 0.05, "ar_coeffs": (0.8,)}    
        self.params = dict()
        for siteid in self.test_siteids:
            self.params[siteid] = site_params.copy()
        self.sigmodel.set_all_signal_params(self.params)

    def test_arrival_likelihood(self):
        self.sigmodel.set_signals(gen_random_segments((2,), 1000))
        ll = self.sigmodel.arrival_likelihood(10500, 8, 360, 10, 0, 2, 0)
        self.assertTrue(ll < 0)
        

    # Generate a couple of random signals, pass them to Sigmodel to be
    # converted to C structures, and then try to retrieve them. We
    # want to verify that nothing crashes and we get out more or less
    # what we put in.
    def test_set_signals(self):

        for signal_len in (100, 10, 1000):
            segments_in = gen_random_segments(self.test_siteids, signal_len)
            self.sigmodel.set_signals(segments_in)
            segments_out = self.sigmodel.get_signals()
        
            self.assertEqual(len(segments_in), len(segments_out))
            for (seg_in, seg_out) in zip(segments_in, segments_out):
                self.assertEqual(len(seg_in), len(seg_out))
                for chan in seg_in.keys():
                    chan_in = seg_in[chan]
                    chan_out = seg_out[chan]
                    for band in chan_in.keys():
                        self.assertTrue((chan_in[band].data == chan_out[band].data).all())
                        for stat in ('npts', 'siteid', 'chanid', 'starttime_unix', 'sampling_rate'):
                            self.assertEqual(chan_in[band].stats[stat], chan_in[band].stats[stat])

#    def test_detection_likelihood(self):
#        pass

    def test_learn_noise_params(self):
        
        pass

    def test_learn_shape_params(self):
        # TODO: should really sample from the model, then verify that
        # what we learn fits the sample signal. right now we just
        # verify that we learning *something*.

        self.set_default_params()
        
        for siteid in self.test_siteids:
            self.sigmodel.set_fake_detections([ (1, siteid-1, 10005, 10, 1, 1, 1), ])
            self.sigmodel.set_signals(gen_random_segments((siteid,), 1000))
            params = priors.SignalPrior.learn_envelope_params(self.sigmodel, siteid, self.params)
            print params
            print 0.8, params[siteid]["env_p_onset"]
            self.assertNotEqual(params[siteid]["env_p_onset"], 0.8)

        self.set_default_params()

    def test_learn_wiggle_params(self):

        self.set_default_params()

        for siteid in self.test_siteids:
            self.sigmodel.set_fake_detections([ (1, siteid-1, 10005, 10, 1, 1, 1), ])
            self.sigmodel.set_signals(gen_random_segments((siteid,), 1000))
            params = priors.SignalPrior.learn_ar_params(self.sigmodel, siteid, self.params)
            print params
            self.assertNotEqual(params[siteid]["ar_noise_sigma2"], 0.05)
            self.assertNotEqual(params[siteid]["ar_coeffs"][0], 0.8)

        self.set_default_params()

    def test_birth_propose(self):
        pass

    def test_infer(self):
        pass

if __name__ == '__main__':
    unittest.main()
