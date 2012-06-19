import unittest

import numpy as np
from obspy.core import Trace, Stream, UTCDateTime

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plot

import sigvisa, sigvisa_util, learn
import signals.armodel.model, signals.armodel.learner
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
                 'channel': chan, 'band': 'narrow_envelope_2.00_3.00', 'npts': len(data), 'sampling_rate': 1,
                 'starttime_unix': 10000, 'siteid': siteid, 'chanid': chanid, 'starttime': UTCDateTime(10000)}
        trace = Trace(data = data, header=stats)
        channels[chan] = {"narrow_envelope_2.00_3.00": trace}
    return channels

class TestPurePythonFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_learn_armodel(self):
        arparams = np.array((1.531985598646, -1.0682484475528535, 1.0396481745808401, -1.3279255479118346, 0.98655767845516618, -0.83922136571517214, 0.76677157354780778, -0.59579319975231027, 0.36945613335446836, -0.17841016307209667))
        model = signals.armodel.model.ARModel(arparams, signals.armodel.model.ErrorModel(0, .1), c=0)
        s1 = model.sample(100000)
        s2 = model.sample(1000)
        s3 = model.sample(55555)
        ar_learner = signals.armodel.learner.ARLearner([s1,s2,s3], 40)
        params, std = ar_learner.yulewalker(10)

        for v in (np.array(params)-arparams):
            self.assertAlmostEqual(v, 0, places=1)

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

        self.sigmodel = learn.load_sigvisa("parameters", start_time, end_time, "spectral_envelope", site_up, sites, phasenames, phasetimedef, load_signal_params = False)
        self.test_siteids = (2,5)
        self.set_default_params()

    def set_default_params(self):
        arparams = np.array((1.531985598646, -1.0682484475528535, 1.0396481745808401, -1.3279255479118346, 0.98655767845516618, -0.83922136571517214, 0.76677157354780778, -0.59579319975231027, 0.36945613335446836, -0.17841016307209667))
        noise_mean = 0
        noise_var = .001
        wiggle_var = .001

        self.ar_noise_model = signals.armodel.model.ARModel(arparams, signals.armodel.model.ErrorModel(0, np.sqrt(noise_var)), c=noise_mean)

        band = sigvisa.canonical_band_num("narrow_envelope_2.00_3.00")
        chan_BHZ = sigvisa.canonical_channel_num("BHZ")
        chan_BHE = sigvisa.canonical_channel_num("BHE")
        chan_BHN = sigvisa.canonical_channel_num("BHN")
        chan_ha = sigvisa.canonical_channel_num("horiz_avg")

        for siteid in self.test_siteids:
            for chan in [chan_BHZ, chan_BHE, chan_BHN, chan_ha]:
                self.sigmodel.set_noise_process(siteid, band, chan, noise_mean, noise_var, arparams);
                self.sigmodel.set_wiggle_process(siteid, band, chan, 1, noise_mean, wiggle_var, arparams);


    def test_noise_trace_likelihood(self):
        segments = gen_random_segments((2,), 40)
        params = np.array(((100000, 1, 0, 1, 0, -0.02),))

        ll1 = self.sigmodel.trace_likelihood(segments[0]['BHZ']['narrow_envelope_2.00_3.00'], [1,], params)
        ll2 = self.ar_noise_model.lklhood(segments[0]['BHZ']['narrow_envelope_2.00_3.00'].data)
        print ll1, ll2
        self.assertAlmostEqual(ll1, ll2, places=1)

    def test_segment_likelihood(self):
        segments = gen_random_segments((2,), 1000)
        params = np.array(((10200, 4, 7, 2, 5, -0.02),))

        ll = self.sigmodel.segment_likelihood(segments[0], [1,], params)
        print "got ll", ll
        self.assertTrue(ll < 0)

    def test_sample_segment(self):
        params = np.array(((10200, 4, 7, 0, 7, -0.02),))
        seg_template = self.sigmodel.generate_segment(10000, 11000, 2, 1, [1,], params)
        seg_sample = self.sigmodel.sample_segment(10000, 11000, 2, 1, [1,], params)

        pp = PdfPages(os.path.join('logs', "test_segments.pdf"))
        plot.plot_segment(seg_template, title="generated", band="narrow_envelope_2.00_3.00")
        pp.savefig()
        plot.plot_segment(seg_sample, title="sampled", band="narrow_envelope_2.00_3.00")
        pp.savefig()
        pp.close()

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
