import unittest

import shutil, os

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa
from source.event import Event
from signals.common import Waveform, Segment, load_waveform_from_file
from signals.mask_util import *
from signals.io import load_event_station
from signals.template_models.paired_exp import PairedExpTemplateModel
from signals.armodel.model import ARModel, ErrorModel, load_armodel_from_file
from signals.armodel.learner import ARLearner
from signals.noise_model import model_path, construct_and_save_hourly_noise_models, get_noise_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting.plot


class TestMasks(unittest.TestCase):

    def setUp(self):
        self.mask1 = np.zeros((100,))
        self.mask1[2:20] = np.ones((18,))
        self.mask1[80:98] = np.ones((18,))

        self.mask2 = np.zeros((100,))
        self.mask2[0:20] = np.ones((20,))
        self.mask2[40:60] = np.ones((20,))
        self.mask2[80:100] = np.ones((20,))

    def test_grow_mask(self):

        goal_mask1 = np.zeros((100,))
        goal_mask1[0:24] = np.ones((24,))
        goal_mask1[76:100] = np.ones((24,))

        mask1 = grow_mask(self.mask1, 4)
        self.assertTrue( (mask1 == goal_mask1).all())

        goal_mask2 = np.zeros((100,))
        goal_mask2[0:24] = np.ones((24,))
        goal_mask2[36:64] = np.ones((28,))
        goal_mask2[76:100] = np.ones((24,))

        mask2 = grow_mask(self.mask2, 4)
        self.assertTrue( (mask2 == goal_mask2).all())

    def test_mask_blocks(self):
        blocks1 = mask_blocks(self.mask1)
        self.assertEqual( blocks1, [ (2, 20), (80, 98) ] )

        blocks2 = mask_blocks(self.mask2)
        self.assertEqual( blocks2, [ (0, 20), (40, 60), (80, 100) ] )

    def test_mirror(self):

        signal1 = range(100)
        ma1 = ma.masked_array(data=signal1, mask = self.mask1)
        ma1 = mirror_missing(ma1)
        self.assertTrue( (ma1.data == [ 0, 1, 1, 0, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 99, 98, 98, 99]).all() )

        signal2 = range(100)
        ma2 = ma.masked_array(data=signal2, mask = self.mask2)
        ma2 = mirror_missing(ma2)
        self.assertTrue( (ma2.data == [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60]).all() )



class TestWaveform(unittest.TestCase):

    def setUp(self):
        mask = np.zeros((1000,))
        mask[100:200] = np.ones((100,))

        self.data1 = ma.masked_array(np.random.randn(1000), mask)
        self.bhz = Waveform(data = self.data1, srate = 10, stime=11111103, sta="CTA", chan="BHZ")

    def test_segment_stats(self):
        self.assertEqual( self.bhz['sta'], "CTA")
        self.assertEqual( self.bhz['srate'], 10)
        self.assertEqual( self.bhz['npts'], 1000)
        self.assertEqual( self.bhz['stime'], 11111103)
        self.assertEqual( self.bhz['etime'], 11111203)

    def test_local_stats(self):
        self.assertEqual( self.bhz['chan'], "BHZ")
        self.assertEqual( self.bhz['filter_str'], "")
        self.assertEqual( self.bhz['freq_low'], 0.0)
        self.assertEqual( self.bhz['freq_high'], 5.0)

    def test_filters(self):
        bhz_23 = self.bhz.filter("freq_2.0_3.0")
        bhz_23_env = bhz_23.filter("env")
        bhz_23_env_smooth = bhz_23_env.filter("smooth")

        bhz_alt = self.bhz.filter("freq_2.0_3.0;env;smooth")
        self.assertIs(bhz_23_env_smooth, bhz_alt)

    def test_timeslice(self):
        a = self.bhz[0:100]


    def test_mirror_missing(self):

        self.bhz.data = mirror_missing(self.bhz.data)
        self.bhz.data.mask=np.zeros((1000,))


        plotting.plot.plot_waveform(self.bhz)
        plt.savefig("mirror")

    def test_obspy_trace(self):
        tr = self.bhz.as_obspy_trace()

    def test_pickle(self):
        self.bhz.dump_to_file("test_bhz.wave")
        loaded = load_waveform_from_file("test_bhz.wave")
        self.assertEqual(self.bhz.segment_stats, loaded.segment_stats)
        self.assertEqual(self.bhz.my_stats, loaded.my_stats)
        self.assertAlmostEqual(np.sum(self.bhz.data - loaded.data), 0)

class TestSegments(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)

        mask1 = np.zeros((1000,))
        mask1[000:100] = np.ones((100,))

        mask2 = np.zeros((1000,))
        mask2[600:700] = np.ones((100,))

        self.data1 = ma.masked_array(np.random.randn(1000), mask=mask1)
        self.data1.data[000:100] = np.ones((100,))* -100000000000000
        self.data2 = ma.masked_array(np.random.randn(1000), mask=mask2)
        self.data2.data[600:700] = np.ones((100,))* -100000000000000
        self.data3 = self.data2 + self.data1

        self.bhz = Waveform(data = self.data1, srate = 10, stime=103, sta="CTA", chan="BHZ")
        self.bhe = Waveform(data = self.data2, srate = 10, stime=103, sta="CTA", chan="BHE")
        self.bhn = Waveform(data = self.data3, srate = 10, stime=103, sta="CTA", chan="BHN")

        self.seg = Segment([self.bhz, self.bhe, self.bhn])

    def test_segment_stats(self):

        self.assertEqual( self.seg['sta'], "CTA")
        self.assertEqual( self.seg['srate'], 10)
        self.assertEqual( self.seg['npts'], 1000)
        self.assertEqual( self.seg['stime'], 103)
        self.assertEqual( self.seg['etime'], 203)

    def test_segment_filter_caching(self):
        filter_str = "freq_2.0_3.0;env;smooth"
        s1 = self.seg.with_filter(filter_str)
        s2 = self.seg.with_filter(filter_str)
        self.assertIs(s1, s2)

    def test_segment_filter_order(self):
        filter_str = "freq_2.0_3.0;env;smooth"
        reorder_str = "env;freq_2.0_3.0;smooth"

        partial_str1 = "freq_2.0_3.0"
        partial_str2 = "env;smooth"

        s1 = self.seg.with_filter(filter_str)
        s2 = self.seg.with_filter(reorder_str)
        s3 = self.seg.with_filter(partial_str1).with_filter(partial_str2)
        s4 = self.seg.with_filter(partial_str2).with_filter(partial_str1)

        self.assertIs(s1, s2)
        self.assertIs(s1, s3)
        self.assertIs(s1, s4)

        s5 = self.seg.with_filter(partial_str2)
        self.assertIsNot(s1, s5)

    def test_waveform_filter_caching_full(self):
        filter_str = "freq_2.0_3.0;env;smooth"
        s1 = self.seg.with_filter(filter_str)
        bhz_filtered = s1['BHZ']
        bhz_filtered_alt = self.seg['BHZ'].filter(filter_str)
        bhz_filtered_alt1 = self.seg['BHZ'].filter(filter_str)
        self.assertIs(bhz_filtered, bhz_filtered_alt)
        self.assertIs(bhz_filtered_alt, bhz_filtered_alt1)

    def test_waveform_filter_caching_partial(self):
        filter_str = "freq_2.0_3.0;env;smooth"
        bhz = self.seg['BHZ']

        bhz_full = bhz.filter(filter_str, preserve_intermediate=True)

        bhz_partial = bhz
        for single_filter in filter_str.split(';'):
            bhz_partial = bhz_partial.filter(single_filter)

        self.assertIs(bhz_full, bhz_partial)

    def test_old_style(self):
        old_seg = self.seg.as_old_style_segment()
        a =  self.seg.with_filter('freq_2.0_3.0')
        self.assertIs(a['BHN'].data, old_seg['BHN']['freq_2.0_3.0'].data)

    def test_load_plot(self):
        plotting.plot.plot_segment(self.seg)
        plt.savefig("fake")

        s = self.seg.with_filter('center;env')
        plotting.plot.plot_segment(s)
        plt.savefig("fake_env")

        s = s.with_filter('freq_2.0_3.0')
        plotting.plot.plot_segment(s)
        plt.savefig("fake_env_freq23")


class TestIO(unittest.TestCase):

    def test_load_plot(self):
        self.seg = load_event_station(evid=5301405, sta="URZ")

        self.assertEqual(self.seg['sta'], "URZ")

        bhn = self.seg['BHN']
        bhz = self.seg['BHZ']

        self.assertIs(bhz['event_arrivals'], bhn['event_arrivals'])
        self.assertIs(bhz['arrivals'], bhn['arrivals'])

        plotting.plot.plot_segment(self.seg)
        plt.savefig("URZ_5301405")

        s = self.seg.with_filter('center;env')
        plotting.plot.plot_segment(s)
        plt.savefig("URZ_5301405_env")

        s = s.with_filter('freq_2.0_3.0')
        plotting.plot.plot_segment(s)
        plt.savefig("URZ_5301405_env_2_3")

        s = s.with_filter('smooth')
        plotting.plot.plot_segment(s)
        plt.savefig("URZ_5301405_env_2_3_smooth")

class TestAutoregressiveModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_AR_learning(self):
        true_params = np.array([.8, .1])
        true_std = .11
        errormodel = ErrorModel(mean=0, std=true_std)
        true_model = ARModel(true_params, errormodel)
        sampled_data = true_model.sample(1000)

        learner = ARLearner(sampled_data)
        learned_params, learned_std = learner.yulewalker(2)

        self.assertAlmostEqual(np.sum(true_params - learned_params), 0, places=1)
        self.assertAlmostEqual(true_std, learned_std, places=1)

    def test_pickle(self):
        true_params = np.array([.8, .1])
        true_std = .11
        errormodel = ErrorModel(mean=0, std=true_std)
        true_model = ARModel(true_params, errormodel)
        true_model.dump_to_file("test.armodel")

        loaded_model = load_armodel_from_file("test.armodel")
        self.assertAlmostEqual(np.sum(true_model.params - loaded_model.params), 0)
        self.assertAlmostEqual(true_model.em.std, loaded_model.em.std)

class TestNoiseModels(unittest.TestCase):

    def test_train_noise_model(self):
        ev = Event(evid=5301405)

        sta='URZ'
        chan='BHZ'
        filter_str='freq_2.0_3.0;env'

        # delete existing saved noise models
        hour = (int(ev.time/3600)-1)
        hour_dir, model_fname = model_path(sta, chan, filter_str, srate=40, order=17, hour_time=hour*3600)
        try:
            shutil.rmtree(os.path.realpath(hour_dir))
        except OSError as e:
            pass
        try:
            os.remove(hour_dir)
        except OSError as e:
            pass

        # the first model we request should actually build models for all frequency bands
        s = Sigvisa()
        for freq in s.bands:
            # test that we can load models, and are not reconstructing them differently each time
            model1 = get_noise_model(sta=sta, chan=chan, filter_str=freq+';env', time=ev.time, srate=40, order=17)
            model2 = get_noise_model(sta=sta, chan=chan, filter_str=freq+';env', time=ev.time, srate=40, order=17)
            self.assertAlmostEqual(np.sum(np.asarray(model1.params) - np.asarray(model2.params)), 0)

        weird_band = 'freq_12_14'
        old_bands = s.bands
        s.bands = old_bands + (weird_band,)
        model1 = get_noise_model(sta=sta, chan=chan, filter_str=weird_band+';env', time=ev.time, srate=40, order=17)
        model2 = get_noise_model(sta=sta, chan=chan, filter_str=weird_band+';env', time=ev.time, srate=40, order=17)
        self.assertAlmostEqual(np.sum(np.asarray(model1.params) - np.asarray(model2.params)), 0)
        s.bands = old_bands

class TestSignalLikelihood(unittest.TestCase):

    def setUp(self):
        self.seg = load_event_station(evid=5301405, sta="URZ").with_filter('freq_2.0_3.0;env')
        self.event = Event(evid=5301405)
        self.tm =  PairedExpTemplateModel(run_name = "", model_type="dummy")

    def test_generate(self):
        st = self.seg['stime']
        param_vals = np.array( ((st+10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04))  )
        bhz_23_template =(('P', 'S'), param_vals)
        template = self.tm.generate_template_waveform(template_params=bhz_23_template, model_waveform = self.seg['BHZ'])

        sampled = self.tm.generate_template_waveform(template_params=bhz_23_template, model_waveform = self.seg['BHZ'], sample=True)

        plotting.plot.plot_waveform(template, logscale=True)
        plt.savefig('template.png')

        plotting.plot.plot_waveform(sampled, logscale=True)
        plt.savefig('sampled.png')

    def test_iid_cost(self):
        smoothed = self.seg.with_filter("smooth")
        st = self.seg['stime']
        param_vals = np.array( ((st+10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04))  )
        ll = self.tm.waveform_log_likelihood_iid(smoothed['BHZ'], (('P', 'S'), param_vals))
        print ll

        missing_bhz = self.seg.with_filter("smooth")['BHZ']
        missing_bhz.data[50:5000] = ma.masked
        ll_missing = self.tm.waveform_log_likelihood_iid(missing_bhz, (('P', 'S'), param_vals))
        print "missing iid ll", ll_missing

        self.assertGreater(ll_missing, ll)

    def test_likelihood(self):
        st = self.seg['stime']
        param_vals = np.array( ((st+10.0, 15.0, 10.0, -.01), (st + 50.0, 15.0, 15.0, -.04))  )
        ll = self.tm.waveform_log_likelihood(self.seg['BHZ'], (('P', 'S'), param_vals))
        print ll

        missing_bhz = self.seg['BHZ']
        missing_bhz.data[50:5000] = ma.masked
        ll_missing = self.tm.waveform_log_likelihood(missing_bhz, (('P', 'S'), param_vals))
        print "missing ll", ll_missing

        self.assertGreater(ll_missing, ll)

if __name__ == '__main__':
    unittest.main()
