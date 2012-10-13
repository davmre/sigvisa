import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from signals.common import Waveform, Segment


class TestWaveform(unittest.TestCase):

    def setUp(self):
        mask = np.zeros((1000,1))
        mask[27] = 1
        self.data1 = ma.masked_array(np.random.randn(1000), mask)
        self.bhz = Waveform(data = self.data1, srate = 10, stime=103, sta="CTA", chan="BHZ")

    def test_retrieve_data(self):
        self.assertTrue( (self.bhz[0:2] == self.data1[0:2] ).all())

    def test_segment_stats(self):
        self.assertEqual( self.bhz['sta'], "CTA")
        self.assertEqual( self.bhz['srate'], 10)
        self.assertEqual( self.bhz['npts'], 1000)
        self.assertEqual( self.bhz['stime'], 103)
        self.assertEqual( self.bhz['etime'], 203)

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

    def test_obspy_trace(self):
        tr = self.bhz.as_obspy_trace()

class TestSegments(unittest.TestCase):

    def setUp(self):

        self.data1 = ma.masked_array(np.random.randn(1000))
        self.data2 = ma.masked_array(np.random.randn(1000))
        self.data3 = self.data1 + self.data2

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
        self.assertIs(s1['BHZ'], s2['BHZ'])

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


if __name__ == '__main__':
    unittest.main()
