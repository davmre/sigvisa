import unittest

import numpy as np




class TestWaveform(unittest.TestCase):

    def setUp(self):
        self.data1 = np.random.randn(1000)
        self.bhz = Waveform(data = self.data1, srate = 10, stime=103, sta="CTA", chan="BHZ")

    def test_retrieve_data(self):
        self.assertEqual( self.bhz[0:2], self.data1[0:2])

    def test_segment_stats(self):
        self.assertEqual( self.bhz['sta'], "CTA")
        self.assertEqual( self.bhz['srate'], 10)
        self.assertEqual( self.bhz['npts'], 1000)
        self.assertEqual( self.bhz['stime'], 103)
        self.assertEqual( self.bhz['etime'], 203)

    def test_local_stats(self):
        self.assertEqual( self.bhz['chan'], "SELF.BHZ")
        self.assertEqual( self.bhz['filtering'], "")
        self.assertEqual( self.bhz['freq_low'], 0.0)
        self.assertEqual( self.bhz['freq_high'], 5.0)



    def test_filters(self):
        bhz_23 = self.bhz.filter("freq_2_3")
        bhz_23_env = bhz_23.filter("env")
        bhz_23_env_smooth = bhz_23_env.filter("smooth")

        bhz_alt = self.bhz.filter("freq_2_3;env;smooth")
        assertIdentity(bhz_23_env_smooth, bhz_alt)

    def test_obspy_trace(self):
        tr = self.bhz.as_obspy_trace()

class TestSegments(unittest.TestCase):

    def setUp(self):
        
        self.data1 = np.random.randn(1000)
        self.data2 = np.random.randn(1000)
        self.data3 = data1 + data2

        self.bhz = Waveform(data = self.data1, srate = 10, stime=103, sta="CTA", chan="BHZ")
        self.bhe = Waveform(data = self.data2, srate = 10, stime=103, sta="CTA", chan="BHE")
        self.bhn = Waveform(data = self.data3, srate = 10, stime=103, sta="CTA", chan="BHN")

        self.seg = Segment([bhz, bhe, bhn])

    def test_segment_stats(self):

        self.assertEqual( self.seg['sta'], "CTA")
        self.assertEqual( self.seg['srate'], 10)
        self.assertEqual( self.seg['npts'], 1000)
        self.assertEqual( self.seg['stime'], 103)
        self.assertEqual( self.seg['etime'], 203)

    def test_filters(self):
        filter_str = "freq_2_3;env;smooth"
        s1 = self.seg.filter(filter_str)
        bhz_filtered = s1['bhz']
        bhz_filtered_alt = self.seg['bhz'].filter(filter_str)

        assertIdentity(bhz_filtered, bhz_filtered_alt)

    def test_old_style(self):
        old_seg = self.seg.as_old_style_segment()
        assertEqual(old_seg['bhn'].data == self.seg['bhz'].data) 

if __name__ == '__main__':
    unittest.main()
