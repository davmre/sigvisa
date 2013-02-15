import unittest

import shutil
import os

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa
from sigvisa.source.event import get_event
from sigvisa.signals.common import Waveform, Segment, load_waveform_from_file
from sigvisa.models.noise.noise_model import *


class TestNoiseModels(unittest.TestCase):

    def setUp(self):
        self.s = Sigvisa()

        self.ev = get_event(evid=5301405)

        self.sta = 'URZ'
        self.chan = 'BHZ'
        self.filter_str = 'freq_2.0_3.0;env'

        s = self.s
        ev = self.ev
        sta = self.sta
        chan = self.chan
        filter_str = self.filter_str

    def test_immediate_noise_model(self):
        s = self.s
        ev = self.ev
        sta = self.sta
        chan = self.chan
        filter_str = self.filter_str

        model1 = get_noise_model(
            sta=sta, chan=chan, filter_str=filter_str, time=ev.time, srate=40, order=17, force_train=True)
        self.assertAlmostEqual(model1.c, 0.3324, places=2)

    """"
    def _test_across_several_hours(self):
        t_start = 1238917955 - 6 * 3600
        t_max = t_start + 14 * 24 * 3600

        t = t_start
        f = open('means.txt', 'w')
        fs = open('stds.txt', 'w')
        for d in range(14):
    #    for d in range(2,3):
            for h in range(24):
    #        for h in range(3, 4):
                t = t_start + d * 24 * 3600 + h * 3600
                try:
                    nm = get_noise_model(sta="URZ", chan="BHZ", time=t, filter_str="freq_2.0_3.0;env")
                    print "nm for time", t, " has mean", nm.c
                    f.write("%f, " % nm.c)
                    fs.write("%f, " % nm.em.std)
                except NoNoiseException as e:
                    print e
                    f.write(", ")
                    fs.write(", ")
                    continue

            f.write("\n")
            fs.write("\n")
        f.close()
        fs.close()
        """


if __name__ == '__main__':
    unittest.main()
