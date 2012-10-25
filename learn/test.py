import unittest

import numpy as np
import numpy.ma as ma

from sigvisa import Sigvisa

from source.event import Event

from signals.common import Waveform, Segment
from signals.io import load_event_station
from signals.template_model import ExponentialTemplateModel

from learn.fit_shape_params import fit_event_segment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting.plot


class TestFit(unittest.TestCase):

    def test_fit_template(self):

        s = Sigvisa()

        event = Event(evid=5301405)

        siteid = s.name_to_siteid_minus1['URZ'] + 1

        fit_event_segment(event.as_tuple(), siteid, runid=-1, init_runid=None, plot=False, wiggles=None)



if __name__ == '__main__':
    unittest.main()
