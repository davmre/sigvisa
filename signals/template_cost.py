"""

Code for computing the cost/likelihood of an envelope template with respect to a waveform.

"""


import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils.geog
import obspy.signal.util

from signals.coda_decay_common import *

import sigvisa_c


def set_dummy_wiggles(sigmodel, tr, phaseids):
    c = sigvisa_c.canonical_channel_num(tr.stats.channel)
    b = sigvisa_c.canonical_band_num(tr.stats.band)
    for pid in phaseids:
        sigmodel.set_wiggle_process(tr.stats.siteid, b, c, pid, 1, 0.05, np.array([.8,-.2]))

def set_noise_process(sigmodel, tr):
    c = sigvisa_c.canonical_channel_num(tr.stats.channel)
    b = sigvisa_c.canonical_band_num(tr.stats.band)
    arm = tr.stats.noise_model
    sigmodel.set_noise_process(tr.stats.siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))

def set_noise_processes(sigmodel, seg):
    for chan in seg.keys():
        c = sigvisa_c.canonical_channel_num(chan)
        for band in seg[chan].keys():
            b = sigvisa_c.canonical_band_num(band)
            siteid = seg[chan][band].stats.siteid
            try:
                arm = seg[chan][band].stats.noise_model
            except KeyError:
#                print "no noise model found for chan %s band %s, not setting.." % (chan, band)
                continue
            sigmodel.set_noise_process(siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))






