import os, errno, sys, time, traceback
import numpy as np
from scipy import stats


from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser

import plot
import sigvisa
import utils.geog
import obspy.signal.util
import itertools

from signals.coda_decay_common import *
from signals.armodel.learner import ARLearner
from signals.armodel.model import ARModel



def main():

    cursor = db.connect().cursor()

    parser = OptionParser()
    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str", help="siteid of station for which to learn wiggle model (default: all)")
    parser.add_option("-r", "--runids", dest="runids", default=None, type="str", help="runid of the extracted wiggles to use")
    parser.add_option("-p", "--phaseids", dest="phaseids", default=None, type="str", help="phaseids (P_PHASES)")
    parser.add_option("-c", "--channels", dest="channels", default=None, type="str", help="channels (all)")
    parser.add_option("-o", "--outfile", dest="outfile", default="parameters/signal_wiggles.txt", type="str", help="filename to save output (parameters/signal_wiggles.txt)")
    (options, args) = parser.parse_args()

    runids = [int(r) for r in options.runids.split(',')]
    phaseids = P_PHASEIDS if options.phaseids is None else [int(r) for r in options.phaseids.split(',')]
    channels = chans if options.channels is None else [s for s in options.channels.split(',')]
    siteids = None if options.siteids is None else [int(s) for s in options.siteids.split(',')]
    runid_cond = "(" + " or ".join(["runid=%d" % r for r in runids])  + ")"
    print runid_cond

    f = open(options.outfile, 'w')

    for (siteid, phaseid, channel, band) in itertools.product(siteids, phaseids, channels, bands):
        short_band = band[16:]

        sta = siteid_to_sta(siteid, cursor)
        phase = phaseid_to_name(phaseid)

        print sta, phase, channel, short_band

        sql_query = "select fname from sigvisa_wiggle_wfdisc where %s and siteid=%d and phaseid=%d and band='%s' and chan='%s'" % (runid_cond, siteid, phaseid, short_band, channel)
        print sql_query
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        wiggles = []
        for row in rows:
            fname = row[0]
            print " loading %s..." % fname
            w = np.loadtxt(fname)

            if len(w) > 500:
                wiggles.append(w)
        print "loaded %d wiggles." % len(wiggles)

        if len(wiggles) > 5:
            ar_learner = ARLearner(wiggles)
            params, std = ar_learner.yulewalker(20)
            params_str = str(len(params)) + " " + " ".join([str(p) for p in params])
            line = "%s %s %s %s %f %f %s" % (sta, phase, channel, short_band, ar_learner.c, std, params_str)
        else:
            line = "%s %s %s %s 1 0.042897 2 1.15979568629 -0.162206492945" % (sta, phase, channel, short_band)
        print "writing line", line
        f.write(line + "\n")
    f.close()

if __name__ == "__main__":
    main()
