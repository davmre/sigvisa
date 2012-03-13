
import os, sys, traceback
import numpy as np, scipy


from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages

import plot
import learn, sigvisa_util
import priors.SignalPrior
from utils.waveform import *
from utils.plot_multi_station_params import *
import utils.geog
import obspy.signal.util

from utils.draw_earth import draw_events, draw_earth, draw_density
import utils.nonparametric_regression as nr
from utils.predict_envelope_shape import read_shape_data


params = dict()

def main():
    for fname in sys.argv[1:]:

        label = fname[13:24]
        outfile = 'logs/envelope_shape_dist_%s.pdf' % (label)
        pp = PdfPages(outfile)
        print "saving plots to", outfile
    
        l = read_shape_data(fname)

        print l.keys()

        try:
            for band in sorted(l.keys()):

                lbr = clean_points(l[band])

                band = band[19:]

                if lbr.shape[0] == 0:
                    continue
            
                siteid = int(lbr[0, SITEID_COL])
                print siteid
                phase = "P" if lbr[0, PHASEID_COL] == 1 else "S"
                key_base = "%d %s %s" % (siteid, phase, band)

                regional_bs = [lbr[i, B_COL] for i in range (lbr.shape[0]) if lbr[i, DISTANCE_COL] < 1000 ]
                tele_bs = [lbr[i, B_COL] for i in range (lbr.shape[0]) if lbr[i, DISTANCE_COL] >= 1000 ]
                

            #        print lbr
            #    print regional_bs
            #        print tele_bs
        
                if len(regional_bs) > 0:

                    plt.figure()
                    title = "sta %d phase %s regional %s (%d arrivals)" % (siteid, phase, band, len(regional_bs))
                    plt.title(title)
                    n, bins, patches = plt.hist(regional_bs, normed=1)
                    mu  = np.mean(regional_bs)
                    sigma = np.sqrt(np.var(regional_bs))
                    if sigma == 0:
                        sigma = 0.02
                    bincenters = 0.5*(bins[1:]+bins[:-1])
                    y = mlab.normpdf( bincenters, mu, sigma)
                    plt.plot(bincenters, y, 'r--', linewidth=1)
                    pp.savefig()
                    
                    params["r " + key_base] = [mu, sigma]

                if len(tele_bs) > 0:

                    plt.figure()
                    plt.title("sta %d phase %s tele %s (%d arrivals)" % (siteid, phase, band, len(tele_bs)))
                    n, bins, patches = plt.hist(tele_bs, normed=1)
                    mu  = np.mean(tele_bs)
                    sigma = np.sqrt(np.var(tele_bs))
                    if sigma == 0:
                        sigma = 0.02
                    bincenters = 0.5*(bins[1:]+bins[:-1])
                    y = mlab.normpdf( bincenters, mu, sigma)
                    plt.plot(bincenters, y, 'r--', linewidth=1)
                    pp.savefig()
                    
                    params["t " + key_base] = [mu, sigma]

        finally:
            pp.close()


    f = open("parameters/CodaDecay.txt", 'w')
    for k in sorted(params.keys()):
        f.write("%s %f %f\n" % (k, params[k][0], params[k][1]))
    f.close()

if __name__ == "__main__":
    main()
