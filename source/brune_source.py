# Implements a Brune source spectrum model for natural seismic events.
#
# these routines refer to:
# Sipkin, "A Correction to Body-wave Magnitude mb Based on Moment Magnitude MW". Seismological Research Letters, 2003.
# Myers et al, "Lithospheric-scale structure across the Bolivian Andes". Journal of Geophysical Research, 1998.

import numpy as np

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from source.common import *
# from sigvisa import Sigvisa

def short_band_to_hz(short_band):
    return np.median([float(x) for x in short_band.split('_')[1:]])


def mb_to_M0(mb):
    # eqn (2) in Sipkin
    MW = 1.48*mb - 2.42

    # eqn (5) in Myers
    M0 = np.exp((1.5*MW +9.1)*np.log(10))

    return M0


def source_freq_logamp(event, f, phase):

    # notation roughly follows eqn (2) in Fisk, and most constants are from Fisk as well

    P_phases = ['P', 'Pn']
    S_phases = ['S', 'Sn', 'Lg']

    M0 = mb_to_M0(event.mb)
    
    rho_s = 2700 # source density, kg/m^3
    rho_r = 2500 # receiver density, kg/m^3

    # choose velocities and other coefficients according to P vs S phase
    if phase in P_phases:
        R = 0.44  # radiation pattern coefficient
        v_s = 6100 # source velocity, m/s
        v_r = 5000 # receiver velocity, m/s

        # hard-code the value for this term since the fifth power is
        # annoying to compute explicitly.
        #sq = np.sqrt(rho_s * rho_r * v_s**5 * v_r)
        sq = 533901911953403.5
        K = 1.5 # corner frequency constant, from Myers
        cp = 0.41

    elif phase in S_phases:
        R = 0.60
        v_s = 3526
        v_r = 2890
        sq = 103111374869011.25
        K = 0.33
        cp = 0.49
    else:
        raise Exception("don't know how to compute source amplitude for phase %s" % (phase))
    # Fisk version of corner frequency calculation:
#    c = 0.41 if phase in P_phases else 0.49
#    sigma = 10 # assume stress drop of 10 bars
#    f_0 = c * v_s * (sigma / M0)**(1.0/3.0)

    # Myers version of coda frequency calculation:
    dsigma = 10.0 # assume stress drop of 10 bars
    #beta = 4.5 # S-wave velocity in km/s
    #f_0 = K * beta * (16 * dsigma / (7 * M0))**(1.0/3.0)
    # 1 bar = 10**5 N/m**2
    f_0 = cp * v_s*(dsigma*100000/(M0))**(1.0/3.0)

    F = R / (4*np.pi * sq)

    logS = (np.log(M0) + np.log(F)) / np.log(10)
    logShape = (np.log(1 + (f/f_0)**2   ) ) / np.log(10)
    logamp = logS - logShape

    return logamp, f_0, M0



def source_logamp(event, band, phase):
    f = band_to_hz(band)
    amp, corner, M0 = source_freq_logamp(event, f, phase)
    return amp

def main():

    cursor = db.connect().cursor()

    plot_source_spectra()
    sys.exit(1)

    parser = OptionParser()
    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str", help="siteid of station for which to learn source spectrum model (default: all)")
    parser.add_option("-r", "--runids", dest="runids", default=None, type="str", help="runid of the template fits to use")
    parser.add_option("-c", "--channels", dest="channels", default=None, type="str", help="channels (all)")
    parser.add_option("-o", "--outfile", dest="outfile", default="parameters/source_spectra.txt", type="str", help="filename to save output (parameters/source_spectra.txt)")
    (options, args) = parser.parse_args()

    runids = [int(r) for r in options.runids.split(',')]
    channels = chans if options.channels is None else [s for s in options.channels.split(',')]
    siteids = None if options.siteids is None else [int(s) for s in options.siteids.split(',')]
    runid_cond = "(" + " or ".join(["runid=%d" % r for r in runids])  + ")"
    print runid_cond

    f = open(options.outfile, 'w')

    ssm = SourceSpectrumModel()

    for (siteid, channel) in itertools.product(siteids, channels):
        pass


if __name__ == "__main__":
    main()
