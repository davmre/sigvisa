import numpy as np

from sigvisa.source.common import *

# from sigvisa import Sigvisa


def source_logamp(mb, band, phase):
    f = band_to_hz(band)

    amp, corner = source_freq_logamp(mb, f, phase)

    return amp


def source_freq_logamp(mb, f, phase):

    # Generally follows Fisk (2006)

    P_phases = ['P', 'Pn', "PcP", "pP", "Pg"]
    S_phases = ['S', 'Sn', "ScP", "Lg"]

    # P and S wave source velocities in granite, m/s
    V_sP = 5500.0
    V_sS = 3175.0

    # gravitational constant, m/s^2
    g = 9.81

    # angular frequency as a function of frequency
    w = 2 * np.pi * f

    # yield as a function of MB
    W = np.exp(((mb - 4.45) / .75) * np.log(10))

    # scaled depth as a function of yield
    # h = 122 * W**(1.3/3.0)
    # The scaling relationship should be
    h = 122 * pow(W, 1.0 / 3)
    # h = event.depth * 1000

    # medium density in granite kg/m^3
    rho = 2550

    # elastic radius, for granite, shale, or tuff
    R_e = 162325 * W ** (1.0 / 3.0) * (
        rho * g * h) ** -0.417  # the initial coefficients here are magic constants inferred from the calibration event given in Fisk
    w_1 = 10914.0 / R_e

    # angular corner frequency (depends on phase type following "Fisk conjecture")
    V_s = V_sP if phase in P_phases else V_sS
    w_0 = V_s / R_e

    # cavity radius
    c = 3500.0  # magic number from MM71, maybe not good
    R_c = c * W ** (0.29) * h ** (-0.11)

    # peak pressure
    mu = 6780  # magic number from inverting calibration in Fisk, probably not good
    P_0 = (4.0 * mu / 3.0) * (R_c / R_e) ** 3
    P_p = 1.5 * rho * g * h

    gamma = (V_sP / (2 * V_sS)) ** 2.0

    # compute log-amplitude; eqn (5) in Fisk
    top = gamma * P_p * R_e * np.sqrt(w ** 2 + (w_1 * P_0 / P_p) ** 2)
    bottom = (rho * V_sP * np.sqrt(w ** 2 + w_1 ** 2) * np.sqrt((w_0 ** 2 - gamma * w ** 2) ** 2 + w_0 ** 2 * w ** 2))
    logamp = (np.log(top) - np.log(bottom)) / np.log(10)

    # other notes from fisk:
    # might be necssary to increase the P corner frequency by 30% for larger events (above mb 5.0 or so).
    # various site-specific corrections in elastic radii due to source medium

    f_0 = w_0 / (2.0 * np.pi)

    return logamp, f_0


def main():

    cursor = db.connect().cursor()

    plot_source_spectra()
    sys.exit(1)

    parser = OptionParser()
    parser.add_option("-s", "--siteids", dest="siteids", default=None, type="str",
                      help="siteid of station for which to learn source spectrum model (default: all)")
    parser.add_option("-r", "--runids", dest="runids", default=None, type="str", help="runid of the template fits to use")
    parser.add_option("-c", "--channels", dest="channels", default=None, type="str", help="channels (all)")
    parser.add_option("-o", "--outfile", dest="outfile", default="parameters/source_spectra.txt", type="str",
                      help="filename to save output (parameters/source_spectra.txt)")
    (options, args) = parser.parse_args()

    runids = [int(r) for r in options.runids.split(',')]
    channels = chans if options.channels is None else [s for s in options.channels.split(',')]
    siteids = None if options.siteids is None else [int(s) for s in options.siteids.split(',')]
    runid_cond = "(" + " or ".join(["runid=%d" % r for r in runids]) + ")"
    print runid_cond

    f = open(options.outfile, 'w')

    ssm = SourceSpectrumModel()

    for (siteid, channel) in itertools.product(siteids, channels):
        pass


if __name__ == "__main__":
    main()
