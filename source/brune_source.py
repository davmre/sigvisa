import numpy as np

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from coda_decay_common import *
from database.dataset import *

def short_band_to_hz(short_band):
    return np.median([float(x) for x in short_band.split('_')])


class SourceSpectrumModel:

    # these routines refer to:
    # Sipkin, "A Correction to Body-wave Magnitude mb Based on Moment Magnitude MW". Seismological Research Letters, 2003.
    # Myers et al, "Lithospheric-scale structure across the Bolivian Andes". Journal of Geophysical Research, 1998.




    def mb_to_M0(self, mb):
        # eqn (2) in Sipkin
        MW = 1.48*mb - 2.42

        # eqn (5) in Myers
        M0 = np.exp(1.5*MW +9.1)

        return M0


    def corner_freq(self, mb, phaseid):

        # following eqn (4) in Myers

        if phaseid not in P_PHASEIDS and phaseid not in S_PHASEIDS:
            raise Exception("unknown phaseid %d" % phaseids)

        dsigma = 10.0 # assume stress drop of 10 bars
        beta = 4.5 # S-wave velocity in km/s
        K = 1.5 if phaseid in P_PHASEIDS else 0.33
        M0 = self.mb_to_M0(mb)

        f0 = K * beta**3 * np.sqrt(16 * dsigma / (7 * M0))

        return f0

    def source_logamp(self, mb, phaseid, short_band=None, bandid=None):

        if short_band is None:
            short_band = bandid_to_short_band(bandid)

        f = short_band_to_hz(short_band)
        f0 = self.corner_freq(mb, phaseid)
        M0 = self.mb_to_M0(mb)
        W0 = 1.0 * M0 # should really be F*M0 where F is a
                      # phase-specific constant describing the
                      # radiation pattern, but we fold this into the
                      # learned transfer function instead

        return self.brune_logamp(W0, f, f0)

    def brune_logamp(self, W0, f, f0):
        return np.log(W0) -  2* np.log(1.0 + (f/f0))

    def fit_brune(self, spectrum_dict):

        fv = [(short_band_to_hz(sb), spectrum_dict[sb]) for sb in spectrum_dict.keys()]
        cost = lambda (f0, lW0): np.sum([(v - self.brune_logamp(np.exp(lW0), f, f0))**2 for (f, v) in fv])

        (f0, lW0) = scipy.optimize.fmin(func = cost, x0 = [1, 10])
        print "best params ", f0, lW0, "give cost", cost((f0, lW0))

        return (f0, np.exp(lW0))


def plot_spectrum_dict(pp, spectrum_dict):
    xyl = sorted([(np.log(short_band_to_hz(sb)), spectrum_dict[sb], sb) for sb in spectrum_dict.keys()])
    x = [a[0] for a in xyl]
    y = [a[1] for a in xyl]
    labels = [a[2] for a in xyl]

    plt.plot(x, y)
    plt.xticks(x, labels)

def plot_spectrum_params(pp, f0, W0):
    ssm = SourceSpectrumModel()
    x = [x for x in np.linspace(np.log(0.5), np.log(6), 80)]
    y = [ssm.brune_logamp(W0, np.exp(lf), f0) for lf in x]
    plt.plot(x, y)

    ticks = np.linspace(np.log(0.5), np.log(6), 6)
    plt.xticks(ticks, ["%.2f" % t for t in np.exp(ticks)])

def plot_spectrum(pp, spectrum_dict=None, title="", fits=None):

    f = plt.figure()
    plt.title(title)

    if spectrum_dict is not None:
        plot_spectrum_dict(pp, spectrum_dict)
    if fits is not None:
        plot_spectrum_params(pp, fits[0], fits[1])

    pp.savefig()
    plt.close(f)

def plot_source_spectra(siteid=58, runid=3):

    cursor = db.connect().cursor()
    ssm = SourceSpectrumModel()

    short_bands = [b[16:] for b in bands]
    chan="BHZ"

    stuff = load_shape_data(cursor, chan=chan, runid=runid, short_band="4.00_6.00")
    evids = set(stuff[:, FIT_EVID])

    for evid in evids:

        ev = load_event(cursor, evid)
        evid_stuff = filter_shape_data(stuff, evids=[evid,])
        phaseids = set(evid_stuff[:, FIT_PHASEID])

        base_coda_dir = get_base_dir(int(siteid), int(runid))
        fname = os.path.join(base_coda_dir, "spectra_%d.pdf" % (evid,))
        pp = None

        print "phaseids", phaseids

        for phaseid in phaseids:

            source_spec = dict()
            reciever_spec = dict()
            for short_band in short_bands:

                source_spec[short_band] = ssm.source_logamp(ev[EV_MB_COL], phaseid, short_band)

    #            sql_query = "select fit.coda_height from leb_origin lebo, leb_assoc leba, leb_arrival l, sigvisa_coda_fits fit, static_siteid sid, static_phaseid pid where fit.arid=l.arid and l.arid=leba.arid and leba.orid=lebo.orid and leba.phase=pid.phase and sid.sta=l.sta and sid.id=%d and lebo.evid=%d and pid.id=%d and fit.chan='%s'" % (siteid, evid, phaseid, chan)

                plot=True
                try:
                    sdata = load_shape_data(cursor, chan=chan, short_band=short_band, runid=runid, phaseids=[phaseid,], evids=[evid,])
                    print "got", sdata, "for", short_band, phaseid
                    reciever_spec[short_band] = sdata[0][FIT_CODA_HEIGHT]
                except:
                    plot=False
                    break

            if plot:
                if pp is None:
                    pp = PdfPages(fname)

                corner = ssm.corner_freq(ev[EV_MB_COL], phaseid)
                plot_spectrum(pp, title="source site %d evid %d phase %d\n mb %f corner %f" % (siteid, evid, phaseid, ev[EV_MB_COL], corner), fits=(corner, 1.0))
                (f0, W0) = ssm.fit_brune(reciever_spec)
                plot_spectrum(pp, spectrum_dict=reciever_spec, title="recieved site %d evid %d phase %d\n mb %f f0 %f W0 %f" % (siteid, evid, phaseid, ev[EV_MB_COL], f0, np.log(W0)), fits=(f0, W0))

        if pp is not None:
            pp.close()

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
