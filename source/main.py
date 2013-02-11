import os
import sys

import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sigvisa.source import brune_source, mm_source
from sigvisa.source.event import get_event


def plot_source_spectra():

    # fname = os.path.join(base_coda_dir, "spectra_%.1f.pdf" % (mb,))
    fname = os.path.join("logs", "spectra.pdf")
    pp = PdfPages(fname)

    ev = Event(mb=4.5, depth=0.122, lon=0, lat=0, time=0, natural_source=True)
    print brune_source.source_freq_logamp(ev, 2.5, 'P')
    print mm_source.source_freq_logamp(ev, 2.5, 'P')

    for mb in [4.5, ]:
        ev = Event(mb=mb, depth=0.122, lon=0, lat=0, time=0, natural_source=True)

        freqs = np.logspace(-1, 1, 40)

#        freqs = [1,]

        eq_spectra_P = [brune_source.source_freq_logamp(ev, f, 'P')[0] for f in freqs]
        eq_spectra_S = [brune_source.source_freq_logamp(ev, f, 'S')[0] for f in freqs]
        ex_spectra_P = [mm_source.source_freq_logamp(ev, f, 'P')[0] for f in freqs]
        ex_spectra_S = [mm_source.source_freq_logamp(ev, f, 'S')[0] for f in freqs]

        corner = brune_source.source_freq_logamp(ev, 1, 'P')[1]
        print "brune P corner", corner
        corner = brune_source.source_freq_logamp(ev, 1, 'S')[1]
        print "brune S corner", corner

        corner = mm_source.source_freq_logamp(ev, 1, 'P')[1]
        print "mm P corner", corner
        corner = mm_source.source_freq_logamp(ev, 1, 'S')[1]
        print "mm S corner", corner

        plt.title("Brune P/S source spectra, mb=4.5 (X axis is natural log of freq in Hz)")
        plt.plot(np.log(freqs), eq_spectra_P)
        plt.plot(np.log(freqs), eq_spectra_S)
        pp.savefig()
        plt.clf()

        plt.title("MM P/S source spectra, mb=4.5  (X axis is natural log of freq in Hz)")
        plt.plot(np.log(freqs), ex_spectra_P)
        plt.plot(np.log(freqs), ex_spectra_S)
        pp.savefig()
        plt.clf()

    pp.close()


def main():

    plot_source_spectra()
    sys.exit(1)


if __name__ == "__main__":
    main()
