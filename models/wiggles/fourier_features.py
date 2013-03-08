import numpy as np
from sigvisa import Sigvisa
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle_models import WiggleModelNode

class FourierFeatureNode(WiggleModelNode):

    def __init__(self, fundamental, min_freq, max_freq, npts, srate, logscale=False, basisid=None, family_name=None, **kwargs):

        assert(min_freq < max_freq)

        self.srate = srate
        self.npts = npts
        self.len_s = float(npts-1) / srate
        self.x = np.linspace(0, self.len_s, self.npts)

        self.logscale = logscale

        self.fundamental = fundamental
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.nparams = 2 * int((self.max_freq - self.min_freq) / self.fundamental)

        self.family_name = family_name
        self.basisid = basisid
        if basisid is None:
            s = Sigvisa()
            self.save_to_db(s.dbconn)

        # load template params and initialize Node stuff
        super(FourierFeatureNode, self).__init__(**kwargs)

    def signal_from_features(self, features):
        features = np.reshape(features, (-1, 2))
        s = np.zeros((self.npts,))

        for (i, row) in enumerate(features):
            (amp, phase) = row
            freq = self.min_freq + self.fundamental * i
            s += amp * np.sin(2 * np.pi * (self.x * freq + phase))

        if self.logscale:
            s = np.exp(s)
        else:
            s += 1

        return s

    def basis_decomposition(self, signal):
        assert(len(signal) == self.npts)

        if self.logscale:
            signal = np.log(signal)
        else:
            signal = signal - 1 # we want zero-mean for the Fourier projection

        features = np.zeros((self.nparams/2, 2))
        for i in np.arange(self.nparams/2):
            freq = self.fundamental * i + self.min_freq

            basis1 = np.sin(self.x * 2 * np.pi * freq)
            basis2 = np.cos(self.x * 2 * np.pi * freq)

            c1 = np.dot(signal, basis1) / ((len(signal) - 1) / 2.0)
            c2 = np.dot(signal, basis2) / ((len(signal) - 1) / 2.0)

            if np.isnan(c1):
                c1 = 0
            if np.isnan(c2):
                c2 = 0

            c = complex(c1, c2)

            (amp, phase) = (np.abs(c), np.angle(c))
            phase /= (2 * np.pi)
            # print "freq = ", freq, "c = ", c, "amp/phase = ", amp, phase
            features[i, :] = (amp, phase)

        return features.flatten()

    def basis_type(self):
        return "fourier"

    def dimension(self):
        return self.nparams

    def save_to_db(self, dbconn):
        assert(self.basisid is None)
        sql_query = "insert into sigvisa_wiggle_basis (srate, logscale, family_name, basis_type, npts, dimension, fundamental, min_freq, max_freq) values (%f, '%s', '%s', '%s', %d, %d, %f, %f, %f)" % (self.srate, 't' if self.logscale else 'f', self.family_name, self.basis_type(), self.npts, self.dimension(), self.fundamental, self.min_freq, self.max_freq)
        self.basisid = execute_and_return_id(dbconn, sql_query, "basisid")
        return self.basisid


