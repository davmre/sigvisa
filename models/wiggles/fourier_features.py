import numpy as np
from sigvisa.models.wiggles.featurizer import Featurizer


class FourierFeatures(Featurizer):

    def __init__(self, fundamental=.1, min_freq=0.8, max_freq=3.5, **kwargs):
        super(Featurizer, self).__init__(**kwargs)

        self.fundamental = fundamental
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.nparams = int((self.max_freq - self.min_freq) / self.fundamental) + 1 # one extra feature for std

    def signal_from_features(self, features, srate=None, len_seconds=None, npts=None):
        srate = srate if srate is not None else self.srate

        std = features[0]
        features = np.reshape(features[1:], (-1, 2))

        if npts is not None:
            len_seconds = npts/srate
        elif len_seconds is not None:
            npts = int(len_seconds * srate)
        else:
            raise ValueError("call to signal_from_features must specify either len_seconds or npts")

        x = np.linspace(0, len_seconds, npts)

        s = np.zeros((npts,))

        for (i, row) in enumerate(features):
            (amp, phase) = row

#            (c1, c2) = row
            freq = self.min_freq + self.fundamental * i
#            basis1  =  np.sin(x*2*np.pi*freq)
#            basis2  =  np.cos(x*2*np.pi*freq)

#            s += c1 * basis1
#            s += c2 * basis2

            s += amp * np.sin(x * 2 * np.pi * freq + phase)

        m = np.mean(s)
        s = (s - m) * std + m
        if self.logscale:
            s = np.exp(s)
        else:
            s += 1

        return s

    def basis_decomposition(self, signal, srate=None):
        srate = srate if srate is not None else self.srate

        if self.logscale:
            signal = np.log(signal)
        else:
            signal = signal - 1 # we want zero-mean for the Fourier projection

        std = np.std(signal)
        m = np.mean(signal)
        signal = (signal - m) / std + m

        n_features = int((self.max_freq - self.min_freq) / self.fundamental)
        len_seconds = len(signal) / float(srate)

        x = np.linspace(0, len_seconds, len(signal))
        assert(len(x) == len(signal))

        features = np.zeros((n_features, 2))
        for i in np.arange(n_features):
            freq = self.fundamental * i + self.min_freq

            periods = freq * len_seconds

            basis1 = np.sin(x * 2 * np.pi * freq)
            basis2 = np.cos(x * 2 * np.pi * freq)

            b1d = np.dot(basis1, basis1)
            b2d = np.dot(basis2, basis2)

            c1 = np.dot(signal, basis1) / len(signal)
            c2 = np.dot(signal, basis2) / len(signal)

            if np.isnan(c1):
                c1 = 0
            if np.isnan(c2):
                c2 = 0

#            features[i, :] = [c1, c2]

            c = complex(c1, c2)

            (amp, phase) = (np.abs(c), np.angle(c))
            features[i, :] = (amp, phase)

        return np.concatenate(( (std,), features.flatten()))

    def basis_type(self):
        return "fourier"

    def dimension(self):
        return self.nparams

    def save_to_db(self, dbconn):
        sql_query = "insert into sigvisa_wiggle_basis (srate, logscale, basis_type, dimension, fundamental, min_freq, max_freq) values (%f, '%s', '%s', %d, %f, %f, %f)" % (self.srate, 't' if self.logscale else 'f', self.basis_type(), self.dimension(), self.fundamental, self.min_freq, self.max_freq)
        return execute_and_return_id(dbconn, sql_query, "basisid")


