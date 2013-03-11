import numpy as np
from sigvisa import Sigvisa
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle_models import WiggleModelNode

class FourierFeatureNode(WiggleModelNode):

    def __init__(self, npts, srate, logscale=False, basisid=None, family_name=None, max_freq=None, **kwargs):

        self.srate = srate
        self.npts = npts
        assert( self.npts % 2 == 0 )

        if max_freq:
            self.max_freq = max_freq
        else:
            self.max_freq = srate / 2.0

        self.fundamental = float(self.srate) / self.npts
        self.nparams = int(self.max_freq / self.fundamental) * 2

        self.len_s = float(npts-1) / srate
        self.x = np.linspace(0, self.len_s, self.npts)

        self.logscale = logscale

        self.family_name = family_name
        self.basisid = basisid
        if basisid is None:
            s = Sigvisa()
            self.save_to_db(s.dbconn)

        # load template params and initialize Node stuff
        super(FourierFeatureNode, self).__init__(**kwargs)

    def signal_from_features_naive(self, features):
        assert(len(features) == self.nparams)
        amps = features[:self.nparams/2]
        phases = features[self.nparams/2:] * 2.0 * np.pi

        s = np.zeros((self.npts,))

        for (i, (amp, phase)) in enumerate(zip(amps, phases)):
            freq = self.fundamental * (i+1)
            s += amp * np.cos(2 * np.pi * self.x * freq + phase)

        if self.logscale:
            s = np.exp(s)
        else:
            s += 1

        return s

    def signal_from_features(self, features):
        assert(len(features) == self.nparams)
        amps = features[:self.nparams/2] * (self.npts/2.0)
        phases = features[self.nparams/2:] * 2.0 * np.pi
        coeffs = amps * np.cos(phases) + 1j * amps * np.sin(phases)

        padded_coeffs = np.zeros((self.npts/2.0 + 1,), dtype=complex)
        padded_coeffs[1:len(coeffs)+1] = coeffs

        signal = np.fft.irfft(padded_coeffs)

        if self.logscale:
            signal = np.exp(signal)
        else:
            signal += 1

        return signal

    def basis_decomposition(self, signal):
        assert(len(signal) == self.npts)

        if self.logscale:
            signal = np.log(signal)

        coeffs = 2 * np.fft.rfft(signal) / self.npts
        coeffs = coeffs[1:self.nparams/2+1]

        amps = np.abs(coeffs)[: self.nparams/2]
        phases = np.angle(coeffs)[: self.nparams/2] / (2.0* np.pi)
        return np.concatenate([amps, phases])

    def basis_decomposition_naive(self, signal):
        assert(len(signal) == self.npts)

        if self.logscale:
            signal = np.log(signal)
        else:
            signal = signal - 1 # we want zero-mean for the Fourier projection

        amps = []
        phases = []
        for i in np.arange(self.nparams/2):
            freq = self.fundamental * (i+1)

            basis1 = np.cos(self.x * 2 * np.pi * freq)
            basis2 = -1 * np.sin(self.x * 2 * np.pi * freq)

            c1 = np.dot(signal, basis1) / ((len(signal))/ 2.0)
            c2 = np.dot(signal, basis2) / ((len(signal))/ 2.0)

            if np.isnan(c1):
                c1 = 0
            if np.isnan(c2):
                c2 = 0

            c = complex(c1, c2)

            (amp, phase) = (np.abs(c), np.angle(c) / (2 * np.pi))
            amps.append(amp)
            phases.append(phase)

        return np.concatenate([amps, phases])

    def basis_type(self):
        return "fourier"

    def dimension(self):
        return self.nparams


    def param_num_to_name(self, param_num):
        ptype = "amp" if param_num < self.nparams/2 else "phase"
        freq = ((param_num % (self.nparams/2)) + 1) * self.fundamental
        return "%s_%.3f" % (ptype, freq)

    def param_name_to_num(self, param_name):
        freq = float(param_name.split('_')[1])
        nfreq = np.round(freq / self.fundamental) - 1
        if param_name.startswith("amp_"):
            param_num = nfreq
        elif param_name.startswith("phase_"):
            param_num = self.nparams/2 + nfreq
        else:
            raise ValueError("invalid param name %s" % param_name)

        return int(param_num)

    def save_to_db(self, dbconn):
        assert(self.basisid is None)
        sql_query = "insert into sigvisa_wiggle_basis (srate, logscale, family_name, basis_type, npts, dimension, max_freq) values (%f, '%s', '%s', '%s', %d, %d, %f)" % (self.srate, 't' if self.logscale else 'f', self.family_name, self.basis_type(), self.npts, self.dimension(), self.max_freq)
        self.basisid = execute_and_return_id(dbconn, sql_query, "basisid")
        return self.basisid


