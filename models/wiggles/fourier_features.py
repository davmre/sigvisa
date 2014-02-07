import numpy as np
from sigvisa import Sigvisa
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle_models import WiggleGenerator
from sigvisa.models import DummyModel
from sigvisa.models.distributions import Uniform, Gaussian

import scipy.weave as weave
from scipy.weave import converters


class FourierFeatureGenerator(WiggleGenerator):

    # features are:
    # fourier basis vectors in increasing order of frequency, from the fundamental up to the maximum
    # first n/2 entries: amplitudes, real-valued
    # second n/2 entries: phases, normalized between 0 and 1 (i.e. in radians / (2*pi)).

    def __init__(self, npts, srate, envelope, basisid=None, family_name=None, max_freq=None, **kwargs):

        self.srate = srate
        self.npts = npts
        assert( self.npts % 2 == 0 )

        if max_freq:
            if max_freq > srate/2.0:
                raise ValueError("wiggle frequency (%.2f) exceeds the Nyquist limit (%.2f)!" % (max_freq, srate/2.0))
            self.max_freq = max_freq
        else:
            self.max_freq = srate / 2.0

        self.fundamental = float(self.srate) / self.npts
        self.nparams = int(self.max_freq / self.fundamental) * 2

        self.len_s = float(npts-1) / srate
        self.x = np.linspace(0, self.len_s, self.npts)


        #print "created with envelope", envelope
        self.envelope = envelope

        self.family_name = family_name
        self.basisid = basisid
        if basisid is None:
            s = Sigvisa()
            self.save_to_db(s.dbconn)

        self.uamodel_amp = Gaussian(0, 0.1)
        self.uamodel_phase = Uniform(0, 1.0)

        freqs = [ ( n % (self.nparams/2) + 1) * self.fundamental for n in range(self.nparams)]
        self._params = ["amp_%.3f" % freq if i < self.nparams/2 else "phase_%.3f" % freq for (i, freq) in enumerate(freqs)]

        # load template params and initialize Node stuff
        super(FourierFeatureGenerator, self).__init__(**kwargs)

    def signal_from_features_naive(self, features):
        assert(len(features) == self.nparams)
        if isinstance(features, dict):
            features = self.param_dict_to_array(features)
        else:
            features= np.asarray(features)
        amps = features[:self.nparams/2]
        phases = features[self.nparams/2:] * 2.0 * np.pi

        s = np.zeros((self.npts,))

        for (i, (amp, phase)) in enumerate(zip(amps, phases)):
            freq = self.fundamental * (i+1)
            s += amp * np.cos(2 * np.pi * self.x * freq + phase)

        return s

    def signal_from_features(self, features):
        assert(len(features) == self.nparams)
        if isinstance(features, dict):
            features = self.param_dict_to_array(features)
        else:
            features= np.asarray(features)

        nparams = int(self.nparams)
        npts = int(self.npts)
        padded_coeffs = np.zeros((npts/2 + 1,), dtype=complex)
        twopi = 2*np.pi
        code =  """
int n2 = nparams/2;
padded_coeffs(0) = 0;
for (int i=0; i < n2; ++i) {
    double amp = features(i) * (npts/2);
    double phase = features(i+n2) *twopi;
    padded_coeffs(i+1) = std::complex<double>(amp * cos(phase), amp * sin(phase));
}
"""
        weave.inline(code,['twopi', 'nparams', 'npts', 'features', 'padded_coeffs'],type_converters =
               converters.blitz,verbose=2,compiler='gcc')

        signal = np.fft.irfft(padded_coeffs)

        return signal



    def features_from_signal(self, signal, return_array=False):
        assert(len(signal) == self.npts)

        coeffs = 2 * np.fft.rfft(signal) / self.npts
        coeffs = coeffs[1:self.nparams/2+1]

        amps = np.abs(coeffs)[: self.nparams/2]
        phases = (np.angle(coeffs)[: self.nparams/2] / (2.0* np.pi)) % 1.0
        if return_array:
            return np.concatenate([amps, phases])
        else:
            return self.array_to_param_dict(np.concatenate([amps, phases]))

    def features_from_signal_naive(self, signal):
        assert(len(signal) == self.npts)

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

            (amp, phase) = (np.abs(c), (np.angle(c) / (2 * np.pi)  ) % 1.0)
            amps.append(amp)
            phases.append(phase)

        return self.array_to_param_dict(np.concatenate([amps, phases]))

    def basis_type(self):
        return "fourier"

    def dimension(self):
        return self.nparams

    def params(self):
        return self._params

    def timeshift_param_array(self, a, t_shift):
        # loop over phase params
        dim = self.dimension()
        d2 = dim/2
        params = self.params()
        for i in range(d2):
            freq = self.fundamental * (i+1)
            a[i+d2] = (a[i+d2] + t_shift * freq) % 1.0

    def param_dict_to_array(self, d):
        a = np.asarray([ d[k] for k in self.params() ])
        return a

    def array_to_param_dict(self, a):
        d = {k : v for (k,v) in zip(self.params(), a)}
        return d

    def save_to_db(self, dbconn):
        assert(self.basisid is None)
        sql_query = "insert into sigvisa_wiggle_basis (srate, logscale, family_name, basis_type, npts, dimension, max_freq, envelope) values (%f, '%s', '%s', '%s', %d, %d, %f, '%s')" % (self.srate, 'f', self.family_name, self.basis_type(), self.npts, self.dimension(), self.max_freq, 't' if self.envelope else 'f')
        self.basisid = execute_and_return_id(dbconn, sql_query, "basisid")
        return self.basisid

    def unassociated_model(self, param, nm=None):
        if param.startswith("amp"):
            return self.uamodel_amp
        elif param.startswith("phase"):
            return self.uamodel_phase
        else:
            raise KeyError("unknown param %s" % param)
        #return DummyModel(default_value=0.0)
