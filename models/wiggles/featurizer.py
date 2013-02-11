import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sigvisa.infer.optimize.optim_utils
import scipy.io
import cStringIO

class Featurizer(object):

    def encode_params_from_signal(self, signal, srate):
        # normalize signal and get params
        m = np.mean(signal)
        s = np.std(signal)
        nsignal = (signal - m)/s
        params = self.basis_decomposition(signal=nsignal, srate=srate)
        norm = np.array((m, s))

        # save the params to string in Matlab format
        output = cStringIO.StringIO()
        scipy.io.savemat(output, {"params": params, "norm": norm}, oned_as='row')
        ostr = output.getvalue()
        output.close()
        return ostr

    def signal_from_encoded_params(self, encoded, srate, len_seconds = 30):
        input = cStringIO.StringIO(encoded)
        d = scipy.io.loadmat(input)
        params = d['params']
        m,s = d['norm'][0]

        input.close()

        signal = self.signal_from_features(params, srate=srate, len_seconds=len_seconds)
        return signal * s + m

    def signal_from_features(self, features, srate=None, len_seconds = 30):
        raise Exception("not implemented!")

    def basis_decomposition(self, signal, srate=None):
        raise Exception("not implemented!")

    def project_down(self, signal, srate=None):
        features = self.basis_decomposition(signal)
        return self.signal_from_features(features, len_seconds=len(signal)/self.srate)

    def cost(self, signal, features, srate=None):
        return np.linalg.norm(signal - self.signal_from_features(features, len_seconds = len(signal) / self.srate), 1)
