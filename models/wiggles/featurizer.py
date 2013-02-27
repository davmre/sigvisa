import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sigvisa.infer.optimize.optim_utils
import scipy.io
import cStringIO


def load_basis(basisid):
    sql_query = "select srate, logscale, basis_type, dimension, fundamental, min_freq, max_freq, training_set_fname, training_sta, training_chan, training_band, training_phase, basis_fname from sigvisa_wiggle_basis where basisid=%d" % (basisid,)
    cursor.execute(sql_query)
    basis_info = cursor.fetchone()
    logscale = (basis_info[1].lower().startswith('t'))

    if basis_info[2] == "fourier":
        featurizer = FourierFeatures(fundamental = basis_info[4], min_freq=basis_info[5], max_freq=basis_info[6], srate = basis_info[0], logscale=logscale)
    else:
        raise NotImplementedError('wiggle basis %s not yet implemented' % basis_info[2])

    return featurizer

class Featurizer(object):

    def __init__(self, srate=None, logscale=False):
        self.srate = srate
        self.logscale = logscale

    def encode_params_from_signal(self, signal, srate):
        # normalize signal and get params
        params = self.basis_decomposition(signal=nsignal, srate=srate)

        # save the params to string in Matlab format
        output = cStringIO.StringIO()
        scipy.io.savemat(output, {"params": params}, oned_as='row')
        ostr = output.getvalue()
        output.close()
        return ostr

    def signal_from_encoded_params(self, encoded, srate, len_seconds=30):
        input = cStringIO.StringIO(encoded)
        d = scipy.io.loadmat(input)
        params = d['params']

        input.close()

        signal = self.signal_from_features(params, srate=srate, len_seconds=len_seconds)
        return signal

    def signal_from_features(self, features, srate=None, len_seconds=30):
        raise NotImplementedError("abstract base class!")

    def basis_decomposition(self, signal, srate=None):
        raise NotImplementedError("abstract base class!")

    def project_down(self, signal, srate=None):
        features = self.basis_decomposition(signal)
        return self.signal_from_features(features, len_seconds=len(signal) / self.srate)

    def cost(self, signal, features, srate=None):
        return np.linalg.norm(signal - self.signal_from_features(features, len_seconds=len(signal) / self.srate), 1)

    def save_to_db(self):
        raise NotImplementedError("abstract base class!")
