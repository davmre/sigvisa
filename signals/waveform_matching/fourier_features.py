import numpy as np


class FourierFeatures(object):

    def __init__(self, fundamental=.1, max_freq=3.5, srate = 40):
        self.fundamental = fundamental
        self.max_freq = max_freq
        self.srate = 40

    def signal_from_features(self, features, len_seconds = 30):
        x = np.linspace(0, len_seconds, len_seconds*self.srate)

        s = np.zeros((len_seconds*self.srate,))

        for (i, feature_pair) in self.features:
            freq = self.fundamental*i 
            
            (amp, phase) = feature_pair
            s += amp * np.sin(x*2*np.pi*freq + phase)
            
        return s
            
    def features_from_signal(self, signal):
        fft = np.fft.rfft(signal)

        
            
