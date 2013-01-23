import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import learn.optimize
import scipy.io
import cStringIO
from signals.waveform_matching.featurizer import Featurizer

class FourierFeatures(Featurizer):

    def __init__(self, fundamental=.1, min_freq=0.8, max_freq=3.5, srate = None):
        self.fundamental = fundamental
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.srate = srate

    def signal_from_features(self, features, srate=None, len_seconds = 30):
        srate = srate if srate is not None else self.srate
        
        x = np.linspace(0, len_seconds, len_seconds*srate)

        s = np.zeros((len_seconds*srate,))

        for (i, row) in enumerate(features):
            (amp, phase) = row
            

#            (c1, c2) = row

            freq = self.min_freq + self.fundamental*i 
#            basis1  =  np.sin(x*2*np.pi*freq)
#            basis2  =  np.cos(x*2*np.pi*freq)
            
#            s += c1 * basis1
#            s += c2 * basis2

            s += amp * np.sin(x*2*np.pi*freq + phase)
            

        s = s/np.std(s) - np.mean(s)
        return s


    def basis_decomposition(self, signal, srate=None):
        srate = srate if srate is not None else self.srate
        
        n_features = int((self.max_freq - self.min_freq)/self.fundamental)
        len_seconds = len(signal)/float(srate)

        x = np.linspace(0, len_seconds, len_seconds*srate)

        features = np.zeros((n_features, 2))
        for i in np.arange(n_features):
            freq = self.fundamental*i + self.min_freq

            periods = freq*len_seconds

            basis1  =  np.sin(x*2*np.pi*freq)
            basis2  =  np.cos(x*2*np.pi*freq)


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


        return features         


def main():
    ff = FourierFeatures(fundamental=1/20.0, min_freq=0.8, max_freq=3.5)


#    wave = np.loadtxt("test.wave")
    
    features = np.zeros((35, 2))
    features[5,:] = [1, 1]
    wave = ff.signal_from_features(features, len_seconds = 30)

    wave = wave/np.std(wave) - np.mean(wave)
    plt.figure()
    plt.plot(wave)

    f = ff.basis_decomposition(wave)
    plt.figure()
    plt.plot(f[:, 0])
#    plt.plot(features[:, 0])

    print f
#    print features

    plt.figure()
    s = ff.signal_from_features(f, len_seconds = len(wave) / 40)
    plt.plot(s)
    
    plt.figure()
    plt.plot(s)
    plt.plot(wave)

    offby = s/wave
    print "off by", np.mean(offby), np.median(offby)

    plt.show()

if __name__ == "__main__":
    main()
        
            
