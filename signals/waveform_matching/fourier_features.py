import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import learn.optimize

class FourierFeatures(object):

    def __init__(self, fundamental=.1, min_freq=0.8, max_freq=3.5, srate = 40):
        self.fundamental = fundamental
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.srate = 40

    def signal_from_features(self, features, len_seconds = 30):
        x = np.linspace(0, len_seconds, len_seconds*self.srate)

        s = np.zeros((len_seconds*self.srate,))

        for (i, row) in enumerate(features):
            print row
            (amp, phase) = row
            freq = self.fundamental*i 
            
            s += amp * np.sin(x*2*np.pi*freq + phase)
            
        return s
            
    def features_from_signal(self, signal):

        # initialize features from an fft
        fft = np.fft.rfft(signal)
        len_seconds = len(signal)/self.srate
        fft_fundamental = 2.0/len_seconds
        fund_ratio = self.fundamental / fft_fundamental
        n_features = int(self.max_freq/self.fundamental)
        features = np.zeros((n_features, 2))
        for i in np.arange(n_features):
            c = fft[int((i+1)*fund_ratio) -1]
            a = np.array((np.abs(c), np.angle(c)))
            features[i, :] = a

        print features
        s = self.signal_from_features(features, len_seconds = len_seconds)
        plt.plot(s)
        plt.show()

        fit_cost = lambda features : np.linalg.norm(signal - self.signal_from_features(features, len_seconds = len_seconds), 1)
        
        optim_features = learn.optimize.minimize_matrix(fit_cost, features, "bfgs")
                 
        s2 = self.signal_from_features(optim_features, len_seconds = len_seconds)
        print optim_features
        plt.plot(s2)
        plt.show()



def main():
    ff = FourierFeatures()
    wave = np.loadtxt("test.wave")
#    plt.plot(wave)
#    plt.show()
    f = ff.features_from_signal(wave)

if __name__ == "__main__":
    main()
        
            
