import numpy as np

class ARModel:
    #params: array of parameters
    #p: number of parametesr
    #em: error model
    #sf: sampling frequency in Hz.
    def __init__(self, params, em, c=0, sf=40):
        self.params = params
        self.p = len(params)
        self.em = em
        self.c = c
        self.sf = sf

    #samples based on the defined AR Model
    #if init data is not given, then the first p points are
    #sampled from error model (i.e. normally distributed)
    def sample(self, num, initdata=[]):
        data = np.zeros(num)
        for t in range(num):
            if t < self.p:
                if len(initdata) == 0:
                    data[t] = self.em.sample()
                else:
                    assert len(initdata) == self.p
                    data[t] = initdata[t]
            else:
                s = self.c
                for i in range(self.p):
                    s += self.params[i] * data[t-i-1]
                data[t] = s + self.em.sample()
        return data

    # likelihood in log scale
    def lklhood(self, data):
        prob = 0
        for t in range(len(data)):
            expected = self.c
            for i in range(self.p):
                if t > i:
                    expected += self.params[i] * (data[t-i-1] - self.c)
            actual = data[t]
            error = actual - expected
            prob += self.em.lklhood(error)
        # normalize the sum of probability (no dependency on p value)
#        return prob/(len(data)-self.p)*len(data)
        return prob

    #given data as argument,
    def errors(self, data):
        out = np.zeros(len(data)-self.p)
        for t in range(len(data)-self.p):
            expected = self.c
            for i in range(self.p):
                expected += self.params[i] * data[t+self.p-i-1]
            actual = data[t+self.p]
            out[t] = actual - expected
        return out

    #returns optimal psd determined by the ar parameters, in log scale
    def psd(self, size=1024):
        params = self.params
        std = self.em.std
        ws = np.linspace(0,0.5,size)
        S = np.zeros(size)
        p = len(params)

        for j in range(size):
            w = ws[j]
            sigma = 0
            for k in range(p):
                sigma += params[k]*np.exp(-2*np.pi*w*complex(0,1)*(k+1))
            S[j] = 2*(np.log(std)-np.log(np.abs(1-sigma)))

        return (ws*self.sf,S)

    #returns residual sum of squares of log scale psd values
    def psdrss(self, psd, r=0.8):
        n = len(psd)
        S = self.psd(size=n)[1]
        rss = 0
        for i in range(int(float(n)*0.8)):
            rss += np.square(psd[i] - S[i])
        return rss


# error model obeys normal distribution
class ErrorModel:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(loc = self.mean, scale = self.std)

    # likelihood in log scale
    def lklhood(self, x):
        t1 = np.log(self.std) + 0.5 * np.log(2 * np.pi)
        t2 = 0.5 * np.square((x - self.mean) / self.std)

        return -(t1 + t2)