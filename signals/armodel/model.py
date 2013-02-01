import numpy as np
import numpy.ma as ma
import cPickle

def load_armodel_from_file(fname):
    f = open(fname, 'r')
    try:
        srate = int(f.readline().split(" ")[1])
        c = float(f.readline().split(" ")[1])
        _ = f.readline()
        mean = float(f.readline().split(" ")[1])
        std = float(f.readline().split(" ")[1])
        _ = f.readline()
        _ = f.readline()
        params = []
        for line in f:
            params.append(float(line))
        em = ErrorModel(mean, std)
        arm = ARModel(params, em, c=c, sf=0)
    except Exception as e:
        raise Exception("error reading AR model from file %s: %s" % (fname, str(e)))
    finally:
        f.close()
    return arm
#    return cPickle.load(open(fname, 'rb'))

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
    def lklhood(self, data, zero_mean=False):
        if not isinstance(data, (list, tuple)):
            data = [data,]

        if zero_mean:
            c = 0
        else:
            mean = self.c

        prob = 0
        for d in data:
            d_prob = 0
            skipped = 0
            masked =  ma.getmaskarray(d)
            for t in range(len(d)):
                if masked[t]:
                    continue

                expected = c
                for i in range(self.p):
                    if t > i and not masked[t-i-1]:
                        expected += self.params[i] * (d[t-i-1] - c)
                actual = d[t]
                error = actual - expected
                ell = self.em.lklhood(error)
                d_prob += ell
            # normalize the sum of probability (no dependency on p value)
            #prob += d_prob/(len(d)-skipped)*len(d)
            prob += d_prob

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

    def dump_to_file(self, fname):
        f = open(fname, 'w')
        f.write("srate %d\n" % self.sf)
        f.write("c %.9f\n" % self.c)
        f.write("\n")
        f.write("mean %.9f\n" % self.em.mean)
        f.write("std %.9f\n" % self.em.std)
        f.write("\n")
        f.write("params\n")
        for p in self.params:
            f.write("%.8f\n" % p)
        f.close()

        #cPickle.dump(self, open(fname, 'wb'), protocol=0)


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
        ll = -(t1 + t2)

        return ll
