import model
import numpy as np
import stat
import matplotlib.mlab

"""
learn given data via yule-walker of degree p
perform crossvaldiation
perform aic

"""

class ARLearner:
    def __init__(self, data, sf=40):
        self.data = data
        self.n = len(data)
        self.params = {}
        self.std = {}
        self.sf = sf
            
    #approximate params and std via yule-walker
    def yulewalker(self, p):
        if p in self.params:
            return (self.params[p], self.std[p])

        r = stat.autocorr(self.data)
        R = np.zeros([p,p])
        for row in range(p):
            for col in range(p):
                R[row][col] = r[np.abs(row-col)]
        
        r1 = np.zeros([p,1])
        for i in range(p):
            r1[i] = r[i+1]

        phi = np.dot(np.linalg.inv(R),r1)
        params = [phi[i][0] for i in range(p)]
        std = np.sqrt(r[0] - sum([params[k]*r[k+1] for k in range(p)]))
        self.params[p] = params
        self.std[p] = std
        return (params, std)
    
    def psd(self):
        y, x = matplotlib.mlab.psd(self.data, len(self.data), 1)
        return (x*self.sf, np.log(y))

    def segment(self, n, j):
        assert j < n
        l = len(self.data)
        return self.data[(j*l)/n:((j+1)*l)/n]
    
    def aic(self, p, const=500):
        lnr = ARLearner(self.data)
        params, std = lnr.yulewalker(p)
        em = model.ErrorModel(0, std)
        arm = model.ARModel(params, em)
        return 2*(len(self.data)/const)*p-2*arm.lklhood(self.data)


# n-fold crossvalidation
    def crossval(self, p, n=5):
        sum_L = 0.0
        l = len(self.data)
        d_segs = {}
        for i in range(n):
            d_segs[i] = self.segment(n, i)
    
        for i in range(n):
            training_d = d_segs[i]
            lnr = ARLearner(training_d)
            ar, std = lnr.yulewalker(p)
            em = model.ErrorModel(0, std)
            arm = model.ARModel(ar, em)
            for j in range(n):
                if i != j:
                    test_d = d_segs[j]
                    sum_L += arm.lklhood(test_d)
        return sum_L
    
    def psdcrossval(self, p, n=5):
        sum_L = 0.0
        l = len(self.data)
        # i_th data segment for training
        # the rest for calculating likelihood
        psds = {}
        for i in range(n):
            lnr = ARLearner(self.segment(n,i))
            psds[i] = lnr.psd()[1]
        
        for i in range(n):
            training_d = self.segment(n, i)
            lnr = ARLearner(training_d)
            params, std = lnr.yulewalker(p)
            em = model.ErrorModel(0,std)
            arm = model.ARModel(params, em)
            for j in range(n):
                if i != j:
                    test_psd = psds[j]
                    sum_L += arm.psdrss(test_psd)
        return np.sqrt(sum_L/((n-1)*l))