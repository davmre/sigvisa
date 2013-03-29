import model
import numpy as np
import numpy.ma as ma
import stat
import matplotlib.mlab

"""
learn given data via yule-walker of degree p
perform crossvaldiation
perform aic

"""


class ARLearner:
    def __init__(self, data, sf=40):
        if not isinstance(data, (list, tuple)):
            data = [data, ]
        self.c = np.mean(np.concatenate(data))
        self.norm_data = [d - self.c for d in data]
        self.params = {}
        self.std = {}
        self.sf = sf

    def residual_std(self, p):
        params = self.params[p]
        res = []
        for nd in self.norm_data:
            for t in range(p, len(nd)):
                expected = sum([params[k] * nd[t - k - 1] for k in range(p)])
                actual = nd[t]
                res.append(expected - actual)
        res = np.array(res)
        std = np.std(res)

        # maximum absolute deviation: robust to outliers, scaled to be
        # a consistent estimator of Gaussian population std.
        madstd = 1.4826 * np.median(np.abs(res - np.mean(res)))
        return std

    # approximate params and mean/std via yule-walker
    def yulewalker(self, p):
        if p in self.params:
            return (self.params[p], self.std[p])

        r = stat.autocorr(self.norm_data, p)
        R = np.zeros([p, p])
        for row in range(p):
            for col in range(p):
                R[row][col] = r[np.abs(row - col)]

        r1 = np.zeros([p, 1])
        for i in range(p):
            r1[i] = r[i + 1]

        phi = np.dot(np.linalg.inv(R), r1)
        params = [phi[i][0] for i in range(p)]
        self.params[p] = params
        std = np.sqrt(r[0] - sum([params[k] * r[k + 1] for k in range(p)]))

#        std = self.residual_std(p)
        self.std[p] = std
        return (params, std)

    def segment(self, data, n, j):
        assert j < n
        l = len(data)
        return data[(j * l) / n:((j + 1) * l) / n]

    # use cross-validation to select the best model
    def cv_select(self, max_p=30, skip=2):
        best_ll = np.float('-inf')
        best_p = 0
        for p in np.array(range(0, max_p, skip)) + 1:
            ll = self.crossval(p)
            print " model order %d gives ll %f" % (p, ll)
            if ll > best_ll:
                best_ll = ll
                best_p = p

        ar, std = self.yulewalker(best_p)
        em = model.ErrorModel(0, std)
        return model.ARModel(ar, em, c=self.c)

    # n-fold crossvalidation
    def crossval(self, p, n=5):
        sum_L = 0.0

        for i in range(n):
            training_list = []
            test_list = []
            for d in self.norm_data:
                train_slice_list = []
                for j in range(n):
                    if j != i:
                        train_slice_list.append(self.segment(d, n, j))
                        #training_list.append(self.segment(d, n, j))
                training_list.append(np.concatenate(train_slice_list))
                test_list.append(self.segment(d, n, i))

            lnr = ARLearner(training_list)
            ar, std = lnr.yulewalker(p)
            em = model.ErrorModel(0, std)
            arm = model.ARModel(ar, em)
            sum_L += arm.log_p(test_list)

        """
        d = self.norm_data[0]
        l = len(d)
        train = d[0:int(l*.8)]
        test = d[int(l*.8):]

        lnr = ARLearner([train,])
        ar, std = lnr.yulewalker(p)
        em = model.ErrorModel(0, std)
        arm = model.ARModel(ar, em)
        sum_L = arm.log_p(test)"""

        return sum_L


    ############# WARNING: EVERYTHING BELOW IS BROKEN (since norm_data is now a list) ########
    def psd(self):
        y, x = matplotlib.mlab.psd(self.norm_data[0], len(self.norm_data[0]), 1)
        return (x * self.sf, np.log(y))

    def aic(self, p, const=500):
        lnr = ARLearner(self.norm_data)
        params, std = lnr.yulewalker(p)
        em = model.ErrorModel(0, std)
        arm = model.ARModel(params, em)
        return 2 * (len(self.norm_data) / const) * p - 2 * arm.log_p(self.norm_data)

    def psdcrossval(self, p, n=5):
        sum_L = 0.0
        l = len(self.norm_data)
        # i_th data segment for training
        # the rest for calculating likelihood
        psds = {}
        for i in range(n):
            lnr = ARLearner(self.segment(n, i))
            psds[i] = lnr.psd()[1]

        for i in range(n):
            training_d = self.segment(n, i)
            lnr = ARLearner(training_d)
            params, std = lnr.yulewalker(p)
            em = model.ErrorModel(0, std)
            arm = model.ARModel(params, em)
            for j in range(n):
                if i != j:
                    test_psd = psds[j]
                    sum_L += arm.psdrss(test_psd)
        return np.sqrt(sum_L / ((n - 1) * l))
