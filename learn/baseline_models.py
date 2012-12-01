import numpy as np

X_LON, X_LAT, X_DEPTH, X_DIST, X_AZI  = range(5)

class ConstantModel(object):

    def __init__(self, X=None, y=None, fname=None):
        if fname is not None:
            self.load_trained_model(fname)
            return

        self.mean = np.mean(y)

    def save_trained_model(self, fname):
        np.savetxt(fname, [self.mean,])

    def load_trained_model(self, fname):

        with  open(fname, 'r') as f:
            self.mean = float(f.read())

    def predict(self, X1):
        X1 = np.asarray(X1)
        if len(X1.shape) == 1 or X1.shape[0] == 1:
            return self.mean
        n = X1.shape[1]
        return self.mean * np.ones((n, 1))


class LinearModel(object):

    def __init__(self, X=None, y=None, fname=None):

        if fname is not None:
            self.load_trained_model(fname)
            return

        n = len(y)

        self.tele_cutoff = 2000

        mean = np.mean(y)

        regional_dist=[]
        regional_y=[]
        tele_dist=[]
        tele_y=[]
        for i in range(n):
            d = X[i, X_DIST]
            dy = y[i]
            if d > self.tele_cutoff:
                tele_dist.append(d)
                tele_y.append(dy)
            else:
                regional_dist.append(d)
                regional_y.append(dy)

        regional_dist = np.array(regional_dist)
        tele_dist = np.array(tele_dist)

        try:
            regional_data = np.vstack([regional_dist,np.ones((len(regional_dist),))]).T
            self.regional_coeffs, residues, rank, sing = np.linalg.lstsq(regional_data, regional_y, 1e-6)
        except ValueError:
            self.regional_coeffs= np.array([0, mean])
        try:
            tele_data = np.vstack([tele_dist,np.ones((len(tele_dist),))]).T
            self.tele_coeffs, residues, rank, sing = np.linalg.lstsq(tele_data, tele_y, 1e-6)
        except ValueError:
            self.tele_coeffs = np.array([0, mean])

    def save_trained_model(self, fname):
        f = open(fname, 'w')
        f.write("regional %f %f\n" % tuple(self.regional_coeffs))
        f.write("tele %f %f\n" % tuple(self.tele_coeffs))
        f.write("cutoff_km %f\n" % self.tele_cutoff)
        f.close()

    def load_trained_model(self, fname):
        f = open(fname, 'r')
        regional_line = f.readline()
        tele_line = f.readline()
        cutoff_line = f.readline()
        f.close()

        self.regional_coeffs = [float(x) for x in regional_line.split()[1:]]
        self.tele_coeffs = [float(x) for x in tele_line.split()[1:]]
        self.tele_cutoff = float(cutoff_line.split()[1])

    def predict_item(self, x):
        d = x[X_DIST]
        if d > self.tele_cutoff:
            v = self.tele_coeffs[0]*d + self.tele_coeffs[1]
        else:
            v = self.regional_coeffs[0]*d + self.regional_coeffs[1]
        return v

    def predict(self, X1):
        X1 = np.asarray(X1)
        if len(X1.shape) == 1 or X1.shape[0] == 1:
            return self.predict_item(X1)

        results = np.array([self.predict_item(x) for x in X1])

        return results

