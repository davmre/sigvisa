import numpy as np
import numpy.ma as ma
import cPickle

import scipy.weave as weave
from scipy.weave import converters
import time


from sigvisa.models import TimeSeriesDist
from sigvisa.models.noise.noise_model import NoiseModel

class ARModel(NoiseModel):

    # params: array of parameters
    # p: number of parametesr
    # em: error model
    # sf: sampling frequency in Hz.
    def __init__(self, params, em, c=0, sf=40):
        self.params = params
        self.p = len(params)
        self.em = em
        self.c = c
        self.sf = sf

        self.nomask = np.array([False,] * 5000, dtype=bool)

    # samples based on the defined AR Model
    def sample(self, n):
        data = np.zeros(n + self.p)

        # initialize with iid samples
        for t in range(self.p):
            data[t] = self.c + self.em.sample()

        for t in range(self.p, n+self.p):
            s = self.c
            for i in range(self.p):
                s += self.params[i] * (data[t - i - 1] - self.c)
            data[t] = s + self.em.sample()

        return data[self.p:]

    def fastAR_missingData(self, d, c, std, mask=None):
        n = len(d)

        if isinstance(d, ma.masked_array):
            mm = d.mask
            d = d.data - c

        else:
            d = d - c
            mm = mask

        try:
            mm[0]
            m = mm
        except (TypeError,IndexError,AttributeError):
            if len(d) > len(self.nomask):
                self.nomask = np.array([False,] * (len(self.nomask)*2), dtype=bool)
            m = self.nomask

        p = np.array(self.params)
        n_p = len(p)

        tmp = np.zeros((n_p, n_p))

        K = np.zeros((n_p, n_p))
        u = np.zeros((n_p,))

        code = """
        int t_since_mask = 0;
        double s = std;
        double t1 = log(s) + 0.5 * log(2 * 3.141592653589793);

        int converged_to_stationary = 0;

        double d_prob = 0;
        for (int t=0; t < n; ++t) {
            if (m(t)) {
               if (converged_to_stationary) continue;

               if (t_since_mask > n_p+1) {
                  // initialize kalman filtering when we encounter a masked zone

                  for (int i=0; i < n_p; ++i) {
                      for (int j=0; j < n_p; ++j) {
                           K(i,j) = 0;
                      }
                      u(i) = 0;
                  }
                  int len_history = t > n_p ? n_p : t;
                  for (int i=0; i < len_history; ++i) {
                      u(i) = d(t-1-i);
                  }
                  t_since_mask = 0;
               }

               double old_s = K(0,0);
               updateK(K, tmp, p, n_p);
               K(0,0) += std;
               update_u(u, p, n_p);

               if (old_s > std && (fabs(old_s - K(0,0)) < 0.00000000000001)) {
                  converged_to_stationary = 1;
               }

//               printf ("c t = %d std = %f old_s %f diff %f converged=%d\\n", t, K(0,0), old_s, fabs(old_s - K(0,0)), converged_to_stationary);

               continue;
          }


            double expected = 0;
            if (t_since_mask <= n_p) {
                converged_to_stationary = 0;

                t_since_mask++;
                updateK(K, tmp, p, n_p);
                K(0,0) += std;
                update_u(u, p, n_p);


                expected = u(0);
                s = K(0,0);
//                printf ("c t = %d std = %f\\n", t, K(0,0));
                t1 = log(s) + 0.5 * log(2 * 3.141592653589793);
            } else {
                for (int i=0; i < n_p; ++i) {
                    if (t > i && !m(t-i-1)) {
                       double ne = p(i) * d(t-i-1);
                       expected += ne;
                    }
                }
            }

            double err = d(t) - expected;
            double x = err/s;
            double ll = t1 + 0.5 * x*x;
            d_prob -= ll;

        // printf("cm %d err = %.10f x = %.10f ll = %.10f d_prob=%.10f\\n", t, err, x, ll, d_prob);

            if (t_since_mask <= n_p) {
                u(0) = d(t);
                for(int i=0; i < n_p; ++i) {
                    K(0, i) = 0;
                    K(i, 0) = 0;
                }
            } else if (t_since_mask == n_p + 1) {
                s = std;
                t1 = log(s) + 0.5 * log(2 * 3.141592653589793);
                t_since_mask += 1;
            }
        }

        return_val = d_prob;
        """

        support = """
        void updateK(blitz::Array<double, 2> &K, blitz::Array<double, 2> &tmp, blitz::Array<double, 1> p, int n_p) {

               // tmp = K * A.T
               for (int i=0; i < n_p; ++i){  // row of K

                 // first row of A contains the AR params
                 tmp(i,0) = 0;
                 for (int j=0; j < n_p; ++j){
                  tmp(i,0) += K(i,j) * p(j);
                 }

                 // all the other rows are just the shifted identity matrix
                 for (int j=1; j < n_p; ++j){
                    tmp(i,j) = K(i,j-1);
                 }
               }

               // K = A * tmp
               for (int i=0; i < n_p; ++i){  // col of tmp

                 // first row of A contains the AR params
                 K(0, i) = 0;
                 for (int j=0; j < n_p; ++j){
                  K(0, i) += tmp(j,i) * p(j);
                 }

                 // the other rows are the shifted identity
                 for (int j=1; j < n_p; ++j){
                    K(j,i) = tmp(j-1,i);
                 }
               }
             }


         void update_u(blitz::Array<double, 1> &u, blitz::Array<double, 1> &p, int n_p) {
               double newpred = 0;
               for (int i=0; i < n_p; ++i) {
                   newpred += u(i) * p(i);
               }
               for (int i=n_p-1; i > 0; --i) {
                   u(i) = u(i-1);
               }
               u(0) = newpred;
          }

"""

        ll = weave.inline(code,
                          ['n', 'n_p', 'm', 'd', 'std', 'p', 'tmp', 'K', 'u'],
                          type_converters=converters.blitz,
                          support_code=support,
                          compiler='gcc')
        return ll

    def fast_AR(self, d, c, std):
        #
        n = len(d)
        m = d.mask
        d = d.data - c
        p = np.array(self.params)
        n_p = len(p)
        t1 = np.log(std) + 0.5 * np.log(2 * np.pi)

        code = """
        double d_prob = 0;
        for (int t=0; t < n; ++t) {
            if (m(t)) { continue; }

           double expected = 0;
           for (int i=0; i < n_p; ++i) {
               if (t > i && !m(t-i-1)) {
                  double ne = p(i) * d(t-i-1);
                  expected += ne;
               }
           }
           double err = d(t) - expected;
           double x = err/std;
           double ll = (double)t1 + 0.5 * x*x;
           d_prob -= ll;
        // printf("c %d err = %.10f x = %.10f ll = %.10f d_prob=%.10f\\n", t, err, x, ll, d_prob);
        }
        return_val = d_prob;
        """
        ll = weave.inline(code,
                          ['n', 'n_p', 'm', 'd', 't1', 'std', 'p'],
                          type_converters=converters.blitz,
                          compiler='gcc')
        return ll


    def fastAR_grad(self, d, c, std):
        # return the gradient of the log-prob with respect to the signal at each timestep

        n = len(d)
        d = d - c
        p = np.asarray(self.params)
        n_p = len(p)
        var = std**2

        residuals = np.zeros(d.shape)
        grad = np.zeros(d.shape)

        code = """
        double d_prob = 0;
        for (int t=0; t < n; ++t) {

           double expected = 0;
           for (int i=0; i < n_p; ++i) {
               if (t > i ) {
                  double ne = p(i) * d(t-i-1);
                  expected += ne;
               }
           }
           residuals(t) = d(t) - expected;
        // printf("c %d err = %.10f x = %.10f ll = %.10f d_prob=%.10f\\n", t, err, x, ll, d_prob);
        }

        for (int t=0; t < n; ++t) {
            grad(t) -= residuals(t);
            for (int i=0; i < n_p; ++i) {
            // unlike above, i is counting *forward* from t here
                if (t+i+1 >= n) break;
                grad(t) += residuals(t+i+1) * p(i);
            }
            grad(t) /= var;
        }
        """
        weave.inline(code,
                          ['n', 'n_p', 'd', 'var', 'p', 'grad', 'residuals'],
                          type_converters=converters.blitz,
                          compiler='gcc')
        return grad





    def slow_AR(self, d, c, return_debug=False):
        d_prob = 0
        skipped = 0
        masked = ma.getmaskarray(d)
        skipped = 0

        n = len(d)
        m = d.mask
        d = d.data - c
        p = np.array(self.params)
        n_p = len(p)

        A = np.zeros((n_p, n_p))
        A[0, :] = p
        A[1:, :-1] = np.eye(n_p - 1)

        K = None

        u = None

        # update on missing data is
        # K = A*K*A^T + U

        orig_std = self.em.std
        std = orig_std
        t1 = np.log(std) + 0.5 * np.log(2 * np.pi)

        t_since_mask = np.float('inf')

        # suppose an AR2 process, we see t=100 is masked
        # when we get to t=100, we have all our covariances are zero
        # we then update to get K with a sigma^2 in the top right corner
        # now on t=101 we have time_since_mask = 1
        # now there are two components we need for t=101, namely the 100 and 99 components
        # now our prediction for 101 is u[0] after we update u
        # and our variance for 101 is K[0,0] after we update K
        # THEN we observe the real 101 and can zero out the 0th row and column of K,
        # and we can also update the 0th row of u.

        lls = []
        expecteds = []
        errors = []
        for t in range(len(d)):

            if masked[t]:

                # initialize kalman filtering when we encounter a masked zone
                if K is None:
                    K = np.zeros((n_p, n_p))
                    u = np.zeros((n_p,))
                    len_history = min(t, n_p)
                    u[:len_history] = d[t - 1:t - len_history - 1:-1]
                    t_since_mask = 0

                # for each masked timestep, do a Kalman update
                K = np.dot(A, np.dot(K, A.T))
                K[0, 0] += orig_std
                u = np.dot(A, u)
                continue

            # if we have just recently left a masked region, compute
            # process expectation and variance based on the Kalman
            # state.
            if t_since_mask <= n_p:
                t_since_mask += 1
                u = np.dot(A, u)
                K = np.dot(A, np.dot(K, A.T))
                K[0, 0] += orig_std

                expected = u[0]
                std = K[0, 0]
                t1 = np.log(std) + 0.5 * np.log(2 * np.pi)
            # otherwise, compute expectation in the normal way (variance is constant)
            else:
                expected = 0
                for i in range(n_p):
                    if t > i and not masked[t - i - 1]:
                        expected += p[i] * d[t - i - 1]

            actual = d[t]
            error = actual - expected
            t2 = 0.5 * np.square((error - self.em.mean) / std)
            ell = -t2 - t1

            if return_debug:
                lls.append(ell)
                expecteds.append(expected)
                errors.append(error)
            d_prob += ell

            # print "py t %d error %f s %f t2 %f ell %f d_prob %f" % (t, error, std, t2, ell, d_prob)

            # if we have recently left a masked region, continue
            # Kalman updating by adding an observation of the timestep
            # we just saw.
            if t_since_mask <= n_p:
                K[0, :] = 0
                K[:, 0] = 0
                u[0] = d[t]
            # if we are now sufficiently far from a masked region, stop Kalman updating
            elif t_since_mask == n_p + 1:
                K = None
                u = None
                std = orig_std
                t_since_mask += 1
                t1 = np.log(orig_std) + 0.5 * np.log(2 * np.pi)

        if return_debug:
            return d_prob, lls, expecteds, errors
        else:
            return d_prob

    # likelihood in log scale
    def log_p(self, x, zero_mean=False, **kwargs):
        data = x
        if not isinstance(data, (list, tuple)):
            data = [data, ]

        if zero_mean:
            c = 0
        else:
            c = self.c

        prob = 0
        for d in data:
            #if not isinstance(d, ma.masked_array):
            #    d = ma.masked_array(d, mask=[False,] * len(d))

            """
            t1 = time.time()
            d_prob = self.slow_AR(d, c)
            t2 = time.time()
            d_prob_fast_missing = self.fastAR_missingData(d, c, self.em.std)
            t3 = time.time()
            d_prob_fast = self.fast_AR(d, c, self.em.std)
            t4 = time.time()
            print (t2-t1), d_prob, (t3-t2), d_prob_fast_missing, (t4-t3), d_prob_fast
            """

            d_prob = self.fastAR_missingData(d, c, self.em.std, **kwargs)

            prob += d_prob

        return prob

    # likelihood in log scale
    def log_p_grad(self, x, zero_mean=False, **kwargs):
        if zero_mean:
            c = 0
        else:
            c = self.c
        return self.fastAR_grad(x, c, self.em.std, **kwargs)



    def location(self):
        return self.c

    def scale(self):
        return self.em.std

    def predict(self, n):
        return np.ones((n,)) * self.c

    # given data as argument,
    def errors(self, data):
        out = np.zeros(len(data) - self.p)
        for t in range(len(data) - self.p):
            expected = self.c
            for i in range(self.p):
                expected += self.params[i] * data[t + self.p - i - 1]
            actual = data[t + self.p]
            out[t] = actual - expected
        return out

    # returns optimal psd determined by the ar parameters, in log scale
    def psd(self, size=1024):
        params = self.params
        std = self.em.std
        ws = np.linspace(0, 0.5, size)
        S = np.zeros(size)
        p = len(params)

        for j in range(size):
            w = ws[j]
            sigma = 0
            for k in range(p):
                sigma += params[k] * np.exp(-2 * np.pi * w * complex(0, 1) * (k + 1))
            S[j] = 2 * (np.log(std) - np.log(np.abs(1 - sigma)))

        return (ws * self.sf, S)

    # returns residual sum of squares of log scale psd values
    def psdrss(self, psd, r=0.8):
        n = len(psd)
        S = self.psd(size=n)[1]
        rss = 0
        for i in range(int(float(n) * 0.8)):
            rss += np.square(psd[i] - S[i])
        return rss

    @staticmethod
    def load_from_file(fname):
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
            arm = ARModel(params, em, c=c, sf=srate)
        except Exception as e:
            raise Exception("error reading AR model from file %s: %s" % (fname, str(e)))
        finally:
            f.close()
        return arm


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

    def noise_model_type(self):
        return "ar"

    def order(self):
        return len(self.params)


# error model obeys normal distribution
class ErrorModel:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

        self.entropy = -.5*np.log(2*np.pi*np.e*std*std)

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.std)

    # likelihood in log scale
    def lklhood(self, x):
        t1 = np.log(self.std) + 0.5 * np.log(2 * np.pi)
        t2 = 0.5 * np.square((x - self.mean) / self.std)
        ll = -(t1 + t2)

        return ll
