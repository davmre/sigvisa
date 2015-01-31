import numpy as np
import scipy.stats

def conjugate(M, left_multiply_X, MX, result):
    # given an NxN matrix M, and an NxD matrix X,
    # return the DxD matrix X*M*X'.
    # Here X is expressed implicitly as two functions:
    # left_multiply_Xt(v) = X*v
    # right_multiply_X(v) = v*X
    # if these functions are linear-time, then the whole
    # operation is just quadratic (as opposed to cubic in
    # a naive implementation).
    #
    # if arrays MX (NxD) and result (DxD) are specified, they
    # are used as temporary storage, otherwise, new storage
    # will be allocated

    N, _ = M.shape
    D = result.shape[0]

    for i, row in enumerate(M):
        left_multiply_X(row, MX[i,:])
    for i, col in enumerate(MX.T):
        left_multiply_X(col, result[:,i])

    return result

def conjugate_scalar(M, left_multiply_X, MX=None):
    N, _ = M.shape

    if MX is None:
        MX = np.empty((N,))

    left_multiply_X(M.T, MX)
    return left_multiply_X(MX.T)


class StateSpaceModel(object):

    # general notation:
    #
    # transitions:
    # F_k: transition matrix from time k-1 to time k
    # b_k: transition bias from time k-1 to time k
    #    so E[x_k] = F_k*x_{k-1} + b_k
    # Q_k: transition noise covariance from time k-1 to k.
    #   generally assumed to be diagonal.

    def __init__(self):
        pass

    def apply_transition_matrix(self, x, k, result):
        # x: state at time k-1
        # implicitly (in result) return the vector (F_k)x
        # explicitly return the length of that vector (in case result is too big).
        pass

    def transition_covariance(self, P, k, result):
        """
        P: covariance matrix at time k-1
        return predicted covariance matrix at time k,
        implicitly computing FPF'
        """

        try:
            MX = self.tmpDD
        except:
            self.tmpDD = np.empty((self.max_dimension, self.max_dimension))
            MX = self.tmpDD

        Fx = lambda x, r : self.apply_transition_matrix(x, k, r)

        return conjugate(P, Fx, MX, result)

    def transition_bias(self, k, x):
        """
        result: stores x+b_k into x, where b_k is the transition bias
        (i.e., x_k = F_k*x_{k-1} + b_k)
        """
        raise Exception("not implemented")

    def transition_noise_diag(self, k, noise):
        """
        Compute the diagonal of the transition noise covariance Q_k,
        and store into the provided vector. We assume all non-diagonal
        entries of Q_k are zero.
        """
        raise Exception("not implemented")

    def apply_observation_matrix(self, x, k, result):
        # can be applied in two modes. Let H_k be the observation
        # matrix at time k.
        # - If x is a vector, we return the scalar H_k x.
        # - If x is a matrix, store the vector H_k x into the provided
        #   vector, and return None.
        # TODO: find a less ugly way to do this.
        raise Exception("not implemented")

    def observation_bias(self, k):
        """
        Return a scalar bias term for the observation.
        """
        return 0.0

    def observation_variance(self, P, k):
        # P: state covariance at time k
        # returns the (scalar) variance of the (noiseless) observation at time k,
        # given by H_k*P*H_k'
        Hx = lambda x, r=None: self.apply_observation_matrix(x, k, r)
        return conjugate_scalar(P, Hx)

    def observation_noise(k):
        # return the (scalar) observation noise variance at time k
        raise Exception("not implemented")

    def kalman_predict(self, k, xk1, Pk1, xk, Pk):
        # xk1, Pk1: mean and covariance for the state at time k-1
        # xk, Pk: empty vector/matrix in which the mean and covariance
        #         at time k will be stored

        # return values are stored in xk and Pk
        state_size = self.apply_transition_matrix(xk1, k, xk)
        self.transition_bias(k, xk)

        # can use xk1 as temporary storage now since we've
        # already computed the state transition
        self.transition_covariance(Pk1, k, Pk)
        self.transition_noise_diag(k, xk1)
        np.fill_diagonal(Pk, np.diag(Pk) + xk1)
        return state_size

    def kalman_observe(self, k, zk, xk, Pk, KkT, state_size):

        # zk: (scalar) observation at time k
        # xk, Pk: mean and covariance of the predicted state
        #         at time k. These are overwritten with
        #         updated state estimate.
        # KkT: temporary vector of length len(xk)
        #
        # returns:
        #    step_ell: likelihood of this observation, given the
        #              filtered state estimate.
        #    (implicitly) xk, Pk: the updated state estimate

        pred_z = self.apply_observation_matrix(xk, k) + self.observation_bias(k)
        yk = zk - pred_z

        Sk = self.observation_variance(Pk, k)
        assert(Sk > 0)
        Sk += self.observation_noise(k)

        self.apply_observation_matrix(Pk.T, k, KkT)

        xk[:state_size] += KkT[:state_size] * yk / Sk
        Pk[:state_size, :state_size] -= np.outer(KkT[:state_size], KkT[:state_size]/Sk)

        # update log likelihood
        step_ell= scipy.stats.norm.logpdf(yk, loc=0, scale=np.sqrt(Sk))
        assert(not np.isnan(step_ell))

        return step_ell

    def prior_mean(self):
        pass

    def prior_vars(self):
        pass

    def run_filter(self, z, likelihood_only=True):
        N = len(z)
        ell = 0

        xk = np.empty((self.max_dimension,))
        mean = self.prior_mean()
        state_size = len(mean)
        xk[:state_size] = self.prior_mean()
        print "k=0, state size", state_size

        Pk = np.empty((self.max_dimension, self.max_dimension))
        Pk[:state_size, :state_size] = np.diag(self.prior_vars())

        tmp = np.empty((self.max_dimension,))
        ell = self.kalman_observe(0, z[0], xk, Pk, tmp, state_size)
        if not likelihood_only:
            x = np.empty((self.max_dimension, N))
            x[:len(xk), 0:1] = xk

        # create temporary variables
        xk1 = xk.copy()
        Pk1 = Pk.copy()
        for k in range(1, N):
            state_size = self.kalman_predict(k, xk1, Pk1, xk, Pk)

            if not np.isnan(z[k]):
                ell += self.kalman_observe(k, z[k], xk, Pk, tmp, state_size)

            # the result of the filtering update, stored in xk, Pk,
            # now becomes the new "previous" version (xk1, Pk1) for
            # the next round.
            xk1, xk = xk, xk1
            Pk1, Pk = Pk, Pk1

            if not likelihood_only:
                x[:, k:k+1] = xk

        if likelihood_only:
            return ell
        else:
            return ell, x

    def mean_obs(self, N):
        z = np.empty((N,))
        x = np.empty((self.max_dimension,))
        prior_mean = self.prior_mean()
        x[:len(prior_mean)] = prior_mean
        z[0] = self.apply_observation_matrix(x, 0) + self.observation_bias(0)

        pred_state = x.copy()
        for k in range(1, N):
            state_size = self.apply_transition_matrix(x, k, pred_state)
            self.transition_bias(k, pred_state)
            x[:state_size]=pred_state[:state_size]

            z[k] = self.apply_observation_matrix(x, k) + self.observation_bias(k)
        return z


    def prior_sample(self, N):
        x = np.empty((self.max_dimension,))
        prior_mean = self.prior_mean()
        x[:len(prior_mean)] = scipy.stats.multivariate_normal(prior_mean, np.diag(self.prior_vars())).rvs(1)

        pred_obs = self.apply_observation_matrix(x, 0) + self.observation_bias(0)
        z1 = scipy.stats.norm(pred_obs, self.observation_noise(0)).rvs(1)
        z1 = z1.reshape((-1,))

        z = np.empty((N,))
        z[0] = z1

        pred_state = x.copy()
        for k in range(1, N):
            state_size = self.apply_transition_matrix(x, k, pred_state)
            self.transition_bias(k, pred_state)

            noise_diag = np.empty((state_size,))
            self.transition_noise_diag(k, noise_diag)
            noise_cov = np.diag(noise_diag)
            x[:state_size] = scipy.stats.multivariate_normal(pred_state[:state_size], noise_cov).rvs(1)

            pred_obs = self.apply_observation_matrix(x, k) + self.observation_bias(k)
            z[k] = scipy.stats.norm(pred_obs, self.observation_noise(k)).rvs(1)

        return z
