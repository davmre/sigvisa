import numpy as np
import scipy.stats

def udu(M):
    # STOLEN from pykalman.sqrt.bierman (BSD license)

    """Construct the UDU' decomposition of a positive, semidefinite matrix M

    Parameters
    ----------
    M : [n, n] array
        Matrix to factorize

    Returns
    -------
    UDU : UDU_decomposition of size n
        UDU' representation of M
    """
    assert np.allclose(M, M.T), 'M must be symmetric, positive semidefinite'
    n = M.shape[0]

    # perform Bierman's COV2UD subroutine (fucking inclusive indices)
    M = np.triu(M)
    U = np.eye(n)
    d = np.zeros(n)
    for j in reversed(range(2, n + 1)):
        d[j - 1] = M[j - 1, j - 1]
        if d[j - 1] > 0:
            alpha = 1.0 / d[j - 1]
        else:
            alpha = 0.0
        for k in range(1, j):
            beta = M[k - 1, j - 1]
            U[k - 1, j - 1] = alpha * beta
            M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]

    d[0] = M[0, 0]
    return U, d

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

    def run_filter_naive(self, z):
        N = len(z)
        ell = 0

        xk = np.empty((self.max_dimension,))
        mean = self.prior_mean()
        state_size = len(mean)
        xk[:state_size] = self.prior_mean()

        Pk = np.empty((self.max_dimension, self.max_dimension))
        Pk[:state_size, :state_size] = np.diag(self.prior_vars())

        tmp = np.empty((self.max_dimension,))
        ell = self.kalman_observe(0, z[0], xk, Pk, tmp, state_size)

        # create temporary variables
        xk1 = xk.copy()
        Pk1 = Pk.copy()
        for k in range(1, N):
            state_size = self.kalman_predict(k, xk1, Pk1, xk, Pk)

            if not np.isnan(z[k]):
                step_ell = self.kalman_observe(k, z[k], xk, Pk, tmp, state_size)
                ell += step_ell

            # the result of the filtering update, stored in xk, Pk,
            # now becomes the new "previous" version (xk1, Pk1) for
            # the next round.
            xk1, xk = xk, xk1
            Pk1, Pk = Pk, Pk1

        return ell

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


    def kalman_observe_sqrt(self, k, zk, xk, U, d, state_size):

        if self.at_fixed_point and self.stationary(k):
            K = self.cached_gain[:state_size]
            alpha = self.cached_alpha
            U[:state_size,:state_size] = self.cached_obs_U[:state_size,:state_size]
            d[:state_size] = self.cached_obs_d[:state_size]
        else:
            self.at_fixed_point = False

            K = np.zeros((state_size,))
            f = np.zeros((state_size,))

            r = self.observation_noise(k)
            self.apply_observation_matrix(U[:state_size,:state_size], k, f)
            v = d[:state_size] * f

            if np.isnan(v).any():
                raise Exception("v is %s" % v)

            alpha = r + v[0]*f[0]
            d[0] *= r/alpha
            K[0] = v[0]
            u_tmp = np.empty((state_size,))
            for j in range(1, state_size):
                old_alpha = alpha
                alpha += v[j]*f[j]
                d[j] *= old_alpha/alpha
                u_tmp[:] = U[:state_size,j]
                U[:state_size,j] = U[:state_size,j] - f[j]/old_alpha * K
                K += v[j]*u_tmp

            if self.stationary(k):
                if (np.abs(self.cached_obs_d[:state_size] - d[:state_size]) < self.eps_stationary).all():
                    if (np.abs(self.cached_gain[:state_size] - K) < self.eps_stationary).all():
                        if (np.abs(self.cached_obs_U[:state_size,:state_size] - U[:state_size,:state_size]) < self.eps_stationary).all():
                            self.at_fixed_point = True
                if not self.at_fixed_point:
                    self.cached_alpha = alpha
                    self.cached_gain[:state_size] = K
                    self.cached_obs_U[:state_size,:state_size] = U[:state_size,:state_size]
                    self.cached_obs_d[:state_size] = d[:state_size]

        pred_z = self.apply_observation_matrix(xk, k) + self.observation_bias(k)
        yk = zk - pred_z
        xk[:state_size] += K*(yk/alpha)

        step_ell = -.5 * np.log(2*np.pi*alpha) - .5 * (yk)**2 / alpha
        assert(not np.isnan(step_ell))

        return step_ell

    def kalman_predict_sqrt(self, k, xk1, U, d, xk, prev_state_size):
        # here we do the *very* lazy thing of constructing the
        # full cov, adding noise, and then re-factoring it.
        # this is super-expensive compared to the noiseless case
        # (in which we just multiply U by the transition matrix),
        # so given that we usually only have one or two
        # noise dimensions, we should probably try to do
        # some more efficient update.


        # U starts as PxP
        # let state_size be S
        # so F is an SxP matrix
        # and the resulting U will be SxP
        # which we can pad to SxS by including zeros, or by
        # setting the relevant d's to be zero
        #

        state_size = self.apply_transition_matrix(xk1, k, xk)
        self.transition_bias(k, xk)

        if self.at_fixed_point and self.stationary(k):
            return self.cached_pred_U, self.cached_pred_d, prev_state_size
        else:
            self.at_fixed_point = False

            self.transition_noise_diag(k, xk1)
            U1 = U.copy()
            small_size = min(prev_state_size, state_size)
            for i in range(small_size):
                self.apply_transition_matrix(U[:,i], k, U1[:,i])

            if prev_state_size < state_size:
                for i in range(prev_state_size, state_size):
                    U1[:, i] = 0
                    d[i] = 0

            # if no noise, we can save the expensive part
            if np.sum(xk1!= 0) == 0:
                return U1, d, state_size

            P = np.dot(d[:state_size]*U1[:state_size,:state_size], U1[:state_size,:state_size].T)
            np.fill_diagonal(P, np.diag(P) + xk1[:state_size])
            U1[:state_size,:state_size], d[:state_size] = udu(P)

            if np.isnan(U1[:state_size,:state_size]).any():
                raise Exception("U is nan")

            # if this is (almost) the same as the previous invocation,
            # we've reached a stationary state
            if self.stationary(k):
                if (np.abs(self.cached_pred_d - d) < self.eps_stationary).all():
                    if (np.abs(self.cached_pred_U - U1) < self.eps_stationary).all():
                        self.at_fixed_point = True
                if not self.at_fixed_point:
                    self.cached_pred_U[:,:] = U1
                    self.cached_pred_d[:] = d

            return U1, d, state_size



    def run_filter(self, z, *args, **kwargs):
        for x, U, d in self.filtered_states(z, *args, **kwargs):
            pass

        return self.ell

    def init_caches(self):
        self.at_fixed_point = False
        self.wasnan = False
        self.eps_stationary = 1e-10

        self.cached_gain = np.empty((self.max_dimension,))
        self.cached_obs_d = np.empty((self.max_dimension,))
        self.cached_obs_U = np.empty((self.max_dimension, self.max_dimension))
        self.cached_pred_d = np.empty((self.max_dimension,))
        self.cached_pred_U = np.empty((self.max_dimension,self.max_dimension))

    def filtered_states(self, z, no_cache=False):
        """
        Run a factored UDU' Kalman filter (see Bruce Gibbs, Advanced
        Kalman Filtering, Least-Squares and Modeling: A Practical
        Handbook, section 10.2) and compute marginal likelihood of
        the given signal. This method functions as a generator
        to return the filtered state estimates. The run_filter method
        wraps this to return marginal likelihoods instead.
        """

        self.init_caches()

        N = len(z)
        ell = 0

        xk = np.empty((self.max_dimension,))
        mean = self.prior_mean()
        state_size = len(mean)
        xk[:state_size] = self.prior_mean()

        d = np.zeros((self.max_dimension, ))
        d[:state_size] = self.prior_vars()

        U = np.zeros((self.max_dimension,self.max_dimension))
        U[:state_size,:state_size] = np.eye(state_size)

        ell = self.kalman_observe_sqrt(0, z[0], xk, U, d, state_size)
        yield xk, U, d

        # create temporary variables
        xk1 = xk.copy()
        for k in range(1, N):

            if no_cache:
                self.at_fixed_point=False

            U, d, state_size = self.kalman_predict_sqrt(k, xk1, U, d, xk, state_size)

            if no_cache:
                self.at_fixed_point=False


            if np.isnan(z[k]):
                if self.at_fixed_point and not self.wasnan:
                    self.at_fixed_point = False
                self.wasnan = True
            else:
                if self.at_fixed_point and self.wasnan:
                    self.at_fixed_point = False
                self.wasnan = False

                ell += self.kalman_observe_sqrt(k, z[k], xk, U, d, state_size)

            yield xk, U, d

            xk, xk1 = xk1, xk

        self.ell=ell
