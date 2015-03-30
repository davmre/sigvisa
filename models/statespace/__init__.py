import numpy as np
import scipy.stats

def udu(M):
    # this method stolen from pykalman.sqrt.bierman by Daniel Duckworth (BSD license)
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

    oldM = M.copy()

    # perform Bierman's COV2UD subroutine (fucking inclusive indices)
    M = np.triu(M)
    U = np.eye(n)
    d = np.zeros(n)
    for j in reversed(range(2, n + 1)):
        d[j - 1] = M[j - 1, j - 1]
        if d[j - 1] > 0:
            alpha = 1.0 / d[j - 1]
        else:
            if np.abs(d[j-1]) > 1e-8:
                print "WARNING: nonpositive d[%d] %f in udu decomp"  % (j-1, d[j-1])
                import pdb; pdb.set_trace()
            d[j-1] = 0
            alpha = 0.0
        for k in range(1, j):
            beta = M[k - 1, j - 1]
            U[k - 1, j - 1] = alpha * beta
            M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]

    d[0] = M[0, 0]
    if d[0] < 0:
        if np.abs(d[0]) > 1e-8:
            print "WARNING: nonpositive d[%d] %f in udu decomp"  % (0, d[0])
            import pdb; pdb.set_trace()
        d[0] = 0

    if DEBUG_SSM:
        newM = np.dot(d*U, U.T)
        diffM = np.abs(newM-oldM)
        assert (np.max(diffM) < 1e-10)


    return U, d

DEBUG_SSM = False
COMPARE_LANGS = ()
def check_C_mat(item, k, M, size, lang="c", threshold=1e-8):
    if lang not in COMPARE_LANGS :
        return
    try:
        C_item = np.loadtxt("matrices/%s_%s_%d.txt" % (item, lang, k))
    except IOError as e:
        print e
        return

    if len(C_item.shape)==1:
        diff = np.abs(M[:size] - C_item[:size])
    else:
        diff = np.abs(M[:size,:size] - C_item[:size,:size])
    assert(  np.max(diff) < threshold )
    print "passed check for", item, "at step", k


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

    def transition_covariance(self, P, k, P_new):
        try:
            MX = self.tmpDD
        except:
            self.tmpDD = np.empty((self.max_dimension, self.max_dimension))
            MX = self.tmpDD

        N, _ = P.shape
        for i, row in enumerate(P):
            self.apply_transition_matrix(row, k, MX[i,:])
        for i, col in enumerate(MX.T):
            self.apply_transition_matrix(col, k, P_new[:,i])

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

    def observation_noise(k):
        # return the (scalar) observation noise variance at time k
        raise Exception("not implemented")

    def prior_mean(self):
        raise Exception("not implemented")

    def prior_vars(self):
        raise Exception("not implemented")

    def stationary(self, k):
        """
        Return whether the process is (locally) stationary at step k,
        meaning that the transition and observation models have not
        changed since the previous step. This is purely an
        optimization: it is always correct to return False, in which
        case no caching of covariances will be performed.
        """
        raise Exception("not implemented")

    def mean_obs(self, N):
        """
        Compute the mean observation under the prior distribution. That is,
        we initialize the state at the prior mean, compute the
        expected observation for the first timestep, apply the
        transition model (with no noise) to get the mean state at the
        next step, compute the expected observation at that step, etc.

        For many typical models (AR processes, wavelet models with std
        normal priors) this will just compute the zero vector. But in
        models with interesting priors (e.g. wavelet models with
        nonzero means on the coefs), or models with a nonzero observation bias,
        the result can be useful.
        """

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

    def obs_var(self, N):
        P = np.empty((self.max_dimension, self.max_dimension))
        prior_var = self.prior_vars()
        n_start = len(prior_var)
        P[:n_start,:n_start] = np.diag(prior_var)
        noise = np.zeros((self.max_dimension,))

        obs_vars = np.zeros((N,))

        k=0
        self.apply_observation_matrix(P, k, result=noise)
        pred_var = self.apply_observation_matrix(noise, k)
        r = self.observation_noise(k)
        obs_vars[k] = pred_var + r

        for k in range(1, N):

            TP = P.copy()
            state_size = self.apply_transition_matrix(P[:,0], k, TP[:,0])
            # have to range over the full dimension because the state
            # size at the previous timestep could have been larger
            # than now.
            for i in range(self.max_dimension):
                self.apply_transition_matrix(P[:,i], k, TP[:,i])
            for i in range(state_size):
                self.apply_transition_matrix(TP[i,:], k, P[i,:])
            self.transition_noise_diag(k, noise)
            np.fill_diagonal(P, np.diag(P)[:state_size] + noise[:state_size])
            #np.savetxt("matrices/P_py_%d.txt" % k, P)


            self.apply_observation_matrix(P, k, result=noise)
            pred_var = self.apply_observation_matrix(noise[:state_size], k)
            r = self.observation_noise(k)
            obs_vars[k] = pred_var + r


        return obs_vars

    def prior_sample(self, N):
        """
        Draw an observation from the prior distribution by forward-sampling.
        """

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


    def filtered_states(self, z):
        """Run a factored UDU' Kalman filter (see Bruce Gibbs, Advanced
        Kalman Filtering, Least-Squares and Modeling: A Practical
        Handbook, section 10.2) and compute marginal likelihood of
        the given signal. This method functions as a generator
        to return the filtered state estimates. The run_filter method
        wraps this to return marginal likelihoods instead.

        When the observation and transition models are stationary
        (constant over time, as in an AR process), the filtered
        covariance matrix will eventually converge to a fixed
        point. When this occurs, it's not necessary to recompute the
        covariance (or its factored representation) at each step;
        instead we can save computation by using a cached value. This
        implementation attempts to detect fixed points of the covariance
        matrix and cache them appropriately.
        """

        def init_caches():
            self.at_fixed_point = False

            # two ways to reach a fixed point: long sequence of
            # observed values, or long sequence of unobserved
            # values. switching from observed to unobserved or vice
            # versa is really an (implicit) change in the observation
            # model, so we need to track when it happens.
            self.wasnan = False

            # a matrix has reached a fixed point if no entry in the
            # new version differs from the previous by more than this
            # amount.
            self.eps_stationary = 1e-10

            # caches for covariance representations
            self.cached_gain = np.empty((self.max_dimension,))
            self.cached_obs_d = np.empty((self.max_dimension,))
            self.cached_obs_U = np.empty((self.max_dimension, self.max_dimension))
            self.cached_pred_d = np.empty((self.max_dimension,))
            self.cached_pred_U = np.empty((self.max_dimension,self.max_dimension))

        init_caches()

        N = len(z)


        # initialize the state at the prior mean.
        # pre-allocate space for the largest
        # possible state vector, even if we don't
        # use it all at the first timestep
        xk = np.empty((self.max_dimension,))
        mean = self.prior_mean()
        state_size = len(mean)
        xk[:state_size] = self.prior_mean()


        # initializae the covariance to the
        # (diagional) prior.
        d = np.zeros((self.max_dimension, ))
        d[:state_size] = self.prior_vars()
        U = np.zeros((self.max_dimension,self.max_dimension))
        U[:state_size,:state_size] = np.eye(state_size)

        check_C_mat("U_prior", 0, U, state_size)
        check_C_mat("d_prior", 0, d, state_size)
        check_C_mat("xk_prior", 0, xk, state_size)

        # update the state from the observation at the first timestep

        if np.isnan(z[0]):
            self.wasnan=True
            ell = 0
        else:
            self.wasnan=False
            ell = self.kalman_observe_sqrt(0, z[0], xk, U, d, state_size)
        yield xk, U, d


        check_C_mat("U_post_obs", 0, U, state_size)
        check_C_mat("d_post_obs", 0, d, state_size)
        check_C_mat("xk_post_obs", 0, xk, state_size)

        xk1 = xk.copy() # temp variable
        for k in range(1, N):

            # run the transition model to get the state estimate for the current timestep
            U, d, state_size = self.kalman_predict_sqrt(k, xk1, U, d, xk, state_size, force_P=False)

            check_C_mat("U_post_predict", k, U, state_size)
            check_C_mat("d_post_predict", k, d, state_size)
            check_C_mat("xk_post_predict", k, xk, state_size)

            P = np.dot(d[:state_size]*U[:state_size,:state_size], U[:state_size,:state_size].T)
            check_C_mat("P_pred", k, P, state_size, lang="naive")
            check_C_mat("x_pred", k, xk, state_size, lang="naive")

            # if there is an observation at this timestep, update the state estimate
            # accordingly. (and do some bookkeeping for the covariance cache)
            if np.isnan(z[k]):
                if self.at_fixed_point and not self.wasnan:
                    self.at_fixed_point = False

                self.wasnan = True
            else:
                if self.at_fixed_point and self.wasnan:
                    self.at_fixed_point = False
                self.wasnan = False

                ell += self.kalman_observe_sqrt(k, z[k], xk, U, d, state_size)

                check_C_mat("U_post_obs", k, U, state_size)
                check_C_mat("d_post_obs", k, d, state_size)
                check_C_mat("xk_post_obs", k, xk, state_size)

                P = np.dot(d[:state_size]*U[:state_size,:state_size], U[:state_size,:state_size].T)
                assert(np.min(np.linalg.eig(P)[0]) >= 0)
                check_C_mat("P_obs", k, P, state_size, lang="naive")
                check_C_mat("x_obs", k, xk, state_size, lang="naive")

            yield xk, U, d

            xk, xk1 = xk1, xk

        # generators can't return a value, but we save the marginal
        # likelihood so that run_filter can return it.
        self.ell=ell


    def run_filter(self, z, *args, **kwargs):
        """
        Compute the marginal likelihood of an observation under this
        state space model.
        """
        for k, (x, U, d) in enumerate(self.filtered_states(z, *args, **kwargs)):
            pass

        return self.ell

    def kalman_observe_sqrt(self, k, zk, xk, U, d, state_size):
        """
        Incorporate a new observation into the UDU' factored state estimate.
        """

        # Either load the updated covariance and Kalman gain from cache, or compute it if necessary
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

            # for this update, see Gerald Bierman, "Factorization
            # methods for Discrete Sequential Estimation" (1977),
            # section V.3, p. 78.
            # Another reference is Bruce Gibbs, "Advanced Kalman
            # Filtering, Least-Squares, and Modeling" (2011).

            #print "initial U", U
            self.apply_observation_matrix(U[:state_size,:state_size], k, f)
            #print "predicted obs", f
            v = d[:state_size] * f
            # print "got v", v
            if np.isnan(v).any():
                raise Exception("v is %s" % v)

            check_C_mat("U_obs_old", k, U, state_size)
            check_C_mat("d_obs_old", k, d, state_size)
            check_C_mat("f", k, f, state_size)
            check_C_mat("v", k, v, state_size)


            alpha = r + v[0]*f[0]
            # print "   alpha", alpha
            assert(alpha >= 0)
            if alpha > 1e-20:
                d[0] *= r/alpha
            else:
                alpha = 1e-20

            K[0] = v[0]
            u_tmp = np.empty((state_size,))
            for j in range(1, state_size):
                old_alpha = alpha
                incr = v[j]*f[j]
                assert(incr >= 0)
                alpha += incr
                print "   alpha %d %f" % (j, alpha)
                assert(alpha >= 0)
                if alpha > 1e-20:
                    print "d = %f * %f = %f" % (d[j], old_alpha/alpha, d[j]*old_alpha/alpha)
                    d[j] = d[j] * (old_alpha/alpha)

                    # old_alpha / (old_alpha+ incr)
                    # = 1.0 / (1+incr/old_alpha)

                else:
                    alpha = 1e-20
                u_tmp[:] = U[:state_size,j]
                U[:state_size,j] = U[:state_size,j] - K/old_alpha*f[j]
                K += v[j]*u_tmp

            check_C_mat("U_obs", k, U, state_size)
            check_C_mat("d_obs", k, d, state_size)


            # don't bother checking to see if we've reached a
            # fixed point: if we have, we'll realize it when we
            # run the prediction step. Instead, just assume we
            # haven't and update the caches accordingly.
            if self.stationary(k) and not self.at_fixed_point:
                    self.cached_alpha = alpha
                    self.cached_gain[:state_size] = K
                    self.cached_obs_U[:state_size,:state_size] = U[:state_size,:state_size]
                    self.cached_obs_d[:state_size] = d[:state_size]

        assert(not np.isnan(U[:state_size,:state_size]).any())
        assert(not np.isnan(d[:state_size]).any())


        # given the Kalman gain from the covariance update, compute
        # the updated mean vector.
        pred_z = self.apply_observation_matrix(xk, k) + self.observation_bias(k)
        yk = zk - pred_z
        xk[:state_size] += K*(yk/alpha)

        # also compute log marginal likelihood for this observation
        step_ell = -.5 * np.log(2*np.pi*alpha) - .5 * (yk)**2 / alpha

        # print "step %d pred %.4f alpha %.4f z %.4f y %.4f ell %f" % (k, pred_z, alpha, zk, yk, step_ell)

        assert(not np.isnan(step_ell))

        return step_ell


    def check_P(self, U, d, k, P=None):
        state_size = len(self.active_basis[k])
        if P is None:
            P = np.dot(d[:state_size]*U[:state_size,:state_size], U[:state_size,:state_size].T)
        H = self.obs_vector_debug(k)[:state_size]
        obs_var = np.dot(H, np.dot(P, H.T))
        assert (np.abs(obs_var-1) < 1e-8)


    def kalman_predict_sqrt(self, k, xk1, U, d, xk, prev_state_size, force_P=False):
        # here we do the *very* lazy thing of constructing the
        # full cov, adding noise, and then re-factoring it.
        # this is super-expensive compared to the noiseless case
        # (in which we just multiply U by the transition matrix),
        # so given that we usually only have one or two
        # noise dimensions, we should probably try to do
        # some more efficient update.

        # first push the mean through the transition model
        state_size = self.apply_transition_matrix(xk1, k, xk)
        self.transition_bias(k, xk)
        check_C_mat("xk_posttransit", k, xk, state_size)

        if self.at_fixed_point and self.stationary(k):
            U[:state_size,:state_size] = self.cached_pred_U[:state_size,:state_size]
            d[:state_size] = self.cached_pred_d[:state_size]
            return U.copy(), d.copy(), prev_state_size
        else:
            self.at_fixed_point = False

            # use xk1 as temp space to store the transition noise
            self.transition_noise_diag(k, xk1)

            # pushing the covariance P through the
            # transition model F yields FPF'. In a factored
            # representation, this is FUDU'F', so we just
            # need to compute FU.
            U1 = U.copy() # temp space, could be more efficient here
            min_size = min(prev_state_size, state_size)

            check_C_mat("U_pretransit", k, U1, state_size)


            for i in range(prev_state_size):
                self.apply_transition_matrix(U[:,i], k, U1[:,i])

            if prev_state_size < state_size:
                for i in range(prev_state_size, state_size):
                    U1[:, i] = 0
                    d[i] = 0

            check_C_mat("U_posttransit", k, U1, state_size)
            check_C_mat("d_posttransit", k, d, state_size)


            # if there is transition noise, do the expensive reconstruction/factoring step
            if force_P or state_size != prev_state_size or  np.sum(xk1!= 0) != 0:
                # construct the cov matrix
                P = np.dot(d[:prev_state_size]*U1[:state_size,:prev_state_size], U1[:state_size,:prev_state_size].T)

                check_C_mat("P_prenoise", k, P, state_size)
                check_C_mat("P_prenoise", k, P, state_size, lang="naive")

                # add transition noise
                np.fill_diagonal(P, np.diag(P) + xk1[:state_size])

                check_C_mat("P", k, P, state_size)

                #np.savetxt("matrices/P_py_%d.txt" % k, P)
                #np.savetxt("matrices/U_py_%d.txt" % k, U1)

                # get the new factored representation
                U1[:state_size,:state_size], d[:state_size] = udu(P)

                check_C_mat("U_decomp", k, U1, state_size)
                check_C_mat("d_decomp", k, d, state_size)

                #PP = np.dot(d[:state_size]*U1[:, :state_size], U1[:,:state_size].T)
                #assert (np.max(np.abs(PP[:state_size,:state_size]-P)) < 1e-8)

            assert(not np.isnan(U1[:state_size,:state_size]).any())
            assert(not np.isnan(d[:state_size]).any())

            # if our factored representation is (almost) the same as the previous invocation,
            # we've reached a stationary state
            if self.stationary(k):
                # we don't allow a fixed point on the first step after
                # the series becomes stationary, because the caches
                # won't have had time to be updated.
                if k > 0 and self.stationary(k-1):
                    if (np.abs(self.cached_pred_d[:state_size] - d[:state_size]) < self.eps_stationary).all():
                        if (np.abs(self.cached_pred_U[:state_size,:state_size] - U1[:state_size,:state_size]) < self.eps_stationary).all():
                            self.at_fixed_point = True
                if not self.at_fixed_point:
                    self.cached_pred_U[:,:] = U1
                    self.cached_pred_d[:] = d


            return U1, d, state_size

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
        np.savetxt("matrices/P_prenoise_naive_%d.txt" % k, Pk)
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

        H = self.obs_vector_debug(k)
        Sk = np.dot(H, np.dot(Pk, H.T))
        assert(Sk > 0)
        Sk += self.observation_noise(k)

        self.apply_observation_matrix(Pk.T, k, KkT)

        if k==1:
            import pdb; pdb.set_trace()

        xk[:state_size] += KkT[:state_size] * yk / Sk
        Pk[:state_size, :state_size] -= np.outer(KkT[:state_size], KkT[:state_size]/Sk)

        # update log likelihood
        step_ell= scipy.stats.norm.logpdf(yk, loc=0, scale=np.sqrt(Sk))
        assert(not np.isnan(step_ell))

        print "step %d pred %.4f alpha %.4f z %.4f y %.4f ell %f" % (k, pred_z, Sk, zk, yk, step_ell)

        return step_ell

    def kalman_observe_information(self, k, zk, xk, Pk, KkT, state_size):
        pred_z = self.apply_observation_matrix(xk, k) + self.observation_bias(k)
        yk = zk - pred_z

        H = self.obs_vector_debug(k)
        Sk = np.dot(H, np.dot(Pk, H.T))
        Sk += self.observation_noise(k)
        step_ell= scipy.stats.norm.logpdf(yk, loc=0, scale=np.sqrt(Sk))


        invPk = np.linalg.inv(Pk[:state_size,:state_size])
        HtRinvH = np.outer(H[:state_size], H[:state_size]) / self.observation_noise(k)
        Pk_new_internal = invPk + HtRinvH
        Pk_new = np.linalg.inv(Pk_new_internal)
        xk_new = np.dot(invPk, xk[:state_size]) + H[:state_size] * yk/self.observation_noise(k)
        xk_new = np.dot(Pk_new, xk_new)
        xk[:state_size] = xk_new
        Pk[:state_size,:state_size] = Pk_new

        return step_ell

    def run_filter_naive(self, z, likelihood_only=True):
        N = len(z)
        ell = 0

        xk = np.empty((self.max_dimension,))
        mean = self.prior_mean()
        state_size = len(mean)
        xk[:state_size] = self.prior_mean()
        print "k=0, state size", state_size

        Pk = np.zeros((self.max_dimension, self.max_dimension))
        Pk[:state_size, :state_size] = np.diag(self.prior_vars())

        tmp = np.empty((self.max_dimension,))
        ell = self.kalman_observe_information(0, z[0], xk, Pk, tmp, state_size)
        if not likelihood_only:
            x = np.empty((self.max_dimension, N))
            x[:len(xk), 0:1] = xk

        # create temporary variables
        xk1 = xk.copy()
        Pk1 = Pk.copy()
        for k in range(1, N):
            state_size = self.kalman_predict(k, xk1, Pk1, xk, Pk)

            np.savetxt("matrices/P_pred_naive_%d.txt" % k, Pk)
            np.savetxt("matrices/x_pred_naive_%d.txt" % k, xk)

            if not np.isnan(z[k]):
                ell += self.kalman_observe(k, z[k], xk, Pk, tmp, state_size)

            assert(np.min(np.linalg.eig(Pk)[0]) >= 0)
            np.savetxt("matrices/P_obs_naive_%d.txt" % k, Pk)
            np.savetxt("matrices/x_obs_naive_%d.txt" % k, xk)

            # the result of the filtering update, stored in xk, Pk,
            # now becomes the new "previous" version (xk1, Pk1) for
            # the next round.
            xk1, xk = xk, xk1
            Pk1, Pk = Pk, Pk1

        return ell
