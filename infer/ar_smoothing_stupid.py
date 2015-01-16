
import numpy as np

def filter_AR_stationary(x,mask,model):
    """
    Compute filtering distribution for a partially-observed AR process
    """

    n = len(x)
    params = model.params
    sigma2 = model.em.std**2
    n_p = len(params)

    filtered_means = np.zeros((n, n_p))
    filtered_covs = np.zeros((n, n_p, n_p))
    #filtered_inv_covs = np.zeros((n, n_p, n_p))

    tmp_mean = np.zeros((n_p,))
    tmp_cov = np.zeros((n_p, n_p))

    tmp = np.zeros((n_p, n_p))

    if n > 0:
        filtered_covs[0, :, :] = np.eye(n_p) * 1e4

    stationary_cov = False
    stationary_mean = False

    for t in range(0, n-1):

        if mask[t]:
            filtered_means[t+1,:] = filtered_means[t,:]
            filtered_covs[t+1,:,:] = filtered_covs[t,:,:]

            if not stationary_mean:
                update_u(filtered_means[t+1,:], params)
                if np.abs(filtered_means[t+1,0] - filtered_means[t,0]) < 0.00000001:
                    stationary_mean = True

            if not stationary_cov:
                updateK(filtered_covs[t+1, :, :], params, tmp)
                filtered_covs[t+1, 0, 0] += sigma2
                if np.abs(filtered_covs[t+1,0,0] - filtered_covs[t,0,0]) < 0.00000000000001:
                    stationary_cov = True
        else:
            # temporarily put the k|k values into the k+1|k slots
            stationary_mean = False
            stationary_cov = False
            update_obs(filtered_means[t,:], filtered_covs[t, :, :], filtered_means[t+1,:], filtered_covs[t+1, :, :], x[t])

            update_u(filtered_means[t+1,:], params)
            updateK(filtered_covs[t+1, :, :], params, tmp)

            filtered_covs[t+1, 0, 0] += sigma2


    return filtered_means, filtered_covs

def filter_AR_stupid(x, mask, model):
    """
    Compute filtering distribution for a partially-observed AR process
    """

    n = len(x)
    params = model.params
    sigma2 = model.em.std**2
    n_p = len(params)

    filtered_means = np.zeros((n, n_p))
    filtered_covs = np.zeros((n, n_p, n_p))
    #filtered_inv_covs = np.zeros((n, n_p, n_p))

    tmp_mean = np.zeros((n_p,))
    tmp_cov = np.zeros((n_p, n_p))

    tmp = np.zeros((n_p, n_p))

    if n > 0:
        filtered_covs[0, :, :] = np.eye(n_p) * 1.0e4


    for t in range(0, n-1):

        if not mask[t]:
            # temporarily put the k|k values into the k+1|k slots
            update_obs(filtered_means[t,:], filtered_covs[t, :, :], filtered_means[t+1,:], filtered_covs[t+1, :, :], x[t])
        else:
            filtered_means[t+1,:] = filtered_means[t,:]
            filtered_covs[t+1, :, :] = filtered_covs[t, :, :]

        update_u(filtered_means[t+1,:], params)

        updateK(filtered_covs[t+1, :, :], params, tmp)
        filtered_covs[t+1, 0, 0] += sigma2


    return filtered_means, filtered_covs

def update_obs(mean, cov, mean_out, cov_out, obs):
    # update the current Kalman state to incorporate an observed value.
    # since we have a very simple observation model (we deterministically
    # observe the first component of the state vector, equal to the value
    # of the AR process at the current timestep), this is pretty simple.

    # see 'how does the observation model work' in the org-notes for details

    n_p = len(mean)

    r = obs - mean[0]
    mean_out[:] = mean + cov[:,0] * r/cov[0,0]
    for i in range(n_p-1, -1, -1):
        cov_out[:,i] = cov[:,i] - cov[:,0] * cov[0,i]/cov[0,0]

def update_u(u, params):
    newval = np.dot(u, params)
    for i in range(len(u)-1, 0, -1):
        u[i] = u[i-1]
    u[0]=newval

def updateK(K, params, tmp):
    # given covariance K, compute
    # AKA^T

    n_p = len(params)

    # tmp = K * A.T
    for i in range(n_p): # row of K
        # first row of A contains the AR params
        tmp[i,0] = 0
        for j in range(n_p):
            tmp[i,0] += K[i,j] * params[j]

        # all the other rows are just the shifted identity matrix
        for j in range(1, n_p):
            tmp[i,j] = K[i,j-1]

    # K = A * tmp
    for i in range(n_p): # col of tmp
        K[0, i] = 0
        for j in range(n_p):
            K[0, i] += tmp[j,i] * params[j]

        for j in range(1, n_p):
            K[j,i] = tmp[j-1,i]

def updateKtrans(K, params, tmp):
    # given covariance K, compute
    # A^TKA

    n_p = len(params)
    # tmp = K * A
    for i in range(n_p):
        for j in range(0, n_p):
            tmp[i,j] = params[j] * K[i,0]
        for j in range(0, n_p-1):
            tmp[i,j] += K[i, j+1]

    # K = A.T * tmp
    for j in range(0, n_p):
        for i in range(n_p):
            K[i,j] = tmp[0,j] * params[i]
        for i in range(n_p-1):
            K[i,j] += tmp[i+1,j]


def smooth_AR_stupid(x, mask, model, filtered_means, filtered_covs, i_end):
    # implements http://en.wikipedia.org/wiki/Kalman_filter#Modified_Bryson.E2.80.93Frazier_smoother

    # smooths backwards from the end of x, to the index i_end, inclusive.
    # this means that entries filtered_means/covs[i_end:,:] will be converted to their smoothing distributions.
    # returns lambda_squiggle_{i_end} and lambda_hat_{i_end-1}

    params = model.params
    n_p = model.p
    n = len(x)

    Lambda_hat = np.zeros((n_p, n_p))
    Lambda_squiggle = np.zeros((n_p, n_p))
    lambda_hat = np.zeros((n_p),)
    lambda_squiggle = np.zeros((n_p),)


    for t in range(n-1, i_end-1, -1):

        if not mask[t]:
            update_Lambda_squiggle(Lambda_hat, Lambda_squiggle, filtered_covs[t,:,:])
            update_lambda_squiggle(lambda_hat, lambda_squiggle, filtered_covs[t,:,:], x[t] - filtered_means[t,0])
        else:
            lambda_squiggle[:] = lambda_hat
            Lambda_squiggle[:,:] = Lambda_hat


        #m1 = filtered_means[t,0]
        #v1 = filtered_covs[t,0,0]
        smooth_mean(filtered_means[t,:], filtered_covs[t,:,:], lambda_squiggle)
        smooth_cov(filtered_covs[t,:,:], Lambda_squiggle)

        #print '%d: filtered mean %f, observed %f, smoothed mean %f' % (t, m1, x[t], filtered_means[t,0])
        #print '%d: filtered cov %f, smoothed cov %f' % (t, v1, filtered_covs[t,0,0])

        update_Lambda_hat(Lambda_hat, Lambda_squiggle, params)
        update_lambda_hat(lambda_hat, lambda_squiggle, params)

    return lambda_hat, lambda_squiggle, Lambda_hat, Lambda_squiggle

def update_Lambda_hat(Lambda_hat, Lambda_squiggle, params):
    # given Lambda_squiggle[k], compute Lambda_hat[k-1]
    Lambda_hat[:,:] =  Lambda_squiggle
    updateKtrans(Lambda_hat, params, Lambda_squiggle)

def update_Lambda_hat_stupid(Lambda_hat, Lambda_squiggle, params):
    A = np.zeros((len(params), len(params)))
    A[0,:] = params
    A[1:, 0:-1] = np.eye(len(params)-1)

    #A = np.array((params, (1,0,0), (0,1,0)))
    Lambda_hat[:,:] = np.dot(A.T, np.dot(Lambda_squiggle, A))

def update_Lambda_squiggle(Lambda_hat, Lambda_squiggle, filtered_cov):
    # given Lambda_hat[k], compute Lambda_squiggle[k]
    # we assume we're given filtered_cov = P_{k|k-1}

    gain = filtered_cov[:, 0] / filtered_cov[0,0]

    #print "LH SYMMETRY", Lambda_hat
    """
    GH = np.zeros(Lambda_squiggle.shape)
    GH[:,0] = gain
    C = (np.eye(Lambda_squiggle.shape[0]) - GH)
    #Lambda_squiggle[:,:] = np.dot(C.T, np.dot(Lambda_hat, C))
    LS = np.dot(C.T, np.dot(Lambda_hat, C))
    """

    hat_g = np.dot(Lambda_hat, gain)
    hat_gt = np.dot(Lambda_hat.T, gain)
    Lambda_squiggle[:,:] = Lambda_hat[:,:]
    Lambda_squiggle[:, 0] -= hat_g
    Lambda_squiggle[0, :] -= hat_gt
    Lambda_squiggle[0,0]  += np.dot(hat_g,gain)

    #print "diff", np.sum(np.abs(Lambda_squiggle - LS))

    Lambda_squiggle[0,0] += 1.0/filtered_cov[0,0]
    #Lambda_squiggle[:,0] += 1.0/filtered_cov[0,0]
    #Lambda_squiggle[0,1:] += 1.0/filtered_cov[0,0]

def update_lambda_hat(lambda_hat, lambda_squiggle, params):
    # given lambda_squiggle[k], compute lambda_hat[k-1]
    for i in range(len(lambda_hat)-1):
        lambda_hat[i] = params[i]*lambda_squiggle[0] + lambda_squiggle[i+1]
    lambda_hat[-1] = params[-1]*lambda_squiggle[0]

def update_lambda_squiggle(lambda_hat, lambda_squiggle, filtered_cov, x):
    # given lambda_hat[k], compute lambda_squiggle[k]
    # we assume we're given (= wikipedia notation)
    #  filtered_cov = P_{k|k-1}
    #  x = y[k]

    gain = filtered_cov[:, 0] / filtered_cov[0,0]

    """
    n_p = len(lambda_squiggle)
    GH = np.zeros((n_p, n_p))
    GH[:,0] = gain
    C = (np.eye(n_p) - GH)
    ls = np.dot(C.T, lambda_hat)
    """

    lambda_squiggle[:] = lambda_hat[:]
    lambda_squiggle[0] -= np.dot(lambda_hat, gain)

    #print "ls diff", lambda_squiggle-ls

    lambda_squiggle[0] -= x/filtered_cov[0,0]

def smooth_cov(filtered_cov, Lambda_squiggle):
    a = np.dot(filtered_cov,np.dot(Lambda_squiggle, filtered_cov))
    filtered_cov -= a


def smooth_mean(filtered_mean, filtered_cov, lambda_squiggle):
    filtered_mean -= np.dot(filtered_cov, lambda_squiggle)

def smooth_cov_functional(filtered_cov, Lambda_squiggle):
    a = filtered_cov - np.dot(filtered_cov,np.dot(Lambda_squiggle, filtered_cov))

    """
    print "smooth cov. given filtered cov"
    print filtered_cov
    print " and squiggle"
    print Lambda_squiggle
    print " got smoothed cov"
    print a

    """
    return a

def smooth_mean_functional(filtered_mean, filtered_cov, lambda_squiggle):
    a = filtered_mean - np.dot(filtered_cov, lambda_squiggle)
    return a
