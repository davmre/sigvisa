


def coord_descent(f, x0, converge=0.1, steps=None, maxiters=500):
    """
    Use coordinate descent to minimize the function f, starting with a vector x0.

    converge: stop when the function val decreases by less than this amount
    steps: vector of step sizes, one for each dimension
    maxiters: maximum number of iterations
    """

    ncoords = len(x0)
    x = x0.copy()
    v = f(x)
    for i in range(maxiters):
        incr = 0
        for p in np.random.permutation(ncoords):

            # try taking steps in both directions
            step = steps[p]
            x[p] = x[p] + step
            v1 = f(x)
            x[p] = x[p] - 2*step
            v2 = f(x)
            if v <= v1 and v <= v2:
                x[p] = x[p] + step
                continue

            # continue stepping in the best direction, until there's
            # no more improvement.
            if v1 < v2:
                vold = v1
                x[p] = x[p] + 3 * step
                sign = 1
            else:
                vold = v2
                sign = -1
                x[p] = x[p] - step
            vnew = f(x)
            while vnew <= vold:
                x[p] = x[p] + sign*step
                vold = vnew
                vnew = f(x)

            x[p] = x[p] - sign*step
            incr = np.max([v - vold, incr])
            v = vold
        if incr < converge:
            break
        if i % 10 == 0:
            print "coord iter %d incr %f" % (i, incr)
    return x


def optimize_by_phase(f, start_params, bounds, phaseids, method="bfgs", iters=3, maxfun=None):
    nphase_params = len(start_params) / len(phaseids)
    params = start_params.copy()
    for i in range(iters):
        for (pidx, phaseid) in enumerate(phaseids):
            sidx = pidx*nphase_params
            eidx = (pidx+1)*nphase_params
            phase_params = params[sidx:eidx]
            phase_bounds = bounds[sidx:eidx]
            apf = lambda pp : f(np.concatenate([params[:sidx], pp, params[eidx:]]))
            phase_params, c = minimize(apf, phase_params, method=method, bounds=phase_bounds, steps = [.1, .1, .005], maxfun=maxfun)
            print "params", phase_params, "cost", c
            params = np.concatenate([params[:sidx], phase_params, params[eidx:]])
    return params, c

def optimize(f, start_params, bounds, method, phaseids=None, maxfun=None):
    if phaseids is not None:
        return optimize_by_phase(f, start_params, bounds, phaseids, method=method,maxfun=maxfun)
    else:
        return minimize(f, start_params, bounds=bounds, method=method, steps=[.1, .1, .005] * (len(start_params)/3), maxfun=maxfun)




def minimize(f, x0, method="bfgs", bounds=None, steps=None, maxfun=None):
    if method=="bfgs":
        x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f, x0, approx_grad=1, bounds=bounds, epsilon = 1e-1, factr=1e12, maxfun=maxfun)
    elif method=="tnc":
        x1, nfeval, rc = scipy.optimize.fmin_tnc(f, x0, approx_grad=1, bounds=bounds, maxfun=maxfun)
        x1 = np.array(x1)
    elif method=="simplex":
        x1 = scipy.optimize.fmin(f, x0, maxfun=maxfun, xtol=0.01, ftol=0.01)
    elif method=="anneal":
        x1, jmin, T, feval, iters, accept, retval = scipy.optimize.anneal(f, x0, maxeval=maxfun)
    elif method=="coord":
        x1 = coord_descent(f, x0, steps=steps)
    else:
        raise Exception("unknown optimization method %s" % (method))
    return x1, f(x1)

def minimize_matrix(f, start, method, low_bounds=None, high_bounds=None, maxfun=None):
    shape = start.shape
    f = lambda params: f(np.reshape(params, shape))
    bounds = zip(low_bounds.flatten(), high_bounds.flatten())
    result_vector = minimize(f, start.flatten(), method=method, bounds=bounds, maxfun=maxfun) 
    return np.reshape(result_vector, shape)
