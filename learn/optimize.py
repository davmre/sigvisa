import numpy as np
import scipy
import scipy.optimize

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


def minimize(f, x0, method="bfgs", steps=None, **bounds_and_maxfun):
    if method=="bfgs":
        x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f, x0, approx_grad=1, factr=1e10, **bounds_and_maxfun)
    elif method=="tnc":
        x1, nfeval, rc = scipy.optimize.fmin_tnc(f, x0, approx_grad=1, ftol=0.1, **bounds_and_maxfun)
        x1 = np.array(x1)
    elif method=="simplex":
        x1 = scipy.optimize.fmin(f, x0, xtol=0.01, ftol=0.01, **bounds_and_maxfun)
    elif method=="anneal":
        try:
            maxeval = bounds_and_maxfun['maxfun']
        except KeyError:
            maxeval = 100000
        x1, best_cost = scipy.optimize.anneal(f, x0, maxeval=maxeval)
    elif method=="coord":
        x1 = coord_descent(f, x0, steps=steps)
    elif method=="none":
        x1 = x0
    else:
        raise Exception("unknown optimization method %s" % (method))
    return x1, f(x1)

def minimize_matrix(f, start, method, fix_first_col=False, low_bounds=None, high_bounds=None, **other_args):

    if fix_first_col:
        first_col = start[:, 0:1]
        start = start[:, 1:]

        if low_bounds is not None and high_bounds is not None:
            low_bounds = low_bounds[:, 1:]
            high_bounds = high_bounds[:, 1:]
        shape = start.shape
        restore_matrix_form = lambda params: np.hstack([first_col, np.reshape(params, shape)])
    else:
        effective_start_params = start
        shape = start.shape
        restore_matrix_form = lambda params: np.reshape(params, shape)

    f_to_minimize = lambda params: f(restore_matrix_form(params))

    if low_bounds is not None and high_bounds is not None:
        bounds = zip(low_bounds.flatten(), high_bounds.flatten())
        other_args['bounds'] = bounds
    else:
        assert(low_bounds is None and high_bounds is None)

    result_vector, cost = minimize(f_to_minimize, start.flatten(), method=method, disp=False, **other_args)
    return restore_matrix_form(result_vector), cost

