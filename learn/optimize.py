import numpy as np
import scipy
import scipy.optimize
from utils.gradient_descent import gradient_descent, coord_steps

class BoundsViolation(Exception):
    pass


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

def minimize(f, x0, optim_params, bounds=None):

    method = optim_params['method']
    eps = optim_params['eps']
    disp = optim_params['disp']
    normalize = optim_params['normalize']

    if normalize:
        if bounds is not None:
            low_bounds, high_bounds = zip(*bounds)
            low_bounds = np.asarray(low_bounds)
            high_bounds = np.asarray(high_bounds)
            x0 = scale_normalize(x0, low_bounds, high_bounds)
            bounds = [(-1,1) for s in x0]
            f1 = lambda params: f(scale_unnormalize(params, low_bounds, high_bounds))
        else:
            raise Exception("the normalize option requires optimization bounds to be specified")
    else:
        f1 = f

    if method=="bfgscoord":
        iters = 0
        success=False
        x1 = x0
        while iters < optim_params['bfgscoord_iters']:
            x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f1, x1, approx_grad=1, factr=optim_params['bfgs_factr'], epsilon=eps, bounds=bounds, disp=disp)
            success = (d['warnflag'] == 0)
            v1 = best_cost
            x2 = coord_steps(f1, x1, eps=eps, bounds=bounds)
            v2 = f1(x2)

            if success and v2 > v1 * .999 or np.linalg.norm(x2-x1, 2) < 0.00001:
                break
            x1 = x2
            iters += 1

    elif method=="bfgs":
        x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(f1, x0, approx_grad=1, factr=optim_params['bfgs_factr'], epsilon=eps, bounds=bounds, disp=disp)
    elif method=="tnc":
        x1, nfeval, rc = scipy.optimize.fmin_tnc(f1, x0, approx_grad=1, ftol=optim_params['ftol'], bounds=bounds, disp=disp)
        x1 = np.array(x1)
    elif method=="simplex":
        x1 = scipy.optimize.fmin(f1, x0, xtol=optim_params['xtol'], ftol=optim_params['ftol'], disp=disp)
    elif method=="grad":
        x1 = gradient_descent(f1, x0, eps=eps, stopping_eps = optim_params['grad_stopping_eps'], alpha = 0.3, beta = 0.9, bounds=bounds, disp=disp)
    elif method=="coord":
        x1 = coord_descent(f1, x0, steps=steps)
    elif method=="none":
        x1 = x0
    else:
        raise Exception("unknown optimization method %s" % (method))

    if normalize:
        x1 = scale_unnormalize(x1, low_bounds, high_bounds)
    return x1, f(x1)

def scale_normalize(x, low_bounds, high_bounds):
    half_width = (high_bounds - low_bounds)/2
    midpoint = low_bounds + half_width
    return (x - midpoint) / half_width

def scale_unnormalize(x, low_bounds, high_bounds):
    half_width = (high_bounds - low_bounds)/2
    midpoint = low_bounds + half_width
    newx = x * half_width + midpoint
    return newx


def minimize_matrix(f, start, optim_params, low_bounds=None, high_bounds=None, **other_args):

    ffc = optim_params['fix_first_cols']
    if ffc > 0:
        n = ffc
        first_col = start[:, 0:n]
        start = start[:, n:]

        if low_bounds is not None and high_bounds is not None:
            low_bounds = low_bounds[:, n:]
            high_bounds = high_bounds[:, n:]
        shape = start.shape
        restore_matrix_form = lambda params: np.hstack([first_col, np.reshape(params, shape)])
    else:
        effective_start_params = start
        shape = start.shape
        restore_matrix_form = lambda params: np.reshape(params, shape)

    if low_bounds is not None and high_bounds is not None:
        low_bounds = low_bounds.flatten()
        high_bounds = high_bounds.flatten()
        bounds = zip(low_bounds, high_bounds)
    else:
        assert(low_bounds is None and high_bounds is None)
        bounds = None

    start = start.flatten()
    f_to_minimize = lambda params: f(restore_matrix_form(params))


    result_vector, cost = minimize(f_to_minimize, start, optim_params=optim_params,bounds=bounds)
    return restore_matrix_form(result_vector), cost
