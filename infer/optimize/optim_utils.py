import numpy as np
import scipy
import scipy.optimize
from sigvisa.infer.optimize.gradient_descent import gradient_descent, coord_steps, fast_coord_step


class BoundsViolation(Exception):
    pass


def construct_optim_params(optim_param_str=''):

    def copy_dict_entries(keys, src, dest):
        if len(keys) == 0:
            keys = src.keys()
        for key in keys:
            dest[key] = src[key]

    defaults = {
        "method": "bfgscoord",
        "fix_first_cols": 0,
        "normalize": True,
        "maxfun": 15000,
        'disp': False,
        'random_inits': 0,
        "eps": 1e-4,  # increment for approximate gradient evaluation

        "bfgscoord_iters": 5,
        "bfgs_factr": 10,  # used by bfgscoord and bfgs
        "xtol": 0.01,  # used by simplex
        "ftol": 0.01,  # used by simplex, tnc
        "grad_stopping_eps": 1e-4,
    }
    overrides = eval("{%s}" % optim_param_str)

    optim_params = {}
    optim_params['method'] = overrides['method'] if 'method' in overrides else defaults['method']
    copy_dict_entries(["fix_first_cols", "normalize", "disp", "eps", "maxfun", "random_inits"],
                      src=defaults, dest=optim_params)
    method = optim_params['method']

    # load appropriate defaults for each method
    if method == "bfgscoord" or method == "bfgs_fastcoord":
        copy_dict_entries(["bfgscoord_iters", "bfgs_factr"], src=defaults, dest=optim_params)
    elif method == "bfgs":
        copy_dict_entries(["bfgs_factr", ], src=defaults, dest=optim_params)
    elif method == "tnc":
        copy_dict_entries(["ftol", ], src=defaults, dest=optim_params)
    elif method == "simplex":
        copy_dict_entries(["ftol", "xtol", ], src=defaults, dest=optim_params)
    elif method == "grad":
        copy_dict_entries(["grad_stopping_eps", ], src=defaults, dest=optim_params)

    copy_dict_entries([], src=overrides, dest=optim_params)
    return optim_params


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
            x[p] = x[p] - 2 * step
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
                x[p] = x[p] + sign * step
                vold = vnew
                vnew = f(x)

            x[p] = x[p] - sign * step
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
            sidx = pidx * nphase_params
            eidx = (pidx + 1) * nphase_params
            phase_params = params[sidx:eidx]
            phase_bounds = bounds[sidx:eidx]
            apf = lambda pp: f(np.concatenate([params[:sidx], pp, params[eidx:]]))
            phase_params, c = minimize(
                apf, phase_params, method=method, bounds=phase_bounds, steps=[.1, .1, .005], maxfun=maxfun)
            print "params", phase_params, "cost", c
            params = np.concatenate([params[:sidx], phase_params, params[eidx:]])
    return params, c


def optimize(f, start_params, bounds, method, phaseids=None, maxfun=None):
    if phaseids is not None:
        return optimize_by_phase(f, start_params, bounds, phaseids, method=method, maxfun=maxfun)
    else:
        return minimize(f, start_params, bounds=bounds, method=method, steps=[.1, .1, .005] * (len(start_params) / 3), maxfun=maxfun)


def minimize(f, x0, optim_params, fprime=None, bounds=None):
    eps = optim_params['eps']
    normalize = optim_params['normalize']
    random_inits = optim_params['random_inits']

    if normalize:
        if bounds is None:
            raise Exception("the normalize option requires optimization bounds to be specified")
        else:
            low_bounds, high_bounds = zip(*bounds)
            low_bounds = np.asarray(low_bounds)
            high_bounds = np.asarray(high_bounds)

            x0n = scale_normalize(x0, low_bounds, high_bounds)
            bounds = [(-1 if np.isfinite(low_bounds[i]) else np.float('-inf'), 1 if np.isfinite(high_bounds[i]) else np.float('inf')) for (i, s) in enumerate(x0)]
            f1 = lambda params: f(scale_unnormalize(params, low_bounds, high_bounds))
            assert(    (np.abs(x0 - scale_unnormalize(x0n, low_bounds, high_bounds) ) < 0.001 ).all() )
            x0 = x0n

            if fprime is not None:
                # we want to take approximate gradients in the normalized scale
                half_width = (high_bounds - low_bounds) / 2
                bounded = np.isfinite(half_width)
                normalized_eps = eps * np.ones(len(x0))
                if bounded.any():
                    normalized_eps[bounded] = eps * half_width[bounded]
                fp1 = lambda params: fprime(scale_unnormalize(params, low_bounds, high_bounds),
                                            eps=normalized_eps)  \
                                            * normalized_eps/eps
                approx_grad = 0
            else:
                print "fp1 is None, not normalizing gradient..."
                fp1 = None
                approx_grad = 1

            if fprime == "grad_included":
                f1 = lambda params: (f(scale_unnormalize(params, low_bounds, high_bounds)),
                                     fprime(scale_unnormalize(params, low_bounds, high_bounds),
                                            eps=normalized_eps) * normalized_eps/eps )
                fp1 = "grad_included"

    else:
        f1 = f
        fp1 = fprime if fprime != "grad_included" else None
        approx_grad=0 if fprime else 1

    if fprime == "grad_included":
        f_only = lambda x : f1(x)[0]
    else:
        f_only = f1

    x0 = np.asfarray(x0)
    new_params = lambda  :  np.exp(np.log(x0) + np.random.randn(len(x0)) * 1.5)
    starting_points = [x0,] + [new_params() for i in range(random_inits)]

    print starting_points

    best_result = np.inf
    best_val = None
    results = []
    for x in starting_points:
        x1 = _minimize(f1=f1, f_only=f_only, fp1=fp1, approx_grad=approx_grad, x0=x,
                       optim_params=optim_params, bounds=bounds)
        result = f_only(x1)
        results.append(result)
        if result < best_result:
            best_result = result
            best_val = x1

    if normalize:
        best_val = scale_unnormalize(best_val, low_bounds, high_bounds)
    return best_val, best_result

def _minimize(f1, f_only, fp1, approx_grad, x0, optim_params, bounds=None):

    np.seterr(all="raise", under="ignore")
    method = optim_params['method']
    eps = optim_params['eps']
    disp = optim_params['disp']
    maxfun = optim_params['maxfun']


    if method == "bfgscoord":
        iters = 0
        success = False
        x1 = x0
        while iters < optim_params['bfgscoord_iters']:
            x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(
                f1, x1, fprime=fp1, approx_grad=approx_grad, factr=optim_params['bfgs_factr'], epsilon=eps, bounds=bounds, disp=disp, maxfun=maxfun)
            success = (d['warnflag'] == 0)
            print d
            v1 = best_cost
            x2 = coord_steps(f1, fprime=fp1, approx_grad=approx_grad, x=x1, eps=eps, bounds=bounds)
            v2 = f_only(x2)

            if success and v2 > v1 * .999 or np.linalg.norm(x2 - x1, 2) < 0.00001:
                break
            x1 = x2
            iters += 1
    elif method == "bfgs_fastcoord":
        iters = 0
        success = False
        x1 = x0
        while iters < optim_params['bfgscoord_iters']:
            x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(
                f1, x1, fprime=fp1, approx_grad=approx_grad, factr=optim_params['bfgs_factr'], epsilon=eps, bounds=bounds, disp=disp, maxfun=maxfun)
            success = (d['warnflag'] == 0)
            print d
            v1 = best_cost
            x2 = fast_coord_step(f1, approx_grad=approx_grad, x=x1, eps=eps, bounds=bounds)
            v2 = f_only(x2)

            if success and v2 > v1 * .999 or np.linalg.norm(x2 - x1, 2) < 0.00001:
                break
            x1 = x2
            iters += 1
    elif method == "bfgs":
        x1, best_cost, d = scipy.optimize.fmin_l_bfgs_b(
            f1, x0, fprime=fp1, approx_grad=approx_grad, factr=optim_params['bfgs_factr'], epsilon=eps, bounds=bounds, disp=disp, maxfun=maxfun)
    elif method == "tnc":
        x1, nfeval, rc = scipy.optimize.fmin_tnc(f1, x0, fprime=fp1, approx_grad=approx_grad, ftol=optim_params['ftol'], bounds=bounds, disp=disp, maxfun=maxfun)
        x1 = np.array(x1)
    elif method == "simplex":
        x1 = scipy.optimize.fmin(f_only, x0, xtol=optim_params['xtol'], ftol=optim_params['ftol'], disp=disp)
    elif method == "grad":
        x1 = gradient_descent(
            f1, x0, f_grad=fp1, eps=eps, stopping_eps=optim_params['grad_stopping_eps'], alpha=0.3, beta=0.9, bounds=bounds, disp=disp)
    elif method == "coord":
        x1 = coord_descent(f_only, x0, steps=steps)
    elif method == "none":
        x1 = x0
    else:
        raise Exception("unknown optimization method %s" % (method))

    return x1


def scale_normalize(x, low_bounds, high_bounds):
    half_width = (high_bounds - low_bounds) / 2
    bounded = np.isfinite(half_width)
    if not bounded.any(): return x
    midpoint = low_bounds[bounded] + half_width[bounded]
    normalized = (x[bounded] - midpoint) / half_width[bounded]

    newx = x.copy()
    newx[bounded] = normalized
    return newx

def scale_unnormalize(x, low_bounds, high_bounds):
    half_width = (high_bounds - low_bounds) / 2
    bounded = np.isfinite(half_width)
    if not bounded.any(): return x

    midpoint = low_bounds[bounded] + half_width[bounded]
    unnormalized = x[bounded] * half_width[bounded] + midpoint

    newx = x.copy()
    newx[bounded] = unnormalized
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

    result_vector, cost = minimize(f_to_minimize, start, optim_params=optim_params, bounds=bounds)
    return restore_matrix_form(result_vector), cost
