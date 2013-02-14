import numpy as np
import warnings


class LineSearchFailed(Exception):
    pass


def project_into_bounds(x, low_bounds, high_bounds):
    if low_bounds is not None:
        violations = (x < low_bounds)
        if violations.any():
            x[violations] = low_bounds[violations]
    if high_bounds is not None:
        violations = (x > high_bounds)
        if violations.any():
            x[violations] = high_bounds[violations]
    return x


def backtracking_line_search(f, x, step, grad_x, alpha, beta, max_iters=1000, low_bounds=None, high_bounds=None):
    t = 1
    i = 0
    fx = f(x)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            while f(project_into_bounds(x + t * step, low_bounds, high_bounds)) > fx + t * alpha * np.dot(grad_x, step):
                t *= beta
                i += 1

                nx = project_into_bounds(x + t * step, low_bounds, high_bounds)
#                print f(nx), nx, t, step
                if i > max_iters:
                    raise LineSearchFailed("stopping condition not reached after %d iters." % i)
        except RuntimeWarning as w:
            raise LineSearchFailed("runtime warning during function evaluation: " + str(w))

    return t


def coord_steps(f, x, eps=1e-4, bounds=None, maxfun="this_argument_is_ignored", disp=False):
    low_bounds, high_bounds = unpack_bounds(bounds)

    dims = len(x)
    grad_x = approx_gradient(f, x, eps)
    x1 = x
    if disp:
        print "start w/ f", f(x), "x", x, "grad", grad_x
    for d in range(dims):
        step = np.zeros((dims,))
        if grad_x[d] == 0:
            continue
        step[d] = -.3 * grad_x[d] / np.abs(grad_x[d])
        try:
            t = backtracking_line_search(
                f, x1, step, grad_x, alpha=0.3, beta=0.9, max_iters=50, low_bounds=low_bounds, high_bounds=high_bounds)
        except LineSearchFailed as e:
            if disp:
                print e
            continue
        x1 = project_into_bounds(x1 + step * t, low_bounds, high_bounds)
    if disp:
        print "end w/ f", f(x1), "x", x1
    return x1


def gradient_descent(f, x0, eps=1e-4, stopping_eps=1e-2, alpha=0.3, beta=0.9, f_grad=None, bounds=None, max_iters=999999, maxfun="this_argment_is_ignored", return_details=False, disp=False):
    """
    Gradient descent with backtracking line search.
    """
    x = x0.copy()

    low_bounds, high_bounds = unpack_bounds(bounds)

    xvals = [x, ]
    fvals = [f(x), ]
    stop_vals = []
    i = 0
    while True and i < max_iters:

        # compute gradient
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                if f_grad:
                    grad_x = f_grad(f, x)
                else:
                    grad_x = approx_gradient(f, x, eps)
            except RuntimeWarning as w:
                print "error computing gradient:", w
                break

        stopping_criterion = np.linalg.norm(grad_x, 2)
        stop_vals.append(stopping_criterion)
        if disp:
            print "iter %d fval %f norm_grad %f, x = %s" % (i, f(x), stopping_criterion, str(x))
            print "  grad", grad_x
        i += 1

        if stopping_criterion < stopping_eps:
            break
        try:
            t = backtracking_line_search(f, x, -grad_x, grad_x, alpha, beta, low_bounds=low_bounds, high_bounds=high_bounds)
            if t < 1e-10:
                raise LineSearchFailed("tiny step size t=%f" % t)
        except LineSearchFailed as e:
            print "line search failed:", str(e)
            break
        x = project_into_bounds(x - t * grad_x, low_bounds, high_bounds)
        xvals.append(x)
        fvals.append(f(x))

    if return_details:
        return x, np.array(xvals), np.array(fvals), np.array(stop_vals)
    else:
        return x


def approx_gradient(f, x0, eps):
    n = len(x0)
    grad = np.zeros((n,))
    fx0 = f(x0)
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += eps
        grad[i] = (f(x_new) - fx0) / eps

    print "grad eval: eps %f grad %s x0 %s" % (eps, grad, x0)
    return grad


def unpack_bounds(bounds):
    low_bounds = None
    high_bounds = None
    if bounds is not None:
        low_bounds, high_bounds = zip(*bounds)
        low_bounds = np.asarray(low_bounds)
        high_bounds = np.asarray(high_bounds)
    return low_bounds, high_bounds
