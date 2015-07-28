###############################################################################
# BayesianRidge regression
# modified from sklearn 0.16 BayesianRidge code: returns parameter covariance
# matrix as well as mean.


import numpy as np
import math
from scipy import linalg
from sklearn.linear_model.base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils.extmath import fast_logdet, pinvh
#from sklearn.utils import check_arrays


class BayesianRidgeWithCov(LinearModel, RegressorMixin):
    """Bayesian ridge regression

    Fit a Bayesian ridge model and optimize the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).

    Parameters
    ----------
    n_iter : int, optional
        Maximum number of iterations.  Default is 300.

    tol : float, optional
        Stop the algorithm if w has converged. Default is 1.e-3.

    alpha_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter. Default is 1.e-6

    alpha_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.
        Default is 1.e-6.

    lambda_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter. Default is 1.e-6.

    lambda_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.
        Default is 1.e-6

    compute_score : boolean, optional
        If True, compute the objective function at each step of the model.
        Default is False

    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        Default is True.

    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    verbose : boolean, optional, default False
        Verbose mode when fitting the model.


    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array, shape = (n_features)
       estimated precisions of the weights.

    scores_ : float
        if computed, value of the objective function (to be maximized)

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.BayesianRidge()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ... # doctest: +NORMALIZE_WHITESPACE
    BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
            copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
            n_iter=300, normalize=False, tol=0.001, verbose=False)
    >>> clf.predict([[1, 1]])
    array([ 1.])

    Notes
    -----
    See examples/linear_model/plot_bayesian_ridge.py for an example.
    """

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
                 lambda_1=1.e-6, lambda_2=1.e-6, compute_score=False,
                 fit_intercept=True, normalize=False, copy_X=True,
                 verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the model

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        #X, y = check_arrays(X, y, sparse_format='dense', dtype=np.float)
        X, y, X_mean, y_mean, X_std = self._center_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X)
        n_samples, n_features = X.shape

        self.X_ = X
        self.y_ = y

        ### Initialization of the values of the parameters
        alpha_ = 1. / np.var(y)
        lambda_ = 1.

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S ** 2

        ### Convergence loop of the bayesian ridge regression
        for iter_ in range(self.n_iter):

            ### Compute mu and sigma
            # sigma_ = lambda_ / alpha_ * np.eye(n_features) + np.dot(X.T, X)
            # coef_ = sigma_^-1 * XT * y
            sigma_ = np.dot(Vh.T,
                           Vh / (eigen_vals_ + lambda_ / alpha_)[:, None])
            coef_ = np.dot(sigma_, XT_y)
            if self.compute_score:
                logdet_sigma_ = - np.sum(
                    np.log(lambda_ + alpha_ * eigen_vals_))

            ### Update alpha and lambda
            rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)
            gamma_ = (np.sum((alpha_ * eigen_vals_)
                      / (lambda_ + alpha_ * eigen_vals_)))
            lambda_ = ((gamma_ + 2 * lambda_1)
                       / (np.sum(coef_ ** 2) + 2 * lambda_2))
            alpha_ = ((n_samples - gamma_ + 2 * alpha_1)
                      / (rmse_ + 2 * alpha_2))

            ### Compute the objective function
            if self.compute_score:
                s = lambda_1 * np.log(lambda_) - lambda_2 * lambda_
                s += alpha_1 * np.log(alpha_) - alpha_2 * alpha_
                s += 0.5 * (n_features * np.log(lambda_)
                            + n_samples * np.log(alpha_)
                            - alpha_ * rmse_
                            - (lambda_ * np.sum(coef_ ** 2))
                            - logdet_sigma_
                            - n_samples * np.log(2 * np.pi))
                self.scores_.append(s)

            ### Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_ = coef_

        # posterior covariance is
        # (\lambda I + \alpha XX')^{-1}
        # = 1/\alpha (\lambda/\alpha I + X'X)^{-1}
        # which in code notation is:
        # = 1/alpha_ * sigma_
        self.cov_ = 1.0/alpha_ * sigma_

        self._set_intercept(X_mean, y_mean, X_std)
        return self
