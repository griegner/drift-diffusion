"""Three parameter drift diffusion model."""

import autograd.numpy as np
from autograd import hessian, jacobian
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

from .pdf import pdf


class DriftDiffusionModel(BaseEstimator):
    def __init__(self, a=None, t=None, v=None, z=None, cov_estimator=None):
        """Drift diffusion model.

        Parameters
        ----------
        a : float or None
            ...
        t : float or None
            ...
        v : float or None
            ...
        z : float or None
            ...
        cov_estimator : str or None, optional
            The covariance estimator to use.
            Options are "sample-hessian", "outer-product", "misspecification-robust", "autocorrelation-robust", "all".
            If None, the covariance matrix will not be estimated.
            If "all", all covariance matrices will be estimated, by default None.

        Attributes
        ----------
        params_ : np.ndarray of shape (n_params, )
            The fitted parameter vector.
        covariance_ : np.ndarray of shape (n_params, n_params)
            The fitted covariance matrix.
        is_fitted_ : bool
            Indicates whether the estimator has been fitted.
        """
        super().__init__()
        self.a = a
        self.t = t
        self.v = v
        self.z = z
        self.free_params = np.array([a is None, t is None, v is None, z is None])
        self.cov_estimator = cov_estimator

    def _get_params(self, params_):
        iter_params = iter(params_)
        params = []
        for name, free in zip(["a", "t", "v", "z"], self.free_params):
            if free:
                params.append(next(iter_params))
            else:
                params.append(getattr(self, name))
        return params

    def _loglikelihood(self, params_, X, y, epsilon=1e-10):
        all_params = self._get_params(params_)
        return np.log(pdf(X, y, *all_params) + epsilon)

    def _lossloglikelihood(self, params_, X, y):
        loglikelihood_ = self._loglikelihood(params_, X, y)
        return -np.sum(loglikelihood_)

    def _fit_covariance(self, X, y):

        # autograd derivatives
        ll_jacobian = jacobian(self._loglikelihood)
        lll_hessian = hessian(self._lossloglikelihood)

        def sample_hessian():
            hessian_ = lll_hessian(self.params_, X, y)
            return np.linalg.inv(hessian_)

        def outer_product():
            jacobian_ = ll_jacobian(self.params_, X, y)
            return np.linalg.inv(jacobian_.T @ jacobian_)

        def misspecification_robust():
            hessian_inv_ = sample_hessian()
            jacobian_ = ll_jacobian(self.params_, X, y)
            fisher_ = jacobian_.T @ jacobian_
            return hessian_inv_ @ fisher_ @ hessian_inv_

        def newey_west(jac, n_lags=None):
            jac = jac[:, np.newaxis] if jac.ndim == 1 else jac
            if n_lags is None:
                n_lags = int(np.floor(4 * (len(jac) / 100.0) ** (2.0 / 9.0)))
            weights = 1 - np.arange(n_lags + 1) / (n_lags + 1)
            outer_product = jac.T @ jac
            for lag in range(1, n_lags + 1):
                lagged_product = jac[lag:].T @ jac[:-lag]
                outer_product += weights[lag] * (lagged_product + lagged_product.T)
            return outer_product

        def autocorrelation_robust():
            hessian_inv_ = sample_hessian()
            jacobian_ = ll_jacobian(self.params_, X, y)
            newey_west_ = newey_west(jacobian_)
            return hessian_inv_ @ newey_west_ @ hessian_inv_

        cov_estimators = {
            "sample-hessian": sample_hessian,
            "outer-product": outer_product,
            "misspecification-robust": misspecification_robust,
            "autocorrelation-robust": autocorrelation_robust,
        }

        if self.cov_estimator == "all":
            return {k: v() for k, v in cov_estimators.items()}
        else:
            return cov_estimators.get(self.cov_estimator, lambda: None)()

    def fit(self, X, y):

        x = X if X.ndim == 1 else X[:, 0]

        # autograd derivatives
        lll_jacobian = jacobian(self._lossloglikelihood)
        lll_hessian = hessian(self._lossloglikelihood)

        # estimate parameters, covariance matrix
        fit_ = minimize(
            fun=self._lossloglikelihood,
            x0=np.array([1, 0, 0, 0])[self.free_params],  # initial guess
            args=(x, y),
            method="Newton-CG",
            jac=lll_jacobian,
            hess=lll_hessian,
        )
        self.params_ = fit_.x
        self.covariance_ = self._fit_covariance(x, y)

        self.is_fitted_ = True
        return self
