"""Drift diffusion model."""

import autograd.numpy as np
import pandas as pd
from autograd import hessian, jacobian
from formulaic import model_matrix
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from .pdf import pdf


class DriftDiffusionModel(BaseEstimator):
    def __init__(self, a="+1", t0="+1", v="+1", z="+1", cov_estimator="sample-hessian"):
        """Drift diffusion model (DDM) of binary decision making.

        DriftDiffusionModel fits decision making parameters by maximum likelihood estimation.
        The four decision making parameters (`a, t0, v, z`) can each be linear functions of
        sample-by-sample covariate columns in `X`. Each parameter can either be free (by default) or fixed.
        Free parameters are estimated during `fit` and defined with Wilkinson notation to specify linear
        relationships with covariates (e.g. `v = "+1 + coherence"`; see https://matthewwardrop.github.io/formulaic).
        Fixed parameters are set at initialization by passing a float.

        The covariance matrix of the estimator can be computed by one of four methods (see `cov_estimator`),
        each designed to be valid under increasingly general conditions on the outcome `y`.

        Parameters
        ----------
        a : float or str
            decision boundary (`a>0`) +a is upper and -a is lower, by default "+1"
        t0 : float or str
            nondecision time (`t0>=0`) +t0 is time in seconds, by default "+1"
        v : float or str
            drift rate (`-∞<v<+∞`) +v towards +a and -v towards -a, by default "+1"
        z : float or str
            starting point (`-1<z<+1`), +1 is +a and -1 is -a, by default "+1"
        cov_estimator : {"sample-hessian", "outer-product", "misspecification-robust",
                         "autocorrelation-robust", "all"}, by default "sample-hessian"

        Attributes
        ----------
        params_ : ndarray of shape (n_params, )
            estimated free parameters
        covariance_ : ndarray of shape (n_params, n_params)
            estimated covariances of free parameters
            standard errors are the square roots of the diagonal terms
        """
        super().__init__()
        self.a = a
        self.t0 = t0
        self.v = v
        self.z = z
        self.cov_estimator = cov_estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = None  # no type for MLE
        tags.target_tags.required = True  # y to be passed to fit
        tags.target_tags.one_d_labels = True  # y must be 1d
        return tags

    def _get_params(self, params_, X):
        params, idx = [], 0
        for param, x in zip(["a", "t0", "v", "z"], X):
            if x is not None:  # free parameter
                n_features = x.shape[1]
                params.append(x @ np.array(params_[idx : idx + n_features]))
                idx += n_features
            else:  # fixed params
                params.append(getattr(self, param))
        return params

    def _loglikelihood(self, params_, X, y):
        all_params = self._get_params(params_, X)
        return np.log(pdf(y, *all_params))

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
            return cov_estimators[self.cov_estimator]()

    def fit(self, X, y):
        """Fit DDM.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            sample-by-sample covariates
        y : np.Series of shape (n_samples, )
            reaction times (`abs(y)>0`) decision + nondecision time\\
            responses (`sign(y) = {+1, -1}`) +1 is upper and -1 is lower
        """

        # validate y
        y = check_array(y, dtype="numeric", ensure_all_finite=True, ensure_2d=False, estimator=DriftDiffusionModel)

        # validate X by model_matrix() + initialize parameters
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        X_mm, params0 = [], []
        for param in ["a", "t0", "v", "z"]:
            val = getattr(self, param)
            if isinstance(val, str):  # free parameter
                x_mm = model_matrix(val, X, output="numpy", na_action="raise")
                X_mm.append(x_mm)
                params0.append(np.r_[1, np.zeros(x_mm.shape[1] - 1)] if param == "a" else np.zeros(x_mm.shape[1]))
            else:  # fixed parameter
                X_mm.append(None)

        # autograd derivatives
        lll_jacobian = jacobian(self._lossloglikelihood)
        lll_hessian = hessian(self._lossloglikelihood)

        # estimate parameters, covariance matrix
        fit_ = minimize(
            fun=self._lossloglikelihood,
            x0=np.hstack(params0),
            args=(X_mm, y),
            method="Newton-CG",
            jac=lll_jacobian,
            hess=lll_hessian,
        )
        self.fit_ = fit_
        self.params_ = fit_.x
        self.covariance_ = self._fit_covariance(X_mm, y)

        self.is_fitted_ = True
        self.n_features_in_ = len(self.params_)
        return self
