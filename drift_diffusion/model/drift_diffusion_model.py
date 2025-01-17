"""Three parameter drift diffusion model."""

import autograd.numpy as np
from autograd import hessian, jacobian
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


class DriftDiffusionModel(BaseEstimator):
    def __init__(self, a=1.0, v=0, z=0, cov_estimator=None, random_state=None):
        """_summary_

        Parameters
        ----------
        a : float, optional
            _description_, by default 1.0
        v : float, optional
            _description_, by default 0.1
        z : float, optional
            _description_, by default 0.1
        cov_estimator : str or None, optional
            The covariance estimator to use.
            Options are "sample-hessian", "outer-product", "misspecification-robust", "hac-robust", "all".
            If None, the covariance matrix will not be estimated.
            If "all", all covariance matrices will be estimated, by default None.
        random_state : int, RandomState instance, or None
            Random number generator for `sample_from_pdf()`, by default None.
        """
        self.a = a
        self.v = v
        self.z = z
        self.cov_estimator = cov_estimator
        self.random_state = random_state
        self.fitted_ = False

    def _pdf(self, X, y, a, v, z, err=0.001):
        """probability density function (PDF)"""

        x = X if X.ndim == 1 else X[:, 0]

        # to navarro 2009 notation
        a = 2 * a
        v = -y * v
        w = 0.5 * (1 - y * z)

        xt = x / (a**2)  # normalize time

        # k_l: number of terms for large x
        k_l = np.where(
            np.pi * xt * err < 1,
            np.sqrt(-2 * np.log(np.pi * xt * err) / (np.pi**2 * xt)),
            1 / (np.pi * np.sqrt(xt)),
        )
        k_l = np.maximum(k_l, 1 / (np.pi * np.sqrt(xt)))

        # k_s: number of terms for small x
        k_s = np.where(
            2 * np.sqrt(2 * np.pi * xt) * err < 1,
            2 + np.sqrt(-2 * xt * np.log(2 * np.sqrt(2 * np.pi * xt) * err)),
            2,
        )
        k_s = np.maximum(k_s, np.sqrt(xt) + 1)

        # f(x|1,0,w)
        mask_x_s = k_s < k_l

        # small x approximation
        K_s = np.ceil(k_s).astype(int)
        Ks_s = np.arange(-((K_s.max() - 1) // 2), ((K_s.max() - 1) // 2) + 1)
        exp_s = -((w + 2 * Ks_s[:, None]) ** 2) / (2 * xt)
        p_x_s = np.sum((w + 2 * Ks_s[:, None]) * np.exp(exp_s), axis=0) / np.sqrt(
            2 * np.pi * xt**3
        )

        # large x approximation
        K_l = np.ceil(k_l).astype(int)
        Ks_l = np.arange(1, K_l.max() + 1)
        exp_l = -(Ks_l[:, None] ** 2) * (np.pi**2) * xt / 2
        p_x_l = np.pi * np.sum(
            Ks_l[:, None] * np.exp(exp_l) * np.sin(Ks_l[:, None] * np.pi * w), axis=0
        )

        # combine small x and large x approximations
        p = np.where(mask_x_s, p_x_s, p_x_l)

        # f(x|a,v,w)
        p *= np.exp(-v * a * w - (v**2) * x / 2) / (a**2)

        return p

    def sample_from_pdf(self, n_samples=1000, x_range=(1e-5, 3)):

        random_state = check_random_state(self.random_state)

        num = 1000  # discretize pdf
        X = np.r_[np.linspace(*x_range, num), np.linspace(*x_range, num)]
        y = np.r_[-np.ones(num), np.ones(num)]

        pdfs = self._pdf(X, y, self.a, self.v, self.z)
        probs_ = pdfs / np.sum(pdfs)
        samples = random_state.choice(X * y, size=n_samples, p=probs_)
        return np.abs(samples), np.sign(samples)

    def _loglikelihood(self, params_, X, y, epsilon=1e-10):
        return np.log(self._pdf(X, y, *params_) + epsilon)

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

        def hac_robust():
            hessian_inv_ = sample_hessian()
            jacobian_ = ll_jacobian(self.params_, X, y)
            newey_west_ = newey_west(jacobian_)
            return hessian_inv_ @ newey_west_ @ hessian_inv_

        cov_estimators = {
            "sample-hessian": sample_hessian,
            "outer-product": outer_product,
            "misspecification-robust": misspecification_robust,
            "hac-robust": hac_robust,
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

        # estimate parameters
        fit_ = minimize(
            fun=self._lossloglikelihood,
            x0=[1, 0, 0],
            args=(x, y),
            method="Newton-CG",
            jac=lll_jacobian,
            hess=lll_hessian,
        )
        self.params_ = fit_.x
        self.set_params(
            **{k: v for k, v in zip(["a", "v", "z"], self.params_.tolist())}
        )
        self.covariance_ = self._fit_covariance(x, y)
        self.fitted_ = True

        return self
