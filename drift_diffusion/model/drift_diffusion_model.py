"""Drift diffusion model."""

from functools import cache

import autograd.numpy as np
from autograd import hessian, jacobian
from better_optimize import minimize
from formulaic import model_matrix
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from .pdf import pdf


class DriftDiffusionModel(BaseEstimator):
    def __init__(self, a="+1", t0="+1", v="+1", z="+1", cov_estimator="sample-hessian", p_outlier=1e-12, verbose=False):
        """Drift diffusion model (DDM) of binary decision making.

        DriftDiffusionModel fits decision making parameters by maximum likelihood estimation.
        The four decision making parameters (`a, t0, v, z`) can each be linear functions of coefficients and
        sample-by-sample covariate columns in `X`; and each can be fixed, free, or mixed.
        Free parameters/coefficients are estimated during `fit` and defined with Wilkinson notation to specify linear
        relationships with covariates (e.g. `v = "+1 + coherence"`; see https://matthewwardrop.github.io/formulaic).
        Fixed parameters are set at initialization by passing a float. Mixed coefficients can be defined with a dict,
        for example `v={"formula": "+1 + coherence", "fixed": {"coherence": 1.0}}`, where `fixed` coefficients are
        excluded from optimization but included in likelihood evaluation.

        The covariance matrix of the estimator can be computed by one of four methods (see `cov_estimator`),
        each designed to be valid under increasingly general conditions on the outcome `y`.

        Parameters
        ----------
        a : float or str or dict
            decision boundary (`a>0`) +a is upper and -a is lower, by default "+1"
        t0 : float or str or dict
            nondecision time (`t0>=0`) +t0 is time in seconds, by default "+1"
        v : float or str or dict
            drift rate (`-∞<v<+∞`) +v towards +a and -v towards -a, by default "+1"
        z : float or str or dict
            starting point (`-1<z<+1`), +1 is +a and -1 is -a, by default "+1"
        cov_estimator : str
            {"sample-hessian", "outer-product", "misspecification-robust",
             "autocorrelation-robust", "all"}, by default "sample-hessian"
        p_outlier : float
            mixture probability (`0-1`) that a trial is drawn from a uniform outlier
            distribution rather than the DDM, by default 1e-12
        verbose : bool
            whether to display a progress bar

        Attributes
        ----------
        params_ : ndarray of shape (n_params, )
            estimated free parameters/coefficients
        covariance_ : ndarray of shape (n_params, n_params)
            estimated covariances of free parameters/coefficients
            standard errors are the square roots of the diagonal elements
        """
        super().__init__()
        self.a, self.t0, self.v, self.z = a, t0, v, z
        self.cov_estimator = cov_estimator
        self.p_outlier = p_outlier
        self.verbose = verbose

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = None  # no type for MLE
        tags.target_tags.required = True  # y to be passed to fit
        tags.target_tags.one_d_labels = True  # y must be 1d
        return tags

    def _get_model_matrix(self, X):
        X_mm, params0 = [], []
        self.fixed_ = []
        for param in ["a", "t0", "v", "z"]:
            val = getattr(self, param)

            if not isinstance(val, (str, dict)):  # fixed parameter
                X_mm.append(None)
                self.fixed_.append((None, None))
                continue

            is_mixed = isinstance(val, dict)
            formula = val if isinstance(val, str) else val.get("formula")
            fixed = {} if isinstance(val, str) else val.get("fixed", {})
            if is_mixed and not isinstance(formula, str):
                raise ValueError(f"{param} dict must include 'formula' as a string.")
            if is_mixed and not isinstance(fixed, dict):
                raise ValueError(f"{param} dict 'fixed' must be a dict mapping formula coefficients to fixed values ")

            fixed = dict(fixed)
            x_mm = model_matrix(formula, X, output="pandas", na_action="raise")
            if "1" in fixed and "Intercept" in x_mm.columns and "Intercept" not in fixed:
                fixed["Intercept"] = fixed.pop("1")

            unknown_fixed = sorted(set(fixed) - set(x_mm.columns))
            if unknown_fixed:
                raise ValueError(f"{param} fixed coefficients not found in formula: {unknown_fixed}. ")

            free_cols = [c for c in x_mm.columns if c not in fixed]
            x_free = x_mm[free_cols].to_numpy()
            X_mm.append(x_free)
            params0.append(np.r_[1, np.zeros(x_free.shape[1] - 1)] if param == "a" else np.zeros(x_free.shape[1]))

            fixed_cols = [c for c in x_mm.columns if c in fixed]
            self.fixed_.append(
                (x_mm[fixed_cols].to_numpy(), np.array([fixed[c] for c in fixed_cols], dtype=float))
                if fixed_cols
                else (None, None)
            )
        return X_mm, params0

    def _get_params(self, params_, X):
        params, idx = [], 0
        for param, x, fixed_ in zip(["a", "t0", "v", "z"], X, self.fixed_):
            if x is None:  # fixed params
                params.append(getattr(self, param))
                continue
            n_features = x.shape[1]
            param_value = x @ np.array(params_[idx : idx + n_features])
            x_fixed, b_fixed = fixed_
            if x_fixed is not None:
                param_value += x_fixed @ b_fixed
            params.append(param_value)
            idx += n_features
        return params

    def _loglikelihood(self, params_, X, y):
        rt = np.abs(y)
        all_params = self._get_params(params_, X)

        p_ddm = pdf(y, *all_params)  # ddm
        p_outlier = np.ones_like(y) / (rt.max() - rt.min())  # uniform
        p_mix = (1.0 - self.p_outlier) * p_ddm + self.p_outlier * p_outlier  # mixture
        return np.log(p_mix)

    def _lossloglikelihood(self, params_, X, y):
        loglikelihood_ = self._loglikelihood(params_, X, y)
        return -np.sum(loglikelihood_)

    def _fit_covariance(self, X, y):
        ll_jacobian = jacobian(self._loglikelihood)
        lll_hessian = hessian(self._lossloglikelihood)

        @cache
        def get_hessian_inv():
            return np.linalg.inv(lll_hessian(self.params_, X, y))

        @cache
        def get_jacobian():
            return ll_jacobian(self.params_, X, y)

        def sample_hessian():
            return get_hessian_inv()

        def outer_product():
            jacobian_ = get_jacobian()
            fisher_ = jacobian_.T @ jacobian_
            return np.linalg.inv(fisher_)

        def misspecification_robust():
            hessian_inv_ = get_hessian_inv()
            jacobian_ = get_jacobian()
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
            hessian_inv_ = get_hessian_inv()
            jacobian_ = get_jacobian()
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
        y : pd.Series of shape (n_samples, )
            reaction times (`abs(y)>0`) decision + nondecision time\\
            responses (`sign(y) = {+1, -1}`) +1 is upper and -1 is lower
        """

        # validate y
        y = check_array(y, dtype="numeric", ensure_all_finite=True, ensure_2d=False)
        # validate X + initialize parameters
        X_mm, params0 = self._get_model_matrix(X)

        # autograd derivatives
        lll_jacobian = jacobian(self._lossloglikelihood)
        lll_hessp = lambda params_, p, X, y: jacobian(lambda w: lll_jacobian(w, X, y) @ p)(params_)

        # estimate parameters, covariance matrix
        fit_ = minimize(
            f=self._lossloglikelihood,
            x0=np.hstack(params0),
            args=(X_mm, y),
            method="trust-ncg",
            jac=lll_jacobian,
            hessp=lll_hessp,
            progressbar=self.verbose,
            verbose=self.verbose,
        )
        self.fit_ = fit_
        self.params_ = fit_.x
        self.covariance_ = self._fit_covariance(X_mm, y)

        self.is_fitted_ = True
        self.n_features_in_ = len(self.params_)
        return self

    def pdf(self, X, y, alpha=0.05):
        """
        Compute the confidence bands for the PDF function under the fitted DDM.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            sample-by-sample covariates
        y : pd.Series of shape (n_samples, )
            reaction times (`abs(y)>0`) decision + nondecision time\\
            responses (`sign(y) = {+1, -1}`) +1 is upper and -1 is lower
        alpha : float
            two-sided significance level for the confidence bands, by default 0.05

        Returns
        -------
        band : dict
            keys are {"pdf", "+", "-"}
        lower : ndarray of shape (n_samples, )
            `alpha/2` confidence band
        upper : ndarray of shape (n_samples, )
            `1-alpha/2` confidence band
        """
        if self.cov_estimator == "all":
            raise ValueError("pdf() cannot be used when cov_estimator='all', set cov_estimator to a single estimator.")
        check_is_fitted(self)
        X_mm, _ = self._get_model_matrix(X)

        def _pdf(params_):
            return pdf(y, *self._get_params(params_, X_mm))

        pdf_ = _pdf(self.params_)
        pdf_jacobian_ = jacobian(_pdf)(self.params_)

        se = np.sqrt(np.einsum("ij,jk,ik->i", pdf_jacobian_, self.covariance_, pdf_jacobian_))
        z = norm.ppf(1 - alpha / 2)
        return {"pdf": pdf_, "-": pdf_ - z * se, "+": pdf_ + z * se}

    def g(self, X, alpha=0.05):
        """Compute the confidence bands for the g link function under the fitted DDM.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            sample-by-sample covariates
        alpha : float
            two-sided significance level for the confidence bands, by default 0.05

        Returns
        -------
        bands : dict
            keys are {"a","t0","v","z"} and values are dicts with "g", "-", "+"
        """
        if self.cov_estimator == "all":
            raise ValueError("g() cannot be used when cov_estimator='all', set cov_estimator to a single estimator.")
        check_is_fitted(self)

        X_mm, _ = self._get_model_matrix(X)
        n = len(X)
        z = norm.ppf(1 - alpha / 2)

        all_params = self._get_params(self.params_, X_mm)
        param_names = ("a", "t0", "v", "z")

        bands = {}
        idx = 0
        for name, X_p, param_val in zip(param_names, X_mm, all_params):
            g_hat = np.full(n, param_val) if np.isscalar(param_val) else np.asarray(param_val)

            if X_p is None:
                lower, upper = g_hat, g_hat
            else:
                n_features = X_p.shape[1]
                cov_p = self.covariance_[idx : idx + n_features, idx : idx + n_features]
                se = np.sqrt(np.einsum("ij,jk,ik->i", X_p, cov_p, X_p))
                lower, upper = g_hat - z * se, g_hat + z * se
                idx += n_features

            bands[name] = {"g": g_hat, "-": lower, "+": upper}

        return bands
