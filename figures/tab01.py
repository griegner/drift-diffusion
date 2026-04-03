import time

import numpy as np
from hssm import HSSM

from drift_diffusion.model import DriftDiffusionModel


def fit_mle(X, y):
    """MLE with non-robust uncertainties"""
    mle = DriftDiffusionModel(cov_estimator="sample-hessian", p_outlier=1e-12)
    t0 = time.time()
    mle.fit(X, y, params0=np.array([1.0, 0.1, 0.1, 0.1]))
    runtime = time.time() - t0
    return runtime, mle.params_, np.sqrt(np.diag(mle.covariance_))


def fit_mcmc(y_df):
    """MCMC with non-robust uncertainties"""
    mcmc = HSSM(data=y_df, model="ddm")
    t0 = time.time()
    mcmc.sample(cores=1, quiet=True, initvals={"a": 1.0, "t": 0.1, "v": 0.1, "z": 0.55})
    runtime = time.time() - t0
    params_, unc_ = mcmc.summary().loc[["a", "t", "v", "z"], ["mean", "sd"]].to_numpy().T
    params_[-1], unc_[-1] = 2 * params_[-1] - 1, 2 * unc_[-1]  # rescale z to (-1, 1)
    return runtime, params_, unc_


def get_parameter_bias(estimates, true_values):
    """mean standardized point estimate bias"""
    true_values = np.asarray(true_values)
    se = estimates.std(axis=0)
    return np.mean(np.abs((estimates - true_values) / se))


def get_uncertainty_bias(estimates, true_values):
    """mean standardized uncertainty bias"""
    true_values = np.asarray(true_values)
    return np.mean(np.abs((estimates - true_values) / true_values))
