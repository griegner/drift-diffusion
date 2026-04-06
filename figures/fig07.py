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


def bias_sd_rmse(params, params_):
    """params (true values), params_ (matrix of estimates)"""
    params_mean_ = params_.mean(axis=0)
    bias_ = np.mean(params_ - params, axis=0)
    sd_ = np.sqrt(np.mean((params_ - params_mean_) ** 2, axis=0))
    rmse_ = np.sqrt(np.mean((params_ - params) ** 2, axis=0))
    return bias_, sd_, rmse_


def summarize_method(n, method, runtime, params, params_, uncs_):
    """params (true key-values), params_ (matrix of estimates)"""
    param_names = list(params)
    true_params = np.asarray(list(params.values()))
    runtime_mean = float(np.mean(runtime))
    metrics = {
        "param": bias_sd_rmse(true_params, params_),
        "unc": bias_sd_rmse(params_.std(axis=0), uncs_),
    }
    return [
        {
            "n": n,
            "method": method,
            "estimate": estimate,
            "param": param_name,
            "runtime": runtime_mean,
            "bias": bias_value,
            "sd": sd_value,
            "rmse": rmse_value,
        }
        for estimate, (bias_, sd_, rmse_) in metrics.items()
        for param_name, bias_value, sd_value, rmse_value in zip(param_names, bias_, sd_, rmse_)
    ]
