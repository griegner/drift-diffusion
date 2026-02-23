import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def iid_params(n_samples, params, params_s, seed=1):
    rng = np.random.default_rng(seed=1)
    t0_iid = rng.uniform(params["t0"] - params_s["t0"] / 2, params["t0"] + params_s["t0"] / 2, size=n_samples)
    v_iid = rng.normal(params["v"], params_s["v"], size=n_samples)
    z_iid = rng.uniform(params["z"] - params_s["z"] / 2, params["z"] + params_s["z"] / 2, size=n_samples)
    return t0_iid, v_iid, z_iid


def cov_to_corr(cov):
    "ii to standard errors, ij to correlations"
    stderr = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stderr, stderr)
    np.fill_diagonal(corr, stderr)
    return corr
