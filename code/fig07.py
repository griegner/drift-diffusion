#!/usr/bin/env python

"""Figure 07: MLE vs MCMC"""

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
from hssm import HSSM
from joblib import Parallel, delayed

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf

with open("./config.json") as f:
    rc_params = json.load(f)["rc-params"]
    plt.rcParams.update(rc_params)
    so.Plot.config.theme.update(plt.rcParams)


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


def main(n_samples, n_repeats, n_jobs, prefer, csv):
    """fig08"""

    # true parameters and sample sizes to test
    params = {"a": 0.63, "t0": 0.435, "v": 2.23, "z": 0.008}

    @delayed
    def run_simulation(rep, n):
        X = pd.DataFrame({"intercept": np.ones(n)})
        y = sample_from_pdf(**params, n_samples=n, random_state=rep + n)
        y_df = pd.DataFrame({"rt": np.abs(y), "response": np.sign(y)})
        return fit_mle(X, y), fit_mcmc(y_df)

    if csv:
        df = pd.read_csv("data/fig07.csv", index_col=0)
        n_samples = list(df["n"].drop_duplicates())
    else:
        # compare both models across sample sizes using normalized absolute error
        df = []
        for n in n_samples:
            with Parallel(n_jobs=n_jobs, prefer=prefer) as parallel:
                results = parallel(run_simulation(rep, n) for rep in range(n_repeats))
                mle, mcmc = zip(*results)

            for method, values in (("mle", mle), ("mcmc", mcmc)):
                runtime, params_, uncs_ = map(np.stack, zip(*values))
                df.extend(summarize_method(n, method, runtime, params, params_, uncs_))

        df = pd.DataFrame.from_records(df)

    # fig07
    # fig07
    df_plot = (
        df.melt(id_vars=["n", "method", "estimate", "param"], value_vars=["runtime", "rmse"])
        .assign(panel=lambda x: x["variable"].where(x["variable"].eq("runtime"), x["estimate"]))
        .query("not (panel == 'runtime' and param != 'a')")
    )

    (
        so.Plot(df_plot, x="n", y="value", color="param", marker="param", linestyle="method")
        .facet(col="panel")
        .layout(size=(15, 4))
        .add(so.Dot(pointsize=8))
        .add(so.Line())
        .scale(color="binary_r", x=so.Nominal(order=n_samples))
        .share(y=False)
        .label(x="n Trials", y="")
        .save("../results/fig07")
    )


if __name__ == "__main__":
    """set script defaults"""

    parser = argparse.ArgumentParser(description="Figure 07: MLE vs MCMC")
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        default=[500, 1000, 5000, 10000],
        help="number of trials to simulate per repeat (space-separated list of ints)",
    )
    parser.add_argument("--n-repeats", type=int, default=100, help="number of simulation repeats")
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs")
    parser.add_argument(
        "--prefer", choices=["processes", "threads"], default="processes", help="joblib parallel backend"
    )
    parser.add_argument("--csv", action="store_true", help="load precomputed results from data/fig07.csv")
    args = parser.parse_args()

    main(n_samples=args.n_samples, n_repeats=args.n_repeats, n_jobs=args.n_jobs, prefer=args.prefer, csv=args.csv)
