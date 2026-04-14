#!/usr/bin/env python

import argparse
import json

import fig01
import fig05to06
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from formulaic import model_matrix
from tqdm.auto import tqdm

from drift_diffusion.model import DriftDiffusionModel

with open("./code_ocean/code/config.json") as f:
    rc_params = json.load(f)["rc-params"]
    plt.rcParams.update(rc_params)


def fit_ddm_splines(df):
    """estimate DDM parameters as functions of trial by day + 95% CI under the CLT"""

    param_names = ["a", "t0", "beta_v", "z"]
    spline_params = np.load("./code_ocean/data/Rat195Params0.npy")
    param_slices = [spline_params[:4], spline_params[4:8], spline_params[8:13], spline_params[13:]]

    groups = list(df.groupby("day"))
    results = []
    for day, grp in tqdm(groups, desc="fit ddm splines by day"):
        X_mm = model_matrix("bs(trial, df=3, degree=2)", grp, output="pandas", na_action="raise")
        spline_cols = [col for col in X_mm.columns if col != "Intercept"]
        fixed = [{col: val for col, val in zip(spline_cols, values[1:])} for values in param_slices]

        ddm = DriftDiffusionModel(
            a={"formula": "bs(trial, df=3, degree=2)", "fixed": fixed[0]},
            t0={"formula": "bs(trial, df=3, degree=2)", "fixed": fixed[1]},
            v={"formula": "-1 + coherence + bs(trial, df=3, degree=2)", "fixed": fixed[2]},
            z={"formula": "bs(trial, df=3, degree=2)", "fixed": fixed[3]},
        )
        ddm.fit(grp, grp["y"])
        params = ddm.params_
        se = np.sqrt(np.diag(ddm.covariance_))

        res = {"day": day}
        for i, name in enumerate(param_names):
            res[name] = params[i]
            res[f"{name}+"] = params[i] + 1.96 * se[i]
            res[f"{name}-"] = params[i] - 1.96 * se[i]
        results.append(res)

    return pd.DataFrame(results).set_index("day")


def main(subset=False):
    """fig07"""

    # load + preprocess
    df195 = fig01.preproc_df()

    if subset:
        df195 = df195.iloc[::5]

    # datasets used across figure panels
    df195_fig06c = df195.query("trial > 0")
    df195_fig06d = df195.query("day > 35 and trial <= 800")

    # fig 06c
    ddm_fitby_day = fit_ddm_splines(df195_fig06c)
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(7, 4.4), width_ratios=[10, 0.6, 1.5], sharex="col", layout="constrained"
    )
    fig05to06.plot_estimates(axs=axs[0], df=ddm_fitby_day, fitby="day", col="a")
    fig05to06.plot_estimates(axs=axs[1], df=ddm_fitby_day, fitby="day", col="t0")
    fig05to06.plot_estimates(axs=axs[2], df=ddm_fitby_day, fitby="day", col="beta_v")
    fig05to06.plot_estimates(axs=axs[3], df=ddm_fitby_day, fitby="day", col="z")
    fig.savefig("./code_ocean/results/fig06c.pdf")

    # fig 06d
    ddm = DriftDiffusionModel(
        a="bs(trial, df=3, degree=2)",
        t0="bs(trial, df=3, degree=2)",
        v="coherence + bs(trial, df=3, degree=2)",
        z="bs(trial, df=3, degree=2)",
        cov_estimator="autocorrelation-robust",
        verbose=True,
    )
    ddm.fit(df195_fig06d, df195_fig06d["y"], params0=np.load("./code_ocean/data/Rat195Params0.npy"))

    coherence = np.repeat(1, 100)
    trial_min, trial_max = df195_fig06d["trial"].min(), df195_fig06d["trial"].max()
    trial = np.linspace(trial_min, trial_max, 100)
    X = pd.DataFrame({"coherence": coherence, "trial": trial})
    y_limits = fig05to06.get_y_limits()
    # bonferroni correction for simultaneous confidence bands
    g = ddm.g(X, alpha=0.05 / df195_fig06d["trial"].nunique())
    fig, axs = plt.subplots(nrows=4, figsize=(6, 5), sharex=True)
    for i, p in enumerate(["a", "t0", "v", "z"]):
        axs[i].fill_between(trial, g[p]["-"], g[p]["+"], color="k", alpha=0.25)
        axs[i].plot(trial, g[p]["g"], color="k")
        axs[i].set_ylabel(p)
        axs[i].set_ylim(y_limits[p])
    fig.savefig("./code_ocean/results/fig06d.pdf")


if __name__ == "__main__":
    """set script defaults"""
    parser = argparse.ArgumentParser(description="Figure 06")
    parser.add_argument("--subset", action="store_true", help="use subset of data for testing")
    args = parser.parse_args()
    main(subset=args.subset)
