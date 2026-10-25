#!/usr/bin/env python

"""Figure 06 C, D: Application to Rat Decision Making"""

import argparse
import json
from pathlib import Path

import fig05to06
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from formulaic import model_matrix
from tqdm.auto import tqdm

from drift_diffusion.model import DriftDiffusionModel

with open("./config.json") as f:
    rc_params = json.load(f)["rc-params"]
    rc_params["figure.constrained_layout.w_pad"] = 0
    rc_params["figure.constrained_layout.wspace"] = 0
    rc_params["font.size"] = 12
    rc_params["xtick.labelsize"] = 12
    rc_params["ytick.labelsize"] = 12
    plt.rcParams.update(rc_params)


def spline_formula(trial):
    """quadratic B-spline in trial (df=3) with knots and bounds fixed by the pooled trials, so every fit and
    prediction shares one basis; beyond the bounds the basis is held at its boundary value."""

    lower, knot, upper = np.percentile(trial, [0, 50, 100])
    return f"bs(trial, knots=[{knot:g}], degree=2, lower_bound={lower:g}, upper_bound={upper:g}, extrapolation='clip')"


def fit_ddm_pooled(df, spline, path=Path("../data/fig06.npy")):
    """estimate a, t0, z, and the coherence slope of v as spline functions of trial in day;
    warm-started from a previous fit saved at path (delete it to fit from the defaults)"""

    params0 = np.load(path) if path.exists() else None
    if params0 is not None and len(params0) != 16:  # 4 coefficients per parameter; else stale
        params0 = None

    ddm = DriftDiffusionModel(
        a=spline,
        t0=spline,
        v=f"0 + coherence + coherence:{spline}",
        z=spline,
        cov_estimator="autocorrelation-robust",
        verbose=True,
    )
    return ddm.fit(df, df["y"], params0=params0)


def fit_ddm_splines(df, spline, params):
    """estimate DDM parameters at trial 400 by day + 95% CI under the CLT, fixing the pooled spline coefficients"""

    param_names = ["a", "t0", "beta_v", "z"]
    param_slices = np.split(params, 4)  # (intercept or coherence slope, 3 spline coefficients) per parameter

    X_ref = model_matrix(spline, pd.DataFrame({"trial": [400]}), output="pandas")
    spline_cols = [col for col in X_ref.columns if col != "Intercept"]
    b_at_400 = X_ref[spline_cols].values[0]  # shape (3,)

    fixed = [
        {(f"coherence:{col}" if name == "beta_v" else col): val for col, val in zip(spline_cols, values[1:])}
        for name, values in zip(param_names, param_slices)
    ]
    ddm = DriftDiffusionModel(
        a={"formula": spline, "fixed": fixed[0]},
        t0={"formula": spline, "fixed": fixed[1]},
        v={"formula": f"-1 + coherence + coherence:{spline}", "fixed": fixed[2]},
        z={"formula": spline, "fixed": fixed[3]},
        cov_estimator="autocorrelation-robust",
    )

    results = []
    for day, grp in tqdm(df.groupby("day"), desc="fit ddm splines by day"):
        ddm.fit(grp, grp["y"])
        se = np.sqrt(np.diag(ddm.covariance_))

        res = {"day": day}
        for name, beta0, se0, values in zip(param_names, ddm.params_, se, param_slices):
            val_at_400 = beta0 + values[1:] @ b_at_400
            res[name] = val_at_400
            res[f"{name}+"] = val_at_400 + 1.96 * se0
            res[f"{name}-"] = val_at_400 - 1.96 * se0

        results.append(res)

    return pd.DataFrame(results).set_index("day")


def main(subset=False):
    """fig06"""

    # load + preprocess
    df195 = fig05to06.preproc_df()

    if subset:
        df195 = df195.iloc[::5]

    # subsets used across figure panels
    day_q = "trial > 0"
    trial_q = "day > 35 and trial > 0 and trial <= 800"
    ddm_cols = ["a", "t0", "beta_v", "z"]
    spline = spline_formula(df195.query(trial_q)["trial"])

    # pooled spline fit (fig06d), whose spline coefficients are fixed in the by-day fits (fig06c)
    ddm = fit_ddm_pooled(df195.query(trial_q), spline)
    if not subset:
        np.save("../data/fig06.npy", ddm.params_)

    # fig 06c
    label_map = {"a": r"$a$", "t0": r"$t_0$", "beta_v": r"$\beta_v$", "v": r"$\beta_v$", "z": r"$z$"}
    ddm_fitby_day = fit_ddm_splines(df195.query(day_q), spline, ddm.params_)
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(7, 4.4), width_ratios=[10, 0.6, 1.5], sharex="col")
    fig.align_ylabels(axs)
    for row, col in enumerate(ddm_cols):
        fig05to06.plot_estimates(axs=axs[row], df=ddm_fitby_day, fitby="day", col=col)
        axs[row, 0].set_ylabel(label_map[col], rotation=0, fontsize=16)
    axs[0, 2].set(title="ACF")
    axs[3, 0].set(xlabel="Day in Study")
    axs[3, 2].set(xlabel="Lag(Day)")
    fig.savefig("../results/fig06c")

    # fig 06d: v at coherence 1 is the coherence slope beta_v
    trial = np.arange(20, 801, 10)
    idx_400 = np.where(trial == 400)[0][0]
    coherence = np.repeat(1, len(trial))
    X = pd.DataFrame({"coherence": coherence, "trial": trial})
    y_limits = fig05to06.get_y_limits()
    # bonferroni correction for simultaneous confidence bands
    g = ddm.g(X, alpha=0.05 / df195.query(trial_q)["trial"].nunique())
    fig, axs = plt.subplots(nrows=4, figsize=(5.45, 4.4), sharex=True)
    for row, col in enumerate(["a", "t0", "v", "z"]):
        axs[row].fill_between(trial, g[col]["-"], g[col]["+"], color="k", alpha=0.25)
        axs[row].plot(trial, g[col]["g"], color="k")
        axs[row].set(ylim=y_limits[col], yticklabels=[], xlim=[-50, 850], xticks=[0, 200, 400, 600, 800])
        axs[row].plot(400, g[col]["g"][idx_400], "ko", ms=5)
        axs[row].axvline(x=400, color="k", ls="--", alpha=0.5)
        axs[row].axhline(y=g[col]["g"][idx_400], color="k", ls="--", alpha=0.5)
    axs[3].set(xlabel="Trial in Day")
    fig.savefig("../results/fig06d")


if __name__ == "__main__":
    """set script defaults"""

    parser = argparse.ArgumentParser(description="Figure 06 C, D: Application to Rat Decision Making")
    parser.add_argument("--subset", action="store_true", help="use subset of data for testing")
    args = parser.parse_args()

    main(subset=args.subset)
