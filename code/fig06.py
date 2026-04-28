#!/usr/bin/env python

"""Figure 06 C, D: Application to Rat Decision Making"""

import argparse
import json

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


def fit_ddm_splines(df):
    """estimate DDM parameters as functions of trial by day + 95% CI under the CLT"""

    spline_formula = "bs(trial, df=3, degree=2)"
    param_names = ["a", "t0", "beta_v", "z"]
    param_slices = np.split(np.load("../data/fig06.npy"), [4, 8, 13])

    trial_ref = 400
    results = []
    for day, grp in tqdm(df.groupby("day"), desc="fit ddm splines by day"):
        X_mm = model_matrix(spline_formula, grp, output="pandas", na_action="raise")
        spline_cols = [col for col in X_mm.columns if col != "Intercept"]

        ref_row = pd.DataFrame({"trial": [trial_ref]})
        X_ref = model_matrix(spline_formula, ref_row, output="pandas")
        b_at_400 = X_ref[spline_cols].values[0]  # shape (3,)

        fixed = [
            {col: val for col, val in zip(spline_cols, values[2:] if name == "beta_v" else values[1:])}
            for name, values in zip(param_names, param_slices)
        ]

        ddm = DriftDiffusionModel(
            a={"formula": spline_formula, "fixed": fixed[0]},
            t0={"formula": spline_formula, "fixed": fixed[1]},
            v={"formula": "-1 + coherence + bs(trial, df=3, degree=2)", "fixed": fixed[2]},
            z={"formula": spline_formula, "fixed": fixed[3]},
        )
        ddm.fit(grp, grp["y"])
        params = ddm.params_
        se = np.sqrt(np.diag(ddm.covariance_))

        res = {"day": day}
        for name, beta0, se0, values in zip(param_names, params, se, param_slices):
            val_at_400 = beta0 + (values[2:] if name == "beta_v" else values[1:]) @ b_at_400
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

    # fig 06c
    label_map = {"a": r"$a$", "t0": r"$t_0$", "beta_v": r"$\beta_v$", "v": r"$\beta_v$", "z": r"$z$"}
    ddm_fitby_day = fit_ddm_splines(df195.query(day_q))
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(7, 4.4), width_ratios=[10, 0.6, 1.5], sharex="col")
    fig.align_ylabels(axs)
    for row, col in enumerate(ddm_cols):
        fig05to06.plot_estimates(axs=axs[row], df=ddm_fitby_day, fitby="day", col=col)
        axs[row, 0].set_ylabel(label_map[col], rotation=0, fontsize=16)
    axs[0, 2].set(title="ACF")
    axs[3, 0].set(xlabel="Day in Study")
    axs[3, 2].set(xlabel="Lag(Day)")
    fig.savefig("../results/fig06c")

    # fig 06d
    ddm = DriftDiffusionModel(
        a="bs(trial, df=3, degree=2)",
        t0="bs(trial, df=3, degree=2)",
        v="coherence + bs(trial, df=3, degree=2)",
        z="bs(trial, df=3, degree=2)",
        cov_estimator="autocorrelation-robust",
        verbose=True,
    )
    ddm.fit(df195.query(trial_q), df195.query(trial_q)["y"], params0=np.load("../data/fig06.npy"))

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
