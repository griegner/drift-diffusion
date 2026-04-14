#!/usr/bin/env python

import argparse
import json

import fig01
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from formulaic import model_matrix
from joblib import Parallel, delayed
from sklearn.base import clone
from statsmodels.api import OLS
from statsmodels.tsa.stattools import acf
from tqdm.auto import tqdm

from drift_diffusion.model import DriftDiffusionModel

with open("./code_ocean/code/config.json") as f:
    rc_params = json.load(f)["rc-params"]
    plt.rcParams.update(rc_params)


def get_y_limits():
    with open("./code_ocean/code/config.json") as f:
        return json.load(f)["y-limits"]


def fit_beh(df, fitby):
    """estimate mean reaction time and accuracy + 95% CI under the CLT"""
    mean_sem = df.groupby(fitby)[["RT", "correct"]].agg(["mean", "sem"])

    rt_ci = 1.96 * mean_sem[("RT", "sem")]  # CLT
    acc_ci = 1.96 * mean_sem[("correct", "sem")]  # CLT

    result = pd.DataFrame(
        {
            "rt": mean_sem[("RT", "mean")],
            "acc": mean_sem[("correct", "mean")],
            "rt+": mean_sem[("RT", "mean")] + rt_ci,
            "rt-": mean_sem[("RT", "mean")] - rt_ci,
            "acc+": mean_sem[("correct", "mean")] + acc_ci,
            "acc-": mean_sem[("correct", "mean")] - acc_ci,
        }
    )
    return result


def fit_ddm(df, fitby, prefer="threads"):
    """estimates DDM parameters + 95% CI under the CLT"""

    ddm = DriftDiffusionModel(a="+1", t0="+1", v="-1 + coherence", z="+1", cov_estimator="autocorrelation-robust")
    param_names = ["a", "t0", "beta_v", "z"]

    @delayed
    def _fit_ddm(group_key, grp):
        ddm_clone = clone(ddm)
        ddm_clone.fit(grp, grp["y"])
        params = ddm_clone.params_
        se = np.sqrt(np.diag(ddm_clone.covariance_))

        results = {}
        for i, name in enumerate(param_names):
            results[name] = params[i]
            results[f"{name}+"] = params[i] + 1.96 * se[i]
            results[f"{name}-"] = params[i] - 1.96 * se[i]
        return group_key, pd.Series(results)

    groups = list(df.groupby(fitby)[["coherence", "y"]])
    with Parallel(n_jobs=-1, prefer=prefer) as parallel:
        results = parallel(_fit_ddm(name, group) for name, group in tqdm(groups, desc=f"fit ddm by {fitby}"))

    return pd.DataFrame({key: series for key, series in results}).T


def plot_heatmap(ax, df, fitby):
    """plot heatmap of trial counts"""
    if fitby == "trial":
        x, y = df["trial"], df["day"]
        ax.set(yticks=[0, 50, 100], ylabel="Day")
    else:  # fitby == "day"
        x, y = df["day"], (df["hour"] - 18) % 24  # start at 18:00
        ax.set(yticks=[0, 6, 12, 18, 24], yticklabels=["18", "24", "6", "12", "18"], ylabel="Hour")
    ax.scatter(x, y, c=df["RT"], marker="s", s=10, cmap="YlOrRd", vmin=-400)
    ax.invert_yaxis()


def plot_estimates(axs, df, fitby, col, formula=None):
    """plot estimates with CI, histogram, and ACF"""

    axs[0].sharey(axs[1])

    y_limits = get_y_limits()
    if col in y_limits:
        axs[0].set_ylim(y_limits[col])

    axs[0].vlines(df.index, df[f"{col}-"], df[f"{col}+"], color="k", alpha=0.5, lw=2)
    axs[0].scatter(df.index, df[col], color="k", s=2)

    if fitby == "trial":
        axs[0].set(xlim=[-50, 850], xticks=[0, 200, 400, 600, 800])
        df_valid = df.query("index > 0")[[col]].dropna()
        y, X = model_matrix(f"{col} ~ {formula}", df_valid.reset_index(names="x"), output="numpy")
        model = OLS(y[:, 0], X).fit()
        coeffs = model.params
        residuals = model.resid
        axs[0].plot(df_valid.index, X @ coeffs, color="k", lw=0.5)

    axs[0].set_ylabel(col)

    axs[1].hist(df[col], color="gray", bins=25, orientation="horizontal")
    axs[1].axis("off")

    n_lags = len(df) // 3
    acf_input = residuals if fitby == "trial" else df.iloc[17:][col]
    acf_, confint = acf(acf_input, nlags=n_lags, fft=True, bartlett_confint=True, missing="conservative", alpha=0.05)
    lags = -np.arange(n_lags + 1)[::-1]
    axs[2].axhline(y=0, color="k", linestyle="--", lw=1)
    axs[2].plot(lags, acf_[::-1], c="k", lw=1)
    axs[2].fill_between(lags, confint[:, 0][::-1], confint[:, 1][::-1], color="gray", alpha=0.3)
    axs[2].set_yticks([0, 1])


def main(prefer="threads", subset=False):
    """fig05,06"""

    # load + preprocess
    df195 = fig01.preproc_df()

    if subset:
        df195 = df195.iloc[::5]

    # print number of trials (single-line)
    n_trials = df195["RT"].count()
    trials_per_day = df195.groupby("day")["RT"].count()
    trials_per_bin = df195.groupby("trial")["RT"].count()
    print(
        f"n trials: {n_trials} \t n trials/day: mean={trials_per_day.mean():.1f}, std={trials_per_day.std():.1f} \t "
        f"n trials/bin: mean={trials_per_bin.mean():.1f}, std={trials_per_bin.std():.1f}"
    )

    # figure 05a
    hm_fitby_day = df195.query("trial > 0").groupby(["day", "hour"], as_index=False)["RT"].count()
    fig, ax = plt.subplots(figsize=(5.8, 1.2), layout="constrained")
    plot_heatmap(ax=ax, df=hm_fitby_day, fitby="day")
    fig.savefig("./code_ocean/results/fig05a1.pdf")

    n_fitby_day = df195.query("trial > 0").groupby(["day"], as_index=False)["RT"].count()
    fig, ax = plt.subplots(figsize=(5.6, 1), layout="constrained")
    ax.scatter(n_fitby_day["day"], n_fitby_day["RT"], c="k", s=4)
    ax.set(yticks=[0, 500, 1000], ylim=[0, 1300], yticklabels=[])
    ax.axhline(500, ls="--", lw=1, c="k")
    fig.savefig("./code_ocean/results/fig05a2.pdf")

    # figure 05b
    wsf_fitby_trial = df195.query("day > 35 and trial <= 800").groupby("trial")["reward"].mean().cumsum()
    fig, axs = plt.subplots(nrows=2, figsize=(5.6, 1.8), layout="constrained", sharex=True)
    axs[0].scatter(wsf_fitby_trial.index, wsf_fitby_trial.values, c="k", s=4)
    axs[0].set(xlim=[-50, 850], xticks=[0, 200, 400, 600, 800], yticklabels=[])

    n_fitby_trial = df195.query("day > 35 and trial <= 800").groupby(["trial"], as_index=False)["RT"].count()
    axs[1].scatter(n_fitby_trial["trial"], n_fitby_trial["RT"], c="k", s=4)
    axs[1].set(xlim=[-50, 850], xticks=[0, 200, 400, 600, 800], yticks=[0, 500, 1000], ylim=[0, 1300], yticklabels=[])
    axs[1].axhline(500, ls="--", lw=1, c="k")
    fig.savefig("./code_ocean/results/fig05b.pdf")

    # figure 05c
    beh_fitby_day = fit_beh(df=df195.query("trial > 0"), fitby="day")
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(7, 2.2), width_ratios=[10, 0.6, 1.5], sharex="col", layout="constrained"
    )
    plot_estimates(axs=axs[0], df=beh_fitby_day, fitby="day", col="acc")
    plot_estimates(axs=axs[1], df=beh_fitby_day, fitby="day", col="rt")
    fig.savefig("./code_ocean/results/fig05c.pdf")

    # figure 05d
    formula = "bs(x, df=3, degree=2)"
    beh_fitby_trial = fit_beh(df=df195.query("day > 35 and trial <= 800"), fitby="trial")
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(7, 2.2), width_ratios=[10, 0.6, 1.5], sharex="col", layout="constrained"
    )
    plot_estimates(axs=axs[0], df=beh_fitby_trial, fitby="trial", col="acc", formula=formula)
    plot_estimates(axs=axs[1], df=beh_fitby_trial, fitby="trial", col="rt", formula=formula)
    fig.savefig("./code_ocean/results/fig05d.pdf")

    # fig 06a
    ddm_fitby_day = fit_ddm(df=df195.query("trial > 0"), fitby="day", prefer=prefer)
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(7, 4.4), width_ratios=[10, 0.6, 1.5], sharex="col", layout="constrained"
    )
    plot_estimates(axs=axs[0], df=ddm_fitby_day, fitby="day", col="a")
    plot_estimates(axs=axs[1], df=ddm_fitby_day, fitby="day", col="t0")
    plot_estimates(axs=axs[2], df=ddm_fitby_day, fitby="day", col="beta_v")
    plot_estimates(axs=axs[3], df=ddm_fitby_day, fitby="day", col="z")
    fig.savefig("./code_ocean/results/fig06a.pdf")

    # fig06b
    ddm_fitby_trial = fit_ddm(df=df195.query("day > 35 and trial <= 800"), fitby="trial", prefer=prefer)
    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(7, 4.4), width_ratios=[10, 0.6, 1.5], sharex="col", layout="constrained"
    )
    plot_estimates(axs=axs[0], df=ddm_fitby_trial, fitby="trial", col="a", formula=formula)
    plot_estimates(axs=axs[1], df=ddm_fitby_trial, fitby="trial", col="t0", formula=formula)
    plot_estimates(axs=axs[2], df=ddm_fitby_trial, fitby="trial", col="beta_v", formula=formula)
    plot_estimates(axs=axs[3], df=ddm_fitby_trial, fitby="trial", col="z", formula=formula)
    fig.savefig("./code_ocean/results/fig06b.pdf")


if __name__ == "__main__":
    """set script defaults"""
    parser = argparse.ArgumentParser(description="Figures 05-06")
    parser.add_argument("--prefer", choices=["processes", "threads"], default="threads", help="joblib parallel backend")
    parser.add_argument("--subset", action="store_true", help="use subset of data for testing")
    args = parser.parse_args()
    main(prefer=args.prefer, subset=args.subset)
