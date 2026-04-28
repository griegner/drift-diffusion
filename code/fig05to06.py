#!/usr/bin/env python

"""Figures 05-06: Application to Rat Decision Making"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from formulaic import model_matrix
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.base import clone
from statsmodels.api import OLS
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tsa.stattools import acf
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


def get_y_limits():
    """match y limits between figures"""

    with open("./config.json") as f:
        return json.load(f)["y-limits"]


def mat_to_pd(mat):
    """load matlab file and select trial variables"""

    mat = loadmat(mat)
    mat = {k: v.squeeze() for k, v in mat.items() if isinstance(v, np.ndarray) and v.shape == mat["RT"].shape}
    return pd.DataFrame(mat)


def preproc_df(path="../data/Rat195Vectors_241025.mat"):
    """load and preprocess the dataframe for rat195 from Reinagel 2013"""

    R, L = 0.0, np.pi  # dot movement
    df = (
        mat_to_pd(path)
        .query("Valid == 1 and RT == RT")  # keep valid trials and non-null RT
        .assign(trialDate=lambda x: pd.to_datetime(x["trialDate"] - 719529, unit="D"))
        .set_index("trialDate")
        .sort_index()
        .loc["2008-12-03":"2009-03-12"]  # select 24h sessions with constant |coh|
    )

    shifted_index = (df.index - pd.Timedelta(hours=18)).floor("D")  # 6pm-to-6pm days
    all_days = pd.date_range(shifted_index.min(), shifted_index.max(), freq="D")
    day_map = {day: i + 1 for i, day in enumerate(all_days)}

    df = df.assign(
        LR=lambda x: x["dotDirection"].map({R: +1, L: -1}) * x["correct"].map({1: 1, 0: -1}),  # +1 R, -1 L choice
        y=lambda x: x["RT"] * x["LR"],  # signed RT
        coherence=lambda x: x["dotDirection"].map({R: +1, L: -1}) * x["coherence"],  # +coh R, -coh L dot movement
        day=shifted_index.map(day_map),  # 6pm-to-6pm day
        hour=lambda x: x.index.hour + 1,  # 1 to 24
        trial=lambda x: ((x.groupby("day").cumcount() + 1) // 20) * 20,  # trials in day
        reward=lambda x: x["correct"] * x["proposedReward"],
    )

    return df


def fit_beh(df, fitby):
    """estimate mean reaction time +/- CLT 95% CI and accuracy +/- binomial (Wilson) 95% CI"""

    g = df.groupby(fitby)

    rt_mean = g["RT"].mean()
    rt_sem = g["RT"].sem()
    rt_ci = 1.96 * rt_sem  # CLT for RT mean

    n_correct = g["correct"].sum()
    n_trials = g["correct"].count()
    acc = n_correct / n_trials
    acc_low, acc_upr = proportion_confint(n_correct, n_trials, alpha=0.05, method="wilson")

    return pd.DataFrame(
        {
            "rt": rt_mean,
            "acc": acc,
            "rt+": rt_mean + rt_ci,
            "rt-": rt_mean - rt_ci,
            "acc+": acc_upr,
            "acc-": acc_low,
        }
    )


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

        results = {
            key: value
            for i, name in enumerate(param_names)
            for key, value in (
                (name, params[i]),
                (f"{name}+", params[i] + 1.96 * se[i]),
                (f"{name}-", params[i] - 1.96 * se[i]),
            )
        }
        return group_key, pd.Series(results)

    groups = list(df.groupby(fitby)[["coherence", "y"]])
    with Parallel(n_jobs=-1, prefer=prefer) as parallel:
        results = parallel(_fit_ddm(name, group) for name, group in tqdm(groups, desc=f"fit ddm by {fitby}"))

    return pd.DataFrame({key: series for key, series in results}).T


def plot_estimates(axs, df, fitby, col, formula=None):
    """plot estimates with CI, histogram, and ACF"""

    axs[0].sharey(axs[1])

    y_limits = get_y_limits()
    if col in y_limits:
        axs[0].set_ylim(y_limits[col])

    main_colors = "k"
    if fitby == "day":
        main_colors = np.where(df.index <= 35, "red", "k")
    elif fitby == "trial":
        main_colors = np.full(len(df), "k", dtype=object)
        if len(df):
            main_colors[0] = "red"

    axs[0].vlines(df.index, df[f"{col}-"], df[f"{col}+"], color=main_colors, alpha=0.5, lw=2)
    axs[0].scatter(df.index, df[col], color=main_colors, s=2)

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
    df195 = preproc_df()

    if subset:
        df195 = df195.iloc[::5]

    # subsets used across figure panels
    n_trials = df195["RT"].count()
    print(f"n Trials = {n_trials}")
    day_q = "trial > 0"
    trial_q = "day > 35 and trial <= 800"
    ddm_cols = ["a", "t0", "beta_v", "z"]

    # figure 05a
    hm_fitby_day = df195.query(day_q).groupby(["day", "hour"], as_index=False)["RT"].count()
    fig, ax = plt.subplots(figsize=(5.8, 1.1))
    ax.set(ylabel="Hour in Day", yticks=[0, 6, 12, 18, 24], yticklabels=["18", "24", "6", "12", "18"], xticklabels=[])
    hour = (hm_fitby_day["hour"] - 18) % 24
    ax.scatter(hm_fitby_day["day"], hour, c=hm_fitby_day["RT"], marker="s", s=10, cmap="YlOrRd", vmin=-400)
    ax.invert_yaxis()
    fig.savefig("../results/fig05a1")

    n_fitby_day = df195.query(day_q).groupby("day", as_index=False)["RT"].count()
    fig, ax = plt.subplots(figsize=(5.8, 1.4))
    day_colors = np.where(n_fitby_day["day"] <= 35, "red", "k")
    ax.scatter(n_fitby_day["day"], n_fitby_day["RT"], c=day_colors, s=4)
    ax.set(yticks=[0, 500, 1000], ylim=[0, 1300], xlabel=r"Day in Study", ylabel="n Trials")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.axhline(500, ls="--", lw=1, c="k")
    n_trials = n_fitby_day["RT"].mean()
    sd_trials = n_fitby_day["RT"].std()
    ax.text(
        1, 1.01, f"n Trials = {n_trials:.0f} $\\pm$ {sd_trials:.0f}", transform=ax.transAxes, ha="right", va="bottom"
    )
    fig.savefig("../results/fig05a2")

    # figure 05b
    wsf_fitby_trial = df195.query(trial_q).groupby("trial")["reward"].mean().cumsum()
    fig, axs = plt.subplots(nrows=2, figsize=(5.8, 2.4), sharex=True)
    fig.align_ylabels(axs)
    trial_colors = np.where(np.arange(len(wsf_fitby_trial)) == 0, "red", "k")
    axs[0].scatter(wsf_fitby_trial.index, wsf_fitby_trial.values, c=trial_colors, s=4)
    axs[0].set(xlim=[-50, 850], xticks=[0, 200, 400, 600, 800], ylabel="Cumulative\nReward (a.u.)")
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    n_fitby_trial = df195.query(trial_q).groupby("trial", as_index=False)["RT"].count()
    axs[1].scatter(n_fitby_trial["trial"], n_fitby_trial["RT"], c=trial_colors, s=4)
    axs[1].set(xlim=[-50, 850], ylim=[0, 1300], ylabel="n Obs.", xlabel="Trial in Day")
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[1].axhline(500, ls="--", lw=1, c="k")
    fig.savefig("../results/fig05b")

    # figure 05c
    beh_fitby_day = fit_beh(df=df195.query(day_q), fitby="day")
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7, 2.5), width_ratios=[10, 0.6, 1.5], sharex="col")
    fig.align_ylabels(axs)
    plot_estimates(axs=axs[0], df=beh_fitby_day, fitby="day", col="acc")
    plot_estimates(axs=axs[1], df=beh_fitby_day, fitby="day", col="rt")
    axs[0, 0].set(ylabel="ACC")
    axs[1, 0].set(ylabel="RT", xlabel="Day in Study")
    axs[0, 2].set(title="ACF")
    axs[1, 2].set(xlabel="Lag(Day)")
    fig.savefig("../results/fig05c")

    # figure 05d
    formula = "bs(x, df=3, degree=2)"
    beh_fitby_trial = fit_beh(df=df195.query(trial_q), fitby="trial")
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6.6, 2.5), width_ratios=[10, 0.6, 1.5], sharex="col")
    plot_estimates(axs=axs[0], df=beh_fitby_trial, fitby="trial", col="acc", formula=formula)
    plot_estimates(axs=axs[1], df=beh_fitby_trial, fitby="trial", col="rt", formula=formula)
    axs[0, 0].set(ylabel=" ", yticklabels=[])
    axs[1, 0].set(ylabel=" ", yticklabels=[], xlabel="Trial in Day")
    axs[0, 2].set(title="ACF")
    axs[1, 2].set(xlabel="Lag(Trial)", xticks=[-10, 0], xticklabels=[-200, 0])
    fig.savefig("../results/fig05d")

    # fig 06a
    label_map = {"a": r"$a$", "t0": r"$t_0$", "beta_v": r"$\beta_v$", "z": r"$z$"}
    ddm_fitby_day = fit_ddm(df=df195.query(day_q), fitby="day", prefer=prefer)
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(7, 4.4), width_ratios=[10, 0.6, 1.5], sharex="col")
    fig.align_ylabels(axs)
    for row, col in enumerate(ddm_cols):
        plot_estimates(axs=axs[row], df=ddm_fitby_day, fitby="day", col=col)
        axs[row, 0].set_ylabel(label_map[col], rotation=0, fontsize=16)
    axs[0, 2].set(title="ACF")
    axs[3, 0].set(xlabel="Day in Study")
    axs[3, 2].set(xlabel="Lag(Day)")
    fig.savefig("../results/fig06a")

    # fig06b
    ddm_fitby_trial = fit_ddm(df=df195.query(trial_q), fitby="trial", prefer=prefer)
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(6.7, 4.4), width_ratios=[10, 0.6, 1.5], sharex="col")
    for row, col in enumerate(ddm_cols):
        plot_estimates(axs=axs[row], df=ddm_fitby_trial, fitby="trial", col=col, formula=formula)
        axs[row, 0].set(ylabel=" ", yticklabels=[])
    axs[0, 2].set(title="ACF")
    axs[3, 0].set(xlabel="Trial in Day")
    axs[3, 2].set(xlabel="Lag(Trial)", xticks=[-10, 0], xticklabels=[-200, 0])
    fig.savefig("../results/fig06b")


if __name__ == "__main__":
    """set script defaults"""

    parser = argparse.ArgumentParser(description="Figures 05-06: Application to Rat Decision Making")
    parser.add_argument("--prefer", choices=["processes", "threads"], default="threads", help="joblib parallel backend")
    parser.add_argument("--subset", action="store_true", help="use subset of data for testing")
    args = parser.parse_args()

    main(prefer=args.prefer, subset=args.subset)
