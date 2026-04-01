"""figure 03: data analysis results"""

import numpy as np
import pandas as pd
from formulaic import model_matrix
from joblib import Parallel, delayed
from sklearn.base import clone
from statsmodels.api import OLS
from statsmodels.tsa.stattools import acf

from drift_diffusion.model import DriftDiffusionModel


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


def fit_ddm(df, fitby, refit=False):
    """estimates DDM parameters + 95% CI under the CLT"""

    if not refit:  # load pre estimated params + 95% CI
        return pd.read_csv(f"results/ddm-fit-by-{fitby}.csv", index_col=0)

    ddm = DriftDiffusionModel(a="+1", t0="+1", v="-1 + coherence", z="+1", cov_estimator="autocorrelation-robust")
    param_names = ["a", "t0", "beta_v", "z"]

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

    groups = df.groupby(fitby)[["coherence", "y"]]
    results = Parallel(n_jobs=-3)(delayed(_fit_ddm)(name, group) for name, group in groups)

    return pd.DataFrame({key: series for key, series in results}).T


def fit_ddm_splines(df, refit=False):
    """estimate DDM parameters as functions of trial by day + 95% CI under the CLT"""

    if not refit:
        return pd.read_csv("results/ddm-fit-by-day.csv", index_col=0)

    param_names = ["a", "t0", "beta_v", "z"]
    splines = np.load("results/ddm-splines-params.npy")
    param_slices = [splines[:4], splines[4:8], splines[8:13], splines[13:]]

    results = []
    for day, grp in df.query("trial > 0").groupby("day"):
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
