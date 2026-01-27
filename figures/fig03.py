"""figure 03: data analysis results"""

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
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
        return pd.read_csv(f"results/ddm-fit-by-{fitby}.csv", index_col=0, parse_dates=[0], date_format="%Y-%m-%d")

    ddm = DriftDiffusionModel(a="+1", t0="+1", v="-1 + coherence", z="+1", cov_estimator="autocorrelation-robust")
    param_names = ["a", "t0", "beta_v", "z"]

    def _fit_ddm(group_key, grp):
        if len(grp) <= 50:
            return group_key, pd.Series({k: np.nan for k in param_names})

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


def plot_heatmap(ax, df, fitby):
    """plot heatmap of trial counts"""
    if fitby == "trial":
        x, y = df["trial"], df["day"]
        ax.yaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set(ylabel="Day")
    else:  # fitby == "day"
        x, y = df["day"], (df["hour"] - 18) % 24  # start at 18:00
        ax.set(yticks=[0, 6, 12, 18, 24], yticklabels=["18", "24", "6", "12", "18"], ylabel="Hour")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    ax.scatter(x, y, c=df["RT"], marker="s", s=10, cmap="YlOrRd", vmin=-400)


def plot_estimates(axs, df, fitby, col):
    """plot estimates with CI, histogram, and ACF"""
    if fitby == "day":
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))

    axs[0].sharey(axs[1])

    axs[0].vlines(df.index, df[f"{col}-"], df[f"{col}+"], color="k", alpha=0.5, lw=2)
    axs[0].scatter(df.index, df[col], color="k", s=2)
    mean_val = df[col].mean()
    std_val = df[col].std()
    axs[0].axhline(mean_val, ls="--", color="k", lw=0.2)
    axs[0].axhspan(mean_val - std_val, mean_val + std_val, alpha=0.1, color="gray")

    axs[0].text(0.02, 0.8, f"{mean_val:.2f} ± {std_val:.2f}", transform=axs[0].transAxes, fontsize=10)
    axs[0].set_ylabel(col)

    axs[1].hist(df[col], color="gray", bins=25, orientation="horizontal")
    axs[1].axis("off")

    n_lags = len(df) // 3
    acf_ = acf(df[col], nlags=n_lags, fft=True, bartlett_confint=False, missing="conservative")
    axs[2].plot(-np.arange(n_lags + 1)[::-1], acf_[::-1], c="k", lw=1, label="acf")
    axs[2].set_yticks([0, 1])
