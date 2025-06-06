"""utility functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from statsmodels.tsa.stattools import acf, pacf

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.model.pdf import mdf, pdf

plt.rcParams.update({"font.size": 9})


def mat_to_pd(mat):
    """load matlab file and select trial variables"""
    mat = loadmat(mat)
    mat = {k: v.squeeze() for k, v in mat.items() if isinstance(v, np.ndarray) and v.shape == mat["RT"].shape}
    return pd.DataFrame(mat)


def plot_coherence(df_coh, vlines):
    """plot heatmap of coherence ranges"""
    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.scatter(
        df_coh.index, df_coh["coherence"], c=df_coh["coherence"], s=1, marker="_", lw=0.5, cmap="YlOrRd", vmin=0, vmax=1
    )
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("day")
    ax.set_ylabel("coherence")
    [ax.axvline(pd.to_datetime(vline), c="k", lw=0.75) for vline in vlines]


def fit_rt_acc(df, groupby_col):
    """fit mean reaction time and accuracy by grouping column"""

    def _fit_rt_acc(grp):
        if len(grp) <= 50:  # for CLT
            return pd.Series({k: np.nan for k in ["rt", "acc", "rt+", "rt-", "acc+", "acc-"]})
        mean_sem = grp.agg(["mean", "sem"])
        return pd.Series(
            {
                "rt": mean_sem.loc["mean", "RT"],
                "acc": mean_sem.loc["mean", "correct"],
                "rt+": mean_sem.loc["mean", "RT"] + 1.96 * mean_sem.loc["sem", "RT"],
                "rt-": mean_sem.loc["mean", "RT"] - 1.96 * mean_sem.loc["sem", "RT"],
                "acc+": mean_sem.loc["mean", "correct"] + 1.96 * mean_sem.loc["sem", "correct"],
                "acc-": mean_sem.loc["mean", "correct"] - 1.96 * mean_sem.loc["sem", "correct"],
            }
        )

    return df.groupby(groupby_col)[["RT", "correct"]].apply(_fit_rt_acc)


def fit_ddm(df, groupby_col):
    """fit DDM by grouping column"""
    ddm = DriftDiffusionModel(a="+1", t0=0, v="-1 + coherence", z=0, cov_estimator="sample-hessian")

    def _fit_ddm(grp):
        if len(grp) <= 50:
            return pd.Series({k: np.nan for k in ["a", "beta_v", "a+", "a-", "beta_v+", "beta_v-"]})
        ddm.fit(grp, grp["y"])
        (a, beta_v), (se_a, se_beta_v) = ddm.params_, np.sqrt(np.diag(ddm.covariance_))
        return pd.Series(
            {
                "a": a,
                "beta_v": beta_v,
                "a+": a + 1.96 * se_a,
                "a-": a - 1.96 * se_a,
                "beta_v+": beta_v + 1.96 * se_beta_v,
                "beta_v-": beta_v - 1.96 * se_beta_v,
            }
        )

    return df.groupby(groupby_col)[["coherence", "y"]].apply(_fit_ddm)


def plot_fits(df_heatmap, df_fit, x, y, config):
    """
    plots heatmap and parameter estimates (e.g., DDM params or RT/ACC).
    `config` is a dict with keys:
        - var1, var2: column names in df_fit
        - var1_lim, var2_lim: axis limits
        - var1_label, var2_label: axis labels
        - corr_label, acf_label1, acf_label2: label templates (use {} for variable names)
    """

    mosaic = """
    a.b
    cde
    fgh
    """
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(9, 5), width_ratios=[10, 0.8, 3], layout="constrained")

    # plotting arguments
    cmap = "YlOrRd"
    vline_kwargs = dict(color="k", lw=1.5)
    hist_kwargs = dict(color="gray", bins=25, orientation="horizontal")
    text_kwargs = dict(fontsize=9, alpha=0.75)

    # share x,y
    axs["a"].sharex(axs["f"])
    axs["c"].sharex(axs["f"])
    axs["e"].sharex(axs["h"])
    axs["c"].sharey(axs["d"])
    axs["f"].sharey(axs["g"])
    [axs[key].tick_params(labelbottom=False) for key in ["a", "c"]]

    # limit x,y (between panels)
    axs["b"].set_xlim(config["var1_lim"])
    axs["b"].set_ylim(config["var2_lim"])
    axs["c"].set_ylim(config["var1_lim"])
    axs["f"].set_ylim(config["var2_lim"])

    # (a) heatmap
    sc = axs["a"].scatter(
        df_heatmap[x], df_heatmap[y], c=df_heatmap["coherence"], marker="s", cmap=cmap, s=4, vmin=0, vmax=1
    )
    axs["a"].set_ylabel(y)
    if y == "hour":
        axs["a"].set_yticks([1, 6, 12, 18, 24])

    # (a) colorbar
    cbar = fig.colorbar(sc, ax=axs["a"], location="top", ticks=(0, 1), anchor=(1, 1), fraction=0.06, aspect=15, pad=0)
    cbar.set_label("coherence", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    # (b) scatterplot
    corr_ = df_fit[config["var1"]].corr(df_fit[config["var2"]])
    label = config["corr_label"].format(corr=corr_)
    axs["b"].scatter(df_fit[config["var1"]], df_fit[config["var2"]], s=1, color="k")
    axs["b"].set_title(config["var1_label"], fontsize=9)
    axs["b"].set_ylabel(config["var2_label"])
    axs["b"].text(0.98, 0.02, label, ha="right", va="bottom", transform=axs["b"].transAxes, **text_kwargs)

    # (c) var1 by day
    axs["c"].vlines(df_fit.index, df_fit[f"{config['var1']}-"], df_fit[f"{config['var1']}+"], **vline_kwargs)
    axs["c"].set_ylabel(config["var1_label"])

    # (d) var1 hist
    label = (
        rf"$\bar{{{config['var1']}}}$={df_fit[config['var1']].mean():.2f}"
        "\n  "
        rf"$\pm${df_fit[config['var1']].std():.2f}"
    )
    axs["d"].hist(df_fit[config["var1"]], **hist_kwargs)
    axs["d"].text(0.1, 0.95, label, va="top", transform=axs["d"].transAxes, **text_kwargs)
    axs["d"].axis("off")

    # (e) var1 acf
    n_lags = len(df_fit) // 3
    acf_ = acf(df_fit[config["var1"]], nlags=n_lags, fft=True, bartlett_confint=False, missing="conservative")
    pacf_ = pacf(df_fit[config["var1"]].dropna(), nlags=n_lags)
    axs["e"].plot(acf_, c="k", lw=1, label="acf")
    axs["e"].plot(pacf_, c="gray", lw=1, label="pacf")
    axs["e"].legend()
    axs["e"].set_ylabel(config["acf_label1"])

    # (f) var2 by day
    axs["f"].vlines(df_fit.index, df_fit[f"{config['var2']}-"], df_fit[f"{config['var2']}+"], **vline_kwargs)
    axs["f"].set_ylabel(config["var2_label"])
    axs["f"].set_xlabel(x)
    if x == "hour":
        axs["f"].set_xticks([1, 6, 12, 18, 24])

    # (g) var2 hist
    label = (
        rf"$\bar{{{config['var2']}}}$={df_fit[config['var2']].mean():.2f}"
        "\n   "
        rf"$\pm${df_fit[config['var2']].std():.2f}"
    )
    axs["g"].hist(df_fit[config["var2"]], **hist_kwargs)
    axs["g"].text(0.1, 0.95, label, va="top", transform=axs["g"].transAxes, **text_kwargs)
    axs["g"].axis("off")

    # (h) var2 acf
    acf_ = acf(df_fit[config["var2"]], nlags=n_lags, fft=True, bartlett_confint=False, missing="conservative")
    pacf_ = pacf(df_fit[config["var2"]].dropna(), nlags=n_lags)
    axs["h"].plot(acf_, c="k", lw=1)
    axs["h"].plot(pacf_, c="gray", lw=1)
    axs["h"].set_ylabel(config["acf_label2"])
    axs["h"].set_xlabel("lag (days)")


def plot_mixtures(df, df_fit_by_day, coherence):
    """plots mixture density distributions and accuracy curves"""

    ddm = DriftDiffusionModel(a="+1", t0=0, v="-1 + coherence", z=0, cov_estimator="sample-hessian")
    n = 1000
    y = np.linspace(-4, +4, n)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 2), layout="constrained", width_ratios=[3, 2])

    # empirical
    axs[0].hist(df["y"], bins=90, density=True, color="k", alpha=0.15, label=r"$\hat{f}_H$")
    acc = (
        df["y"]
        .groupby(pd.qcut(np.abs(df["y"]), q=50), observed=True)
        .apply(lambda x: (x > 0).astype(int).mean())
        .rename(index=lambda interval: interval.mid)
    )
    axs[1].scatter(acc.index, acc, c="k", alpha=0.25, s=5)

    # a, v mixture
    weights = df.groupby("day")["y"].count().to_numpy() / len(df)
    a_s = df_fit_by_day["a"].to_numpy()
    v_s = (df_fit_by_day["beta_v"] * coherence).to_numpy()
    f_mix = mdf(y, a=a_s, t0=0, v=v_s, z=0, weights=weights)
    acc = f_mix[n // 2 :] / (f_mix[n // 2 :] + f_mix[: n // 2][::-1])
    axs[0].plot(y, f_mix, color="k", alpha=0.75, label=r"$\hat{f}_\text{MIX}$")
    axs[1].plot(y[n // 2 :], acc, color="k", alpha=0.75)

    # non-mixture
    a, beta_v = ddm.fit(df, df["y"]).params_
    v = beta_v * coherence
    f = pdf(y, a=a, t0=0, v=v, z=0)
    axs[0].plot(y, f, color="k", ls="--", alpha=0.75, label=r"$\hat{f}_\text{DDM}$")
    acc = f[n // 2 :] / (f[n // 2 :] + f[: n // 2][::-1])
    axs[1].plot(y[500:], acc, color="k", ls="--", alpha=0.75)

    # adjust plots
    axs[0].set_title(f"coherence: {coherence}")
    axs[0].set_xlim([-4, 4])
    axs[0].legend()
    axs[1].set_ylim([0.5, 1])
    axs[1].set_xlim([0, 4])
