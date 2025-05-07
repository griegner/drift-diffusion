"""utility functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
from statsmodels.tsa.stattools import acf

from drift_diffusion.model import DriftDiffusionModel


def mat_to_pd(mat):
    """load matlab file and select trial variables"""
    mat = loadmat(mat)
    mat = {k: v.squeeze() for k, v in mat.items() if isinstance(v, np.ndarray) and v.shape == mat["RT"].shape}
    return pd.DataFrame(mat)


def fit_ddm(df, groupby_col):
    """fit DDM by grouing column"""
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


def plot_heatmap_and_fits(df_heatmap, fit_df, x, y, figsize=(14, 4)):
    """plot heatmap of coherences and parameter estimates"""
    fig, axs = plt.subplots(nrows=3, figsize=figsize, layout="constrained", sharex=True)

    # cmap = LinearSegmentedColormap.from_list("cmap", ["#CBCBCB", "#000000"])
    cmap = "YlOrRd"
    vline_kwargs = dict(color="k", lw=1.5)
    hist_kwargs = dict(color="gray", bins=25, orientation="horizontal")
    text_kwargs = dict(fontsize=9, alpha=0.75)

    # heatmap
    sc = axs[0].scatter(
        df_heatmap[x], df_heatmap[y], c=df_heatmap["coherence"], marker="s", cmap=cmap, s=4, vmin=0, vmax=1
    )
    if y == "hour":
        axs[0].set_yticks([1, 6, 12, 18, 24])
    axs[0].set_ylabel(y)

    cbar = fig.colorbar(sc, ax=axs[0], location="top", ticks=(0, 1), anchor=(1, 1), fraction=0.06, aspect=15)
    cbar.set_label("coherence", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    # scatterplot
    ax_sc = axs[0].inset_axes([1.24, 0, 0.3, 1])
    ax_sc.scatter(fit_df["a"], fit_df["beta_v"], s=1, color="k")
    ax_sc.set_xlabel(r"$\hat{a}$")
    ax_sc.set_ylabel(r"$\hat{\beta}_v$")
    ax_sc.set_xlim([0.62, 1.66])  # a
    ax_sc.set_ylim([-1.34, 3.48])  # beta_v
    fit_df_scatter = fit_df[["a", "beta_v"]].dropna()
    corr = np.corrcoef(fit_df_scatter["a"], fit_df_scatter["beta_v"])[0, 1]
    ax_sc.text(
        0.98,
        0.02,
        rf"$\text{{corr}}(\hat{{a}},\hat{{\beta}}_v)={corr:.2f}$",
        ha="right",
        va="bottom",
        transform=ax_sc.transAxes,
        fontsize=9,
        alpha=0.75,
    )

    # a
    axs[1].vlines(fit_df.index, fit_df["a-"], fit_df["a+"], **vline_kwargs)
    axs[1].set_ylabel(r"$\hat{a}$")
    axs[1].set_ylim([0.62, 1.66])

    # a hist
    ax_inset = axs[1].inset_axes([1, 0, 0.06, 1])
    ax_inset.hist(fit_df["a"], **hist_kwargs)
    ax_inset.set_ylim(axs[1].get_ylim())
    ax_inset.axis("off")
    label = rf"$\bar{{a}}$={fit_df["a"].mean():.2f}" "\n  " rf"$\pm${fit_df["a"].std():.2f}"
    ax_inset.text(0.2, 0.95, label, va="top", transform=ax_inset.transAxes, **text_kwargs)

    ax_acf_a = ax_inset.inset_axes([4, 0, 5, 1])
    ax_acf_a.plot(acf(fit_df["a"], nlags=50, fft=True, bartlett_confint=False, missing="conservative"), c="k", lw=1)
    ax_acf_a.set_ylabel(r"$\text{corr}(\hat{a})$")

    # beta_v
    axs[2].vlines(fit_df.index, fit_df["beta_v-"], fit_df["beta_v+"], **vline_kwargs)
    axs[2].set_ylabel(r"$\hat{\beta}_v$")
    axs[2].set_xlabel(x)
    axs[2].set_ylim([-1.34, 3.48])
    if x == "hour":
        axs[2].set_xticks([1, 6, 12, 18, 24])

    # beta_v hist
    ax_inset = axs[2].inset_axes([1, 0, 0.06, 1])
    ax_inset.hist(fit_df["beta_v"], **hist_kwargs)
    ax_inset.set_ylim(axs[2].get_ylim())
    ax_inset.axis("off")
    label = rf"$\bar{{\beta}}_v$={fit_df["beta_v"].mean():.2f}" "\n   " rf"$\pm${fit_df["beta_v"].std():.2f}"
    ax_inset.text(0.2, 0.05, label, va="bottom", transform=ax_inset.transAxes, **text_kwargs)

    ax_acf_beta_v = ax_inset.inset_axes([4, 0, 5, 1])
    ax_acf_beta_v.plot(
        acf(fit_df["beta_v"], nlags=50, fft=True, bartlett_confint=False, missing="conservative"), c="k", lw=1
    )
    ax_acf_beta_v.set_xlabel(f"lag ({x})")
    ax_acf_beta_v.set_ylabel(r"$\text{corr}(\hat{\beta}_v)$")
