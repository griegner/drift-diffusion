"""utility functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat

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


def plot_heatmap_and_fits(df_heatmap, fit_df, x, y, figsize=(14, 4), cbar=False):
    """plot heatmap of coherences and parameter estimates"""
    fig, axs = plt.subplots(nrows=3, figsize=figsize, layout="constrained", sharex=True)

    cmap = LinearSegmentedColormap.from_list("cmap", ["#CBCBCB", "#000000"])
    vline_kwargs = dict(color="k", lw=1.5)
    hist_kwargs = dict(color="gray", bins=25, orientation="horizontal")

    # heatmap
    sc = axs[0].scatter(
        df_heatmap[x],
        df_heatmap[y],
        c=df_heatmap["coherence"],
        marker="s",
        cmap=cmap,
        s=4,
        vmin=0,
        vmax=1,
    )
    if y == "hour":
        axs[0].set_yticks([1, 6, 12, 18, 24])
    axs[0].set_ylabel(y)

    # heatmap cbar
    if cbar:
        cbar_ax = axs[0].inset_axes([1.005, 0.05, 0.01, 0.9])
        cb = plt.colorbar(sc, cax=cbar_ax)
        cb.set_label("coherence")
        cb.set_ticks([0, 1])

    # a
    axs[1].vlines(fit_df.index, fit_df["a-"], fit_df["a+"], **vline_kwargs)
    axs[1].set_ylabel(r"$\hat{a}$")
    axs[1].set_ylim([0.62, 1.66])

    # a hist
    ax_inset = axs[1].inset_axes([1, 0, 0.06, 1])
    ax_inset.hist(fit_df["a"], **hist_kwargs)
    ax_inset.set_ylim(axs[1].get_ylim())
    ax_inset.axis("off")

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
