"""utility functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from statsmodels.tsa.stattools import acf, pacf

from drift_diffusion.model import DriftDiffusionModel

plt.rcParams.update({"font.size": 9})


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


def plot_heatmap_and_fits(df_heatmap, fit_df, x, y):
    """plot heatmap of coherences and parameter estimates"""

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
    a_lim, beta_v_lim = [0.62, 1.66], [-1.34, 3.48]
    axs["b"].set_xlim(a_lim)
    axs["b"].set_ylim(beta_v_lim)
    axs["c"].set_ylim(a_lim)
    axs["f"].set_ylim(beta_v_lim)

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
    corr_ = fit_df["a"].corr(fit_df["beta_v"])
    label = rf"$\text{{corr}}(\hat{{a}},\hat{{\beta}}_v)={corr_:.2f}$"
    # axs["b"].set_title(rf"$\text{{corr}}(\hat{{a}},\hat{{\beta}}_v)={corr_:.2f}$", loc="right", **text_kwargs)
    axs["b"].scatter(fit_df["a"], fit_df["beta_v"], s=1, color="k")
    axs["b"].set_title(r"$\hat{a}$", fontsize=9)  # xlabel
    axs["b"].set_ylabel(r"$\hat{\beta}_v$")
    axs["b"].text(0.98, 0.02, label, ha="right", va="bottom", transform=axs["b"].transAxes, **text_kwargs)

    # (c) a by day
    axs["c"].vlines(fit_df.index, fit_df["a-"], fit_df["a+"], **vline_kwargs)
    axs["c"].set_ylabel(r"$\hat{a}$")

    # (d) a hist
    label = rf"$\bar{{a}}$={fit_df["a"].mean():.2f}" "\n  " rf"$\pm${fit_df["a"].std():.2f}"
    axs["d"].hist(fit_df["a"], **hist_kwargs)
    axs["d"].text(0.1, 0.95, label, va="top", transform=axs["d"].transAxes, **text_kwargs)
    axs["d"].axis("off")

    # (e) a acf
    n_lags = len(fit_df) // 3
    acf_ = acf(fit_df["a"], nlags=n_lags, fft=True, bartlett_confint=False, missing="conservative")
    pacf_ = pacf(fit_df["a"].dropna(), nlags=n_lags)
    axs["e"].plot(acf_, c="k", lw=1, label="acf")
    axs["e"].plot(pacf_, c="gray", lw=1, label="pacf")
    axs["e"].legend()
    axs["e"].set_ylabel(r"$\text{corr}(\hat{a})$")

    # (f) beta_v by day
    axs["f"].vlines(fit_df.index, fit_df["beta_v-"], fit_df["beta_v+"], **vline_kwargs)
    axs["f"].set_ylabel(r"$\hat{\beta}_v$")
    axs["f"].set_xlabel(x)
    if x == "hour":
        axs["f"].set_xticks([1, 6, 12, 18, 24])

    # (g) beta_v hist
    label = rf"$\bar{{\beta}}_v$={fit_df["beta_v"].mean():.2f}" "\n   " rf"$\pm${fit_df["beta_v"].std():.2f}"
    axs["g"].hist(fit_df["beta_v"], **hist_kwargs)
    axs["g"].text(0.1, 0.05, label, va="bottom", transform=axs["g"].transAxes, **text_kwargs)
    axs["g"].axis("off")

    # (h) beta_v acf
    acf_ = acf(fit_df["beta_v"], nlags=n_lags, fft=True, bartlett_confint=False, missing="conservative")
    pacf_ = pacf(fit_df["beta_v"].dropna(), nlags=n_lags)
    axs["h"].plot(acf_, c="k", lw=1)
    axs["h"].plot(pacf_, c="gray", lw=1)
    axs["h"].set_ylabel(r"$\text{corr}(\hat{\beta}_v)$")
    axs["h"].set_xlabel("lag (days)")
