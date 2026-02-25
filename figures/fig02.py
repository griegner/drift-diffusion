import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def iid_params(n_samples, params, params_s, seed=1):
    """generate DDM parameters with iid variability (i.e. 7-param DDM)"""
    rng = np.random.default_rng(seed=1)
    t0_iid = rng.uniform(params["t0"] - params_s["t0"] / 2, params["t0"] + params_s["t0"] / 2, size=n_samples)
    v_iid = rng.normal(params["v"], params_s["v"], size=n_samples)
    z_iid = rng.uniform(params["z"] - params_s["z"] / 2, params["z"] + params_s["z"] / 2, size=n_samples)
    return t0_iid, v_iid, z_iid


def cov_to_corr(cov):
    "ii to standard errors, ij to correlations"
    stderr = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stderr, stderr)
    np.fill_diagonal(corr, stderr)
    return corr


def _zero_formatter():
    """format tick labels"""
    return FuncFormatter(
        lambda v, _: ("0" if np.isclose(v, 0) else (f"{v:.2f}" if f"{v:.3f}".endswith("0") else f"{v:.3f}"))
    )


def plot_parameter_distributions(params_df, true_params):
    """plot pairwise parameter distributions"""

    def _diag_plot(x, **kwargs):
        kd_kwargs = dict(color="k", fill=False, bw_adjust=1.2, linewidth=2.5)
        ax = plt.gca()
        sns.kdeplot(x, ax=ax, clip=(x.mean() - 3 * x.std(), x.mean() + 3 * x.std()), **kd_kwargs)
        ax.axvline(true_params[x.name], c="b", lw=2)
        ax.plot([x.mean() - x.std(), x.mean() + x.std()], [np.mean(ax.get_ylim())] * 2, c="r", lw=2)

    def _lower_plot(x, y, **kwargs):
        sns.kdeplot(x=x, y=y, color="k", bw_adjust=1.2, linewidths=0.8, levels=6)
        sns.regplot(x=x, y=y, scatter=False, ci=None, line_kws={"color": "red", "lw": 1.5, "ls": "--"})

    g = sns.PairGrid(params_df, height=1.8, diag_sharey=False, despine=False)
    g.map_upper(lambda *args, **kwargs: plt.gca().axis("off"))
    g.map_lower(_lower_plot)
    g.map_diag(_diag_plot)

    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(_zero_formatter())
        ax.yaxis.set_major_formatter(_zero_formatter())
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([])
    g.tight_layout(w_pad=0.5, h_pad=1.2)
    return g


def plot_covariance_distributions(covs_df, params_df):
    """plot pairwise covariance distributions"""

    def _kde_plot(data, **kwargs):
        kd_kwargs = dict(fill=False, bw_adjust=1.2, linewidth=2.5, common_norm=True)
        sns.kdeplot(data, clip=(data.mean() - 3 * data.std(), data.mean() + 3 * data.std()), **kd_kwargs, **kwargs)

    fg_kwargs = dict(sharex=False, sharey=False, height=1.8, palette="binary", despine=False)
    g = sns.FacetGrid(covs_df.melt(id_vars="estimator"), hue="estimator", col="variable", col_wrap=4, **fg_kwargs)
    g.map(_kde_plot, "value")
    g.set_titles("")
    g.set_xlabels("")
    g.set_ylabels("")

    n_cols = g._ncol
    correlations = cov_to_corr(np.cov(params_df.T)).flatten()
    for idx, (ax, correlation) in enumerate(zip(g.axes.flat, correlations)):
        row, col = divmod(idx, n_cols)
        if row > col:
            ax.set_visible(False)
        else:
            ax.set_yticklabels([])
            ax.yaxis.tick_right()
            linestyle = "-" if row == col else "--"
            ax.axvline(x=correlation, c="r", lw=2, ls=linestyle)
            ax.xaxis.set_major_formatter(_zero_formatter())

    g.tight_layout(w_pad=0.1, h_pad=0)
    return g
