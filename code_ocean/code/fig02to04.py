#!/usr/bin/env python

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter
from sklearn.base import clone
from tqdm import tqdm

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf

with open("./code_ocean/code/config.json") as f:
    rc_params = json.load(f)["rc-params"]
    plt.rcParams.update(rc_params)


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


def zero_formatter():
    """format tick labels"""

    return FuncFormatter(
        lambda v, _: ("0" if np.isclose(v, 0) else (f"{v:.2f}" if f"{v:.3f}".endswith("0") else f"{v:.3f}"))
    )


def plot_parameter_distributions(params_df, true_params):
    """plot pairwise parameter distributions"""

    # clip +/- 3 sd
    params_df = params_df.apply(lambda x: x.where((x >= x.mean() - 3 * x.std()) & (x <= x.mean() + 3 * x.std())))

    def _diag_plot(x, **kwargs):
        kd_kwargs = dict(color="k", fill=False, bw_adjust=1.2, linewidth=2.5)
        ax = plt.gca()
        sns.kdeplot(x, ax=ax, **kd_kwargs)
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
        ax.xaxis.set_major_formatter(zero_formatter())
        ax.yaxis.set_major_formatter(zero_formatter())
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([])
    g.tight_layout(w_pad=0.5, h_pad=1.2)
    return g


def plot_covariance_distributions(covs_df, params_df):
    """plot pairwise covariance distributions"""

    def _kde_plot(x, **kwargs):
        kd_kwargs = dict(fill=False, bw_adjust=1.2, linewidth=2.5, common_norm=True)
        sns.kdeplot(x, clip=(x.mean() - 3 * x.std(), x.mean() + 3 * x.std()), **kd_kwargs, **kwargs)

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
            ax.xaxis.set_major_formatter(zero_formatter())

    g.tight_layout(w_pad=0.1, h_pad=0)
    return g


def main(setting, n_samples, n_repeats, n_jobs, prefer):
    "fig02,03,04"

    kwargs = dict(n_samples=n_samples, n_repeats=n_repeats, random_state=0)
    figure_paths = {
        "constant": ("./code_ocean/results/fig02a.pdf", "./code_ocean/results/fig02b.pdf"),
        "coherence": ("./code_ocean/results/fig03a.pdf", "./code_ocean/results/fig03b.pdf"),
        "iid": ("./code_ocean/results/fig04a.pdf", "./code_ocean/results/fig04b.pdf"),
    }
    path_a, path_b = figure_paths[setting]

    # DDM parameters, from Matzke 2009, Table 3 (Mean)
    params = dict(a=0.63, t0=0.435, v=2.23, z=0.008)
    params_s = dict(t0=0.018, v=1.33, z=0.37)

    def setup_constant():
        """setup constant DDM parameters"""

        param_names = ["a", "t0", "v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        pseudotrue_params = params

        X = pd.DataFrame({"intercept": np.ones(kwargs["n_samples"])})
        ys = sample_from_pdf(**params, **kwargs)
        ddm = DriftDiffusionModel(cov_estimator="all")
        return X, ys, ddm, param_names, cov_names, pseudotrue_params

    def setup_coherence():
        """setup v as linear function of coherence"""

        param_names = ["a", "t0", "beta_v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        beta_v = 1
        coh = np.linspace(
            params["v"] + params_s["v"] * np.sqrt(3), params["v"] - params_s["v"] * np.sqrt(3), kwargs["n_samples"]
        )
        v_coh = beta_v * coh
        pseudotrue_params = {"a": params["a"], "t0": params["t0"], "beta_v": beta_v, "z": params["z"]}

        X = pd.DataFrame({"intercept": np.ones(kwargs["n_samples"]), "coherence": coh})
        ys = sample_from_pdf(a=params["a"], t0=params["t0"], v=v_coh, z=params["z"], **kwargs)
        ddm = DriftDiffusionModel(v="-1+coherence", cov_estimator="all", p_outlier=1e-12)
        return X, ys, ddm, param_names, cov_names, pseudotrue_params

    def setup_iid():
        """setup t0, v, z to vary iid"""

        param_names = ["a", "t0", "v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        n = 10_000
        X_pt = pd.DataFrame({"intercept": np.ones(n)})
        t0_iid, v_iid, z_iid = iid_params(n, params, params_s)
        y_pt = sample_from_pdf(
            a=params["a"], t0=t0_iid, v=v_iid, z=z_iid, n_samples=n, n_repeats=1, random_state=kwargs["random_state"]
        )
        ddm = DriftDiffusionModel()
        ddm.fit(X_pt, y_pt)
        pseudotrue_params = dict(zip(param_names, ddm.params_))

        X = pd.DataFrame({"intercept": np.ones(kwargs["n_samples"])})
        t0_iid, v_iid, z_iid = iid_params(kwargs["n_samples"], params, params_s)
        ys = sample_from_pdf(a=params["a"], t0=t0_iid, v=v_iid, z=z_iid, **kwargs)
        ddm = DriftDiffusionModel(cov_estimator="all")
        return X, ys, ddm, param_names, cov_names, pseudotrue_params

    if setting == "constant":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_constant()
    elif setting == "coherence":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_coherence()
    elif setting == "iid":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_iid()
    else:
        raise ValueError("choose 'constant', 'coherence', or 'iid'")

    @delayed
    def run_simulation(rep):
        ddm_cp = clone(ddm)
        ddm_cp.fit(X, ys[:, rep])
        covs_ = [
            {"estimator": k, **{cov_names[i]: val for i, val in enumerate(cov_to_corr(v).flatten())}}
            for k, v in ddm_cp.covariance_.items()
        ]
        return ddm_cp.params_, covs_

    with Parallel(n_jobs=n_jobs, prefer=prefer) as parallel:
        results = parallel(run_simulation(rep) for rep in tqdm(range(kwargs["n_repeats"])))
        params_, covs_ = zip(*results)
        params_df = pd.DataFrame(params_, columns=param_names)
        covs_df = pd.DataFrame([row for c in covs_ for row in c])

    fig = plot_parameter_distributions(params_df, pseudotrue_params)
    fig.savefig(path_a)

    fig = plot_covariance_distributions(covs_df, params_df)
    fig.savefig(path_b)


if __name__ == "__main__":
    """set script defaults"""

    parser = argparse.ArgumentParser(description="Figures 02-04")
    parser.add_argument(
        "--setting", type=str, default="constant", help="simulation setting: constant, coherence, or iid"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="number of trials to simulate per repeat")
    parser.add_argument("--n-repeats", type=int, default=900, help="number of simulation repeats")
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs")
    parser.add_argument(
        "--prefer", choices=["processes", "threads"], default="processes", help="joblib parallel backend"
    )
    args = parser.parse_args()

    main(args.setting, args.n_samples, args.n_repeats, args.n_jobs, args.prefer)
