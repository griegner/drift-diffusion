#!/usr/bin/env python

"""figures 02-04: simulation results"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter
from pseudotrue import pmf, pseudotrue, sampler_grid
from sklearn.base import clone
from tqdm import tqdm

from drift_diffusion.model import DriftDiffusionModel, pdf
from drift_diffusion.sim import sample_from_pdf

with (Path(__file__).resolve().parent / "config.json").open() as f:
    rc_params = json.load(f)["rc-params"]
    rc_params["figure.constrained_layout.use"] = False
    plt.rcParams.update(rc_params)


def persistent_parameters():
    """return Matzke Table 3 scales with near-neutral engagement and persistent right bias"""

    # rescale Matzke means to noise SD 1 and bounds +/-a (a rounded to 0.63)
    params = dict(a=0.63, t0=0.435, v=2.23, z=0.008)
    # eta: SD; st, sz: full uniform widths (DMAT)
    sd_t0, sd_v, sd_z = 0.183 / np.sqrt(12), 0.133 / 0.1, (2 * 0.037 / 0.125) / np.sqrt(12)
    engaged = {**params, "t0": params["t0"] - sd_t0, "v": params["v"] + sd_v}
    disengaged = {
        **params,
        "t0": params["t0"] + sd_t0,
        "v": params["v"] - sd_v,
        "z": params["z"] + 2 * sd_z,
    }
    # equal weights preserve variances; near-neutral engaged z shifts the pooled mean rightward
    return engaged, disengaged


def persistent_states(n_samples, rng):
    """sample a stationary shared binary state with P(stay)=0.95 and mean runs of 20 trials"""

    state = np.empty(n_samples, dtype=int)
    state[0] = rng.integers(2)
    state[1:] = (state[0] + np.cumsum(rng.random(n_samples - 1) < 1 - 0.95)) % 2
    return state


def persistent_pseudotrue():
    """find continuous-DDM (a, t0, abs_v, z) maximizing the expected working likelihood"""

    # average both states and stimulus directions equally on the sampler's RT grid
    states = persistent_parameters()
    y = sampler_grid([state["t0"] for state in states])
    ys, weights, stims = [], [], []
    for state in states:
        for stim in (-1, 1):
            ys.append(y)
            weights.append(pmf(y, state["a"], state["t0"], stim * state["v"], state["z"]) / 4)
            stims.append(np.full_like(y, stim))
    y, weights, stims = map(np.concatenate, (ys, weights, stims))
    x0 = [np.mean([state[name] for state in states]) for name in ["a", "t0", "v", "z"]]
    theta = pseudotrue(y, weights, lambda p, y: pdf(y, p[0], p[1], p[2] * stims, p[3]), x0)
    return dict(zip(["a", "t0", "abs_v", "z"], theta))


def sample_persistent(n_samples, rep, seed=0):
    """sample RTs with persistent engagement states and independent stimuli"""

    rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
    state = persistent_states(n_samples, rng)
    stim = 2 * rng.integers(2, size=n_samples) - 1
    states = persistent_parameters()
    params = {name: np.array([p[name] for p in states])[state] for name in ["a", "t0", "v", "z"]}
    params["v"] *= stim
    y = sample_from_pdf(**params, n_samples=n_samples, random_state=int(rng.integers(2**32)))
    return stim, y


def cov_to_corr(cov):
    """convert diagonal covariance entries to standard errors and off-diagonal entries to correlations"""

    stderr = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stderr, stderr)
    np.fill_diagonal(corr, stderr)
    return corr


def zero_formatter():
    """format tick labels"""

    return FuncFormatter(
        lambda v, _: "0" if np.isclose(v, 0) else (f"{v:.2f}" if f"{v:.3f}".endswith("0") else f"{v:.3f}")
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

    label_map = {"a": r"$a$", "t0": r"$t_0$", "v": r"$v$", "beta_a": r"$\beta_a$", "z": r"$z$", "abs_v": r"$|v|$"}
    for i, col in enumerate(params_df.columns):
        label = label_map[col]
        g.axes[-1, i].set_xlabel(label, fontsize=22)
        g.axes[i, 0].set_ylabel(label, fontsize=22, rotation=0)

    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(zero_formatter())
        ax.yaxis.set_major_formatter(zero_formatter())
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
    label_map = {"a": r"$a$", "t0": r"$t_0$", "v": r"$v$", "beta_a": r"$\beta_a$", "z": r"$z$", "abs_v": r"$|v|$"}
    param_labels = [label_map[col] for col in params_df.columns]
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

            if row == 0:
                ax.set_xlabel(param_labels[col], fontsize=22)
                ax.xaxis.set_label_position("top")

            if col == n_cols - 1:
                ax.set_ylabel(param_labels[row], fontsize=22, rotation=0)
                ax.yaxis.set_label_position("right")

    g.tight_layout(w_pad=0.1, h_pad=0)
    return g


def main(setting, n_samples, n_repeats, n_jobs, prefer):
    """plot and save fig02,03,04"""

    kwargs = dict(n_samples=n_samples, n_repeats=n_repeats, random_state=720261019 if setting == "persistent" else 0)
    figure_paths = {
        "constant": ("../results/fig02a", "../results/fig02b"),
        "time": ("../results/fig03a", "../results/fig03b"),
        "persistent": ("../results/fig04a", "../results/fig04b"),
    }
    path_a, path_b = figure_paths[setting]

    # mean DDM parameters from Matzke 2009, Table 3
    params = dict(a=0.63, t0=0.435, v=2.23, z=0.008)

    def setup_constant():
        """set up constant DDM parameters"""

        param_names = ["a", "t0", "v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        pseudotrue_params = params

        X = pd.DataFrame({"intercept": np.ones(kwargs["n_samples"])})
        ys = sample_from_pdf(**params, **kwargs)
        ddm = DriftDiffusionModel(cov_estimator="all", p_outlier=0)
        return X, ys, ddm, param_names, cov_names, pseudotrue_params

    def setup_time():
        """set up a as a linear function of normalized time in session"""

        param_names = ["beta_a", "t0", "v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        beta_a = params["a"]
        time = np.linspace(0.8, 1.2, kwargs["n_samples"])
        a_time = beta_a * time
        pseudotrue_params = {"beta_a": beta_a, "t0": params["t0"], "v": params["v"], "z": params["z"]}

        X = pd.DataFrame({"intercept": np.ones(kwargs["n_samples"]), "time": time})
        ys = sample_from_pdf(a=a_time, t0=params["t0"], v=params["v"], z=params["z"], **kwargs)
        ddm = DriftDiffusionModel(a="-1+time", cov_estimator="all", p_outlier=0)
        return X, ys, ddm, param_names, cov_names, pseudotrue_params

    def setup_persistent():
        """set up an unmodeled engagement state; fit a, t0, drift magnitude, and side bias"""

        param_names = ["a", "t0", "abs_v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        pseudotrue_params = persistent_pseudotrue()
        print("pseudo-true:", {name: round(value, 6) for name, value in pseudotrue_params.items()})
        ddm = DriftDiffusionModel(v="-1+stim", cov_estimator="all", autocorrelation_lags=60, p_outlier=0)
        return None, None, ddm, param_names, cov_names, pseudotrue_params

    if setting == "constant":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_constant()
    elif setting == "time":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_time()
    elif setting == "persistent":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_persistent()
    else:
        raise ValueError("choose 'constant', 'time', or 'persistent'")

    @delayed
    def run_simulation(rep):
        ddm_cp = clone(ddm)
        if setting == "persistent":
            stim, y = sample_persistent(n_samples, rep, seed=kwargs["random_state"])
            # fit observed stimulus direction; omit the hidden state
            ddm_cp.fit(pd.DataFrame({"stim": stim}), y)
        else:
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
    # set script defaults

    parser = argparse.ArgumentParser(description="Figures 02-04: Simulation Results")
    parser.add_argument(
        "--setting", type=str, choices=["constant", "time", "persistent"], default="constant", help="simulation setting"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="number of trials to simulate per repeat")
    parser.add_argument("--n-repeats", type=int, default=900, help="number of simulation repeats")
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs")
    parser.add_argument(
        "--prefer", choices=["processes", "threads"], default="processes", help="joblib parallel backend"
    )
    args = parser.parse_args()

    main(args.setting, args.n_samples, args.n_repeats, args.n_jobs, args.prefer)
