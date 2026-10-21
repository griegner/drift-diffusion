#!/usr/bin/env python

"""Figures 02-04: Simulation Results"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter
from scipy.integrate import cumulative_trapezoid
from sklearn.base import clone
from tqdm import tqdm

from drift_diffusion.model import DriftDiffusionModel, pdf
from drift_diffusion.sim import sample_from_pdf

with open("./config.json") as f:
    rc_params = json.load(f)["rc-params"]
    rc_params["figure.constrained_layout.use"] = False
    plt.rcParams.update(rc_params)


def persistent_parameters():
    """Matzke Table 3 scales: near-neutral engagement and persistent right bias."""

    # Convert noise SD 0.1 and boundaries [0, A] to noise SD 1 and bounds +/-a.
    params = dict(a=0.125 / (2 * 0.1), t0=0.435, v=0.223 / 0.1, z=2 * 0.063 / 0.125 - 1)
    # eta is an SD; st and sz are full uniform widths (DMAT convention).
    sd_t0, sd_v, sd_z = 0.183 / np.sqrt(12), 0.133 / 0.1, (2 * 0.037 / 0.125) / np.sqrt(12)
    engaged = {**params, "t0": params["t0"] - sd_t0, "v": params["v"] + sd_v}
    disengaged = {
        **params,
        "t0": params["t0"] + sd_t0,
        "v": params["v"] - sd_v,
        "z": params["z"] + 2 * sd_z,
    }
    # Equal state weights preserve all three source variances. The z mean shifts
    # rightward because the engaged state, rather than the pooled mean, is near zero.
    return engaged, disengaged


def persistent_cdfs():
    """conditional signed-RT CDFs indexed by stimulus direction and hidden state."""

    states = persistent_parameters()
    rt = np.linspace(min(p["t0"] for p in states) + 1e-6, max(p["t0"] for p in states) + 12, 16001)
    y = np.r_[-rt, rt]
    cdfs = []
    for direction in [-1, 1]:
        state_cdfs = []
        for state in states:
            p = {**state, "v": direction * state["v"]}
            density = np.zeros_like(y)
            supported = abs(y) > p["t0"] + 1e-4
            density[supported] = pdf(y[supported], **p)
            integrals = cumulative_trapezoid(density.reshape(2, -1), rt, axis=1, initial=0)
            masses = integrals[:, -1]
            if not np.isclose(masses.sum(), 1, atol=2e-7, rtol=0):
                raise ValueError("Conditional RT grid has insufficient probability mass")
            state_cdfs.append(np.r_[masses[0] - integrals[0, ::-1], masses[0] + integrals[1]] / masses.sum())
        cdfs.append(state_cdfs)
    return np.r_[-rt[::-1], rt], np.array(cdfs)


def persistent_states(n_samples, rng):
    """stationary shared binary state with P(stay)=0.95 and mean runs of 20 trials."""

    state = np.empty(n_samples, dtype=int)
    state[0] = rng.integers(2)
    state[1:] = (state[0] + np.cumsum(rng.random(n_samples - 1) < 1 - 0.95)) % 2
    return state


def sample_persistent(n_samples, rep, signed_rt, cdfs, seed=0):
    """fresh hidden states, iid equiprobable stimuli, and conditional outcome noise."""

    rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
    state = persistent_states(n_samples, rng)
    direction = rng.integers(2, size=n_samples)
    stim = 2 * direction - 1
    u = rng.random(n_samples)
    y = np.empty(n_samples)
    for d in range(2):
        for j in range(2):
            mask = (direction == d) & (state == j)
            y[mask] = np.interp(u[mask], cdfs[d, j], signed_rt)
    return stim, y, state


def cov_to_corr(cov):
    "ii to standard errors, ij to correlations"

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

    label_map = {"a": r"$a$", "beta_a": r"$\beta_a$", "t0": r"$t_0$", "v": r"$v$", "abs_v": r"$|v|$", "z": r"$z$"}
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
    label_map = {"a": r"$a$", "beta_a": r"$\beta_a$", "t0": r"$t_0$", "v": r"$v$", "abs_v": r"$|v|$", "z": r"$z$"}
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


def main(setting, n_samples, n_repeats, n_jobs, prefer, output_dir="../results", seed=None):
    "fig02,03,04"

    if seed is None:
        seed = 720261019 if setting == "persistent" else 0
    kwargs = dict(n_samples=n_samples, n_repeats=n_repeats, random_state=seed)
    figure_paths = {
        "constant": ("fig02a", "fig02b"),
        "time": ("fig03a", "fig03b"),
        "persistent": ("fig04a", "fig04b"),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path_a, path_b = (Path(output_dir) / name for name in figure_paths[setting])

    # DDM parameters, from Matzke 2009, Table 3 (Mean)
    params = dict(a=0.63, t0=0.435, v=2.23, z=0.008)

    def setup_constant():
        """setup constant DDM parameters"""

        param_names = ["a", "t0", "v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        pseudotrue_params = params

        X = pd.DataFrame({"intercept": np.ones(kwargs["n_samples"])})
        ys = sample_from_pdf(**params, **kwargs)
        ddm = DriftDiffusionModel(cov_estimator="all")
        return X, ys, ddm, param_names, cov_names, pseudotrue_params

    def setup_time():
        """setup a as linear function of normalized time in session"""

        param_names = ["beta_a", "t0", "v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        beta_a = params["a"]
        time = np.linspace(0.8, 1.2, kwargs["n_samples"])
        a_time = beta_a * time
        pseudotrue_params = {"beta_a": beta_a, "t0": params["t0"], "v": params["v"], "z": params["z"]}

        X = pd.DataFrame({"intercept": np.ones(kwargs["n_samples"]), "time": time})
        ys = sample_from_pdf(a=a_time, t0=params["t0"], v=params["v"], z=params["z"], **kwargs)
        ddm = DriftDiffusionModel(a="-1+time", cov_estimator="all", p_outlier=1e-12)
        return X, ys, ddm, param_names, cov_names, pseudotrue_params

    def setup_persistent():
        """unmodeled engagement state; fit a, t0, drift magnitude, and side bias."""

        param_names = ["a", "t0", "abs_v", "z"]
        cov_names = [f"{i},{j}" for i in param_names for j in param_names]

        # Independently integrated expected working likelihood, averaging over
        # both hidden states and both observed stimulus directions.
        pseudotrue_params = dict(a=0.613275658969, t0=0.381493834668, abs_v=1.478601882646, z=0.051482449097)
        ddm = DriftDiffusionModel(v="-1+stim", cov_estimator="all", autocorrelation_lags=60)
        return None, None, ddm, param_names, cov_names, pseudotrue_params

    if setting == "constant":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_constant()
    elif setting == "time":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_time()
    elif setting == "persistent":
        X, ys, ddm, param_names, cov_names, pseudotrue_params = setup_persistent()
    else:
        raise ValueError("choose 'constant', 'time', or 'persistent'")

    sampling = persistent_cdfs() if setting == "persistent" else None

    @delayed
    def run_simulation(rep):
        ddm_cp = clone(ddm)
        if setting == "persistent":
            stim, y, state = sample_persistent(n_samples, rep, *sampling, seed=seed)
            # The hidden state is omitted; only observed stimulus direction enters.
            # Initialize at the population optimum while estimating all four coefficients.
            ddm_cp.fit(pd.DataFrame({"stim": stim}), y, params0=np.array(list(pseudotrue_params.values())))
        else:
            ddm_cp.fit(X, ys[:, rep])
        covs_ = [
            {"estimator": k, **{cov_names[i]: val for i, val in enumerate(cov_to_corr(v).flatten())}}
            for k, v in ddm_cp.covariance_.items()
        ]
        # Retain untrimmed numerical results and optimizer diagnostics for validation.
        audit = dict(
            cov=np.array(list(ddm_cp.covariance_.values())),
            success=bool(ddm_cp.fit_.success),
            message=str(ddm_cp.fit_.message),
            gradient_max=float(np.max(abs(ddm_cp.fit_.jac))),
            score_ac1=np.array([np.corrcoef(s[:-1], s[1:])[0, 1] for s in ddm_cp.score_.T]),
        )
        if setting == "persistent":
            audit["state_trials"] = np.bincount(state, minlength=2)
            audit["state_correct"] = np.bincount(state, weights=(np.sign(y) == stim), minlength=2)
        return ddm_cp.params_, covs_, audit

    with Parallel(n_jobs=n_jobs, prefer=prefer) as parallel:
        results = parallel(run_simulation(rep) for rep in tqdm(range(kwargs["n_repeats"])))
        params_, covs_, audits = zip(*results)
        params_df = pd.DataFrame(params_, columns=param_names)
        covs_df = pd.DataFrame([row for c in covs_ for row in c])

    if setting == "persistent":
        accuracy = np.sum([audit["state_correct"] for audit in audits], axis=0) / np.sum(
            [audit["state_trials"] for audit in audits], axis=0
        )
        print(f"Average accuracy: engaged {accuracy[0]:.2%}; disengaged {accuracy[1]:.2%}")

    fig = plot_parameter_distributions(params_df, pseudotrue_params)
    fig.savefig(path_a)

    fig = plot_covariance_distributions(covs_df, params_df)
    fig.savefig(path_b)

    result = dict(
        params=np.array(params_),
        target=np.array(list(pseudotrue_params.values())),
        **{key: np.array([audit[key] for audit in audits]) for key in audits[0]},
    )
    return result


if __name__ == "__main__":
    """set script defaults"""

    parser = argparse.ArgumentParser(description="Figures 02-04: Simulation Results")
    parser.add_argument(
        "--setting", type=str, choices=["constant", "time", "persistent"], default="constant", help="simulation setting"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="number of trials to simulate per repeat")
    parser.add_argument("--n-repeats", type=int, default=900, help="number of simulation repeats")
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs")
    parser.add_argument("--output-dir", default="../results", help="directory for figure panels")
    parser.add_argument("--seed", type=int, help="random seed (default: 720261019 for persistent, 0 otherwise)")
    parser.add_argument(
        "--prefer", choices=["processes", "threads"], default="processes", help="joblib parallel backend"
    )
    args = parser.parse_args()

    main(args.setting, args.n_samples, args.n_repeats, args.n_jobs, args.prefer, args.output_dir, args.seed)
