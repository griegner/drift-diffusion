#!/usr/bin/env python

"""figure 08: MLE with session trends, pupil effects, and hidden engagement states.

Compare outer-product and autocorrelation-robust uncertainties using rRMSE.
Use --csv to replot saved results.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["outer-product", "autocorrelation-robust"]
COLORS = ["#a88140", "#cf3e40"]
PARAM_NAMES = ["a_intercept", "a_time", "t0", "v_stim", "v_stim:pupil", "z"]
PARAM_LABELS = [r"$a$: intercept", r"$a$: session slope", r"$t_0$", r"$v$: stimulus", r"$v$: pupil", r"$z$"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

with (ROOT / "code/config.json").open() as f:
    plt.rcParams.update(json.load(f)["rc-params"])
with (ROOT / "data/fig08-targets.json").open() as f:
    TARGETS = json.load(f)


def simulation_design():
    """fixed pupil traces; keep the original five-session calibration."""
    rng = np.random.default_rng(np.random.SeedSequence([TARGETS["seed"], 17]))
    pupil = gaussian_filter1d(rng.normal(size=(10, 1000)), 20, axis=1, mode="wrap")
    pupil = (0.5 + 0.15 * pupil / pupil.std(axis=1, keepdims=True)).ravel()
    centered = pupil - pupil[:5000].mean()
    beta_p = (2.23 + 1.33 * np.sqrt(3) - 0.1) / np.ptp(pupil[:5000])
    beta_v = 0.1 - beta_p * centered[:5000].min()
    session_time = np.tile(np.linspace(-0.5, 0.5, 1000), 10)
    z = 2 * 0.063 / 0.125 - 1 + np.array([0, 2]) * (2 * 0.037 / 0.125) / np.sqrt(12)
    return dict(
        time=session_time,
        pupil_centered=centered,
        a=0.63 + 0.252 * session_time,
        t0=0.435,
        magnitude=beta_v + beta_p * centered,
        z=z,
    )


def simulate(design, n, rep):
    """sample RTs with persistent states and independent stimulus directions."""
    rng = np.random.default_rng(np.random.SeedSequence([TARGETS["seed"], 2, n, rep]))
    state = np.empty(n, dtype=int)
    state[0] = rng.integers(2)
    state[1:] = (state[0] + np.cumsum(rng.random(n - 1) < 0.05)) % 2
    stim = 2 * rng.integers(2, size=n) - 1
    y = sample_from_pdf(
        a=design["a"][:n],
        t0=design["t0"],
        v=stim * design["magnitude"][:n],
        z=design["z"][state],
        n_samples=n,
        random_state=int(rng.integers(2**32)),
    )
    X = pd.DataFrame(dict(time=design["time"][:n], pupil_centered=design["pupil_centered"][:n], stim=stim))
    return X, y


def fit_mle(X, y):
    """fit the package model and time optimization plus covariance estimation."""
    mle = DriftDiffusionModel(
        a="1 + time",
        t0="1",
        v="0 + stim + stim:pupil_centered",
        z="1",
        p_outlier=0,
        cov_estimator="all",
        autocorrelation_lags=60,
    )
    started = time.perf_counter()
    try:
        for start in [[0.63, 0.20, 0.35, 2.2, 4.0, 0.15], [0.7, 0.10, 0.30, 1.5, 2.0, 0.0]]:
            mle.fit(X, y, params0=np.array(start))
            if np.isfinite(mle.fit_.fun) and np.max(abs(mle.fit_.jac)) < 0.01:
                break
        else:
            raise ValueError("MLE did not converge")
        runtime = time.perf_counter() - started
        cov = np.array([mle.covariance_[method] for method in METHODS])
        valid = np.isfinite(mle.params_).all() and np.isfinite(cov).all()
        valid = valid and all(np.linalg.eigvalsh((c + c.T) / 2).min() > 0 for c in cov)
        return runtime, mle.params_, cov, bool(valid)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return time.perf_counter() - started, np.full(6, np.nan), np.full((2, 6, 6), np.nan), False


def bias_sd_rmse(params, params_):
    """params (true values), params_ (matrix of estimates)"""

    bias_ = np.mean(params_ - params, axis=0)
    sd_ = params_.std(axis=0, ddof=1)
    rmse_ = np.sqrt(np.mean((params_ - params) ** 2, axis=0))
    return bias_, sd_, rmse_


def summarize_method(n, method, runtime, params, params_, uncs_):
    """params (reference values), params_ and uncs_ (estimates across repeats)"""

    runtime_mean = float(np.mean(runtime))
    references = {"param": params, "unc": params_.std(axis=0, ddof=1)}
    metrics = {
        "param": bias_sd_rmse(references["param"], params_),
        "unc": bias_sd_rmse(references["unc"], uncs_),
    }
    return [
        {
            "n": n,
            "method": method,
            "estimate": estimate,
            "param": name,
            "runtime": runtime_mean,
            "bias": bias,
            "sd": sd,
            "rmse": rmse,
            "reference": reference,
            "rrmse": 100 * rmse / abs(reference),
            "common_repeats": len(params_),
        }
        for estimate, (bias_, sd_, rmse_) in metrics.items()
        for name, bias, sd, rmse, reference in zip(PARAM_NAMES, bias_, sd_, rmse_, references[estimate])
    ]


def summarize_fits(fits):
    """summarize both estimators on the same valid repetitions"""

    rows = []
    for n in np.unique(fits["n"]):
        selected = (fits["n"] == n) & fits["valid"]
        params_ = fits["params"][selected]
        if len(params_) < 3:
            raise ValueError(f"Need at least three valid repetitions at n={n}")
        params = np.array(TARGETS["targets"][str(n)])
        runtime = fits["runtime"][selected]
        for m, method in enumerate(METHODS):
            uncs_ = np.sqrt(np.diagonal(fits["cov"][selected, m], axis1=1, axis2=2))
            rows.extend(summarize_method(n, method, runtime, params, params_, uncs_))
        print(f"n={n}: {len(params_)}/{np.sum(fits['n'] == n)} valid fits")
    return pd.DataFrame.from_records(rows)


def plot_benchmark(df):
    """paired rRMSE; reference lines are descriptive, not acceptance cutoffs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    sizes = sorted(df.n.unique())
    x = np.arange(len(sizes))
    offsets = np.linspace(-0.28, 0.28, len(PARAM_NAMES))
    runtime = df.groupby("n").runtime.first().loc[sizes]
    axes[0].plot(x, runtime, "o-", color="0.72", markerfacecolor="0.3", markeredgecolor="0.3", lw=1.25)
    for name, marker, offset in zip(PARAM_NAMES, MARKERS, offsets):
        group = df[df.param == name]
        xx = x + offset
        point = group[(group.method == METHODS[0]) & (group.estimate == "param")].set_index("n").loc[sizes]
        axes[1].plot(
            xx, point.rrmse, marker=marker, color="0.72", markerfacecolor="0.3", markeredgecolor="0.3", lw=1.05
        )
        paired = []
        for method, color in zip(METHODS, COLORS):
            unc = group[(group.method == method) & (group.estimate == "unc")].set_index("n").loc[sizes]
            paired.append(unc.rrmse.to_numpy())
            axes[2].scatter(xx, unc.rrmse, color=color, marker=marker, s=46, edgecolor="white", lw=0.35, zorder=3)
        axes[2].vlines(xx, np.minimum(*paired), np.maximum(*paired), color="0.72", lw=1.1)
    for ax, reference in zip(axes[1:], [10, 20]):
        ax.axhline(reference, color="0.35", ls="--", lw=1, zorder=0)
        ax.text(
            0.98,
            reference,
            f"{reference}% reference",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )
    for ax, panel, label in zip(axes, "ABC", ["Seconds per fit", "Point rRMSE (%)", "SE rRMSE (%)"]):
        ax.set(
            xlabel="n Trials",
            ylabel=label,
            xticks=x,
            xticklabels=[f"{n:,}" for n in sizes],
            ylim=(0, None),
            xlim=(-0.45, len(sizes) - 0.55),
        )
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    handles = [
        Line2D([], [], color=color, marker="o", ls="none", label=label)
        for color, label in zip(COLORS, ["Outer product", "Autocorrelation robust (60 lags)"])
    ]
    handles += [
        Line2D([], [], color="0.3", marker=marker, ls="none", label=label)
        for marker, label in zip(MARKERS, PARAM_LABELS)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    fig.savefig(ROOT / "results/fig08.svg")
    plt.close(fig)


def main(n_samples, n_repeats, n_jobs, csv):
    """run the fixed simulation or replot the prepared benchmark data."""
    if csv:
        df = pd.read_csv(ROOT / "data/fig08.csv").query("n in @n_samples")
    else:
        design = simulation_design()

        @delayed
        def run_simulation(rep, n):
            X, y = simulate(design, n, rep)
            return n, rep, *fit_mle(X, y)

        records = []
        for n in n_samples:
            with Parallel(n_jobs=n_jobs) as parallel:
                records.extend(parallel(run_simulation(rep, n) for rep in range(n_repeats)))
        fits = dict(zip(["n", "rep", "runtime", "params", "cov", "valid"], map(np.array, zip(*records))))
        df = summarize_fits(fits)
        np.savez_compressed(ROOT / "data/fig08-fits.npz", methods=METHODS, param_names=PARAM_NAMES, **fits)
        df.to_csv(ROOT / "data/fig08.csv", index=False)
    plot_benchmark(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        choices=[500, 1000, 5000, 10000],
        default=[1000, 5000, 10000],
        help="number of trials per repeat (500 is half a session)",
    )
    parser.add_argument("--n-repeats", type=int, default=500, help="number of independent simulation repeats")
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs")
    parser.add_argument("--csv", action="store_true", help="replot saved data/fig08.csv")
    args = parser.parse_args()
    if args.n_repeats < 3 or args.n_jobs == 0 or len(set(args.n_samples)) != len(args.n_samples):
        parser.error("need at least three repeats, nonzero n-jobs and unique sample sizes")
    main(args.n_samples, args.n_repeats, args.n_jobs, args.csv)
