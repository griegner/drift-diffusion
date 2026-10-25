#!/usr/bin/env python

"""figure 08: MLE with session trends, pupil effects, and hidden engagement states.

Compare the covariate MLE (sample-Hessian and autocorrelation-robust SEs) with the
seven-parameter HSSM full_ddm (MCMC) on the quantities both estimate, using rRMSE
against generating values. Use --csv to replot saved results.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddm7p import sample_full_ddm
from joblib import Parallel, delayed, effective_n_jobs
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.ndimage import gaussian_filter1d

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261020
CHECKPOINTS = ROOT / "data/fig08-mcmc"
METHODS = ["sample-hessian", "autocorrelation-robust"]
COLORS = ["#a88140", "#cf3e40"]
MCMC_COLOR = "#3b6fb6"
PARAM_NAMES = ["a_intercept", "a_time", "t0", "v_stim", "v_stim:pupil", "z"]
# hssm full_ddm names; z and sz live on [0, 1] in hssm and [-1, 1] here
MCMC_NAMES = ["a", "t", "v_stim", "z", "sz", "sv", "st"]
MCMC_SCALE, MCMC_OFFSET = np.array([1, 1, 1, 2, 2, 1, 1]), np.array([0, 0, 0, -1, 0, 0, 0])
MCMC_FIELDS = ["runtime", "params", "sd", "r_hat", "ess", "valid"]
# quantities both models estimate: the first four MCMC_NAMES, and linear combinations of MLE coefficients
QUANTITIES = ["a_mid", "t0", "v_avg", "z"]
QUANTITY_LABELS = [r"$a$: mid-session", r"$t_0$", r"$v$: at mean pupil", r"$z$"]
MARKERS = ["o", "^", "D", "P"]

with (ROOT / "code/config.json").open() as f:
    plt.rcParams.update(json.load(f)["rc-params"])


def simulation_design():
    """fixed pupil traces; keep the original five-session calibration."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 17]))
    pupil = gaussian_filter1d(rng.normal(size=(10, 1000)), 20, axis=1, mode="wrap")
    pupil = (0.5 + 0.15 * pupil / pupil.std(axis=1, keepdims=True)).ravel()
    centered = pupil - pupil[:5000].mean()
    beta_p = (2.23 + 1.33 * np.sqrt(3) - 0.1) / np.ptp(pupil[:5000])
    beta_v = 0.1 - beta_p * centered[:5000].min()
    session_time = np.tile(np.linspace(-0.5, 0.5, 1000), 10)
    z = 0.008 + np.array([0, 2]) * (2 * 0.037 / 0.125) / np.sqrt(12)
    return dict(
        time=session_time,
        pupil_centered=centered,
        a=0.63 + 0.252 * session_time,
        t0=0.435,
        magnitude=beta_v + beta_p * centered,
        z=z,
    )


def comparable(design, n):
    """contrasts from the MLE coefficients to QUANTITIES, and their generating values in the first n trials"""
    L = np.zeros((len(QUANTITIES), len(PARAM_NAMES)))
    L[0, [0, 1]] = 1, design["time"][:n].mean()  # a at the average session time (the midpoint)
    L[1, 2] = 1
    L[2, [3, 4]] = 1, design["pupil_centered"][:n].mean()  # drift at the average pupil
    L[3, 5] = 1
    truth = [design["a"][:n].mean(), design["t0"], design["magnitude"][:n].mean(), design["z"].mean()]
    return L, np.array(truth)


def simulate(design, n, rep):
    """sample RTs with persistent states and independent stimulus directions."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, n, rep]))
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
    """fit from the package's default start (a=1, all else 0); time optimization plus covariance estimation."""
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
        mle.fit(X, y)
        if not (np.isfinite(mle.fit_.fun) and np.max(abs(mle.fit_.jac)) < 0.01):
            raise ValueError("MLE did not converge")
        runtime = time.perf_counter() - started
        cov = np.array([mle.covariance_[method] for method in METHODS])
        valid = np.isfinite(mle.params_).all() and np.isfinite(cov).all()
        valid = valid and all(np.linalg.eigvalsh((c + c.T) / 2).min() > 0 for c in cov)
        return runtime, mle.params_, cov, bool(valid)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return time.perf_counter() - started, np.full(6, np.nan), np.full((2, 6, 6), np.nan), False


def fit_mcmc(X, y, params_mle, chains, tune, draws, seed):
    """seven-parameter full_ddm with stimulus-coded drift and no other covariates; chains start at the MLE."""
    started = time.perf_counter()
    mle = dict(zip(PARAM_NAMES, params_mle if np.isfinite(params_mle).all() else [0.63, 0, 0.44, 2.2, 0, 0.15]))
    model = sample_full_ddm(
        pd.DataFrame(dict(rt=np.abs(y), response=np.sign(y).astype(int), stim=X["stim"])),
        initvals=dict(
            a=mle["a_intercept"],
            t=mle["t0"] - 0.01,
            v_stim=mle["v_stim"],
            z=(mle["z"] + 1) / 2,
            sz=0.3,
            sv=0.8,
            st=0.02,
        ),
        chains=chains,
        tune=tune,
        draws=draws,
        seed=seed,
        include=[
            dict(
                name="v",
                formula="v ~ 0 + stim",
                link="identity",
                prior={"stim": {"name": "Normal", "mu": 0.0, "sigma": 3.0}},
            )
        ],
    )
    runtime = time.perf_counter() - started
    summary = az.summary(model.traces, var_names=MCMC_NAMES, round_to="none")
    r_hat, ess = summary["r_hat"].to_numpy(), summary["ess_bulk"].to_numpy()
    # st sits at its zero bound and mixes slowly; judge convergence on the other six
    valid = bool(np.all(r_hat[:-1] < 1.05) and np.all(ess[:-1] >= 100))
    mean = MCMC_SCALE * summary["mean"].to_numpy() + MCMC_OFFSET
    sd = MCMC_SCALE * summary["sd"].to_numpy()
    return runtime, mean, sd, r_hat, ess, valid


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
            "n_repeats": len(params_),
        }
        for estimate, (bias_, sd_, rmse_) in metrics.items()
        for name, bias, sd, rmse, reference in zip(QUANTITIES, bias_, sd_, rmse_, references[estimate])
    ]


def summarize(fits, mcmc, design):
    """shared quantities against generating values; with MCMC, both methods are scored on the same
    repetitions (valid MLE and converged MCMC), while MLE runtime averages all valid MLE fits."""

    rows = []
    for n in np.unique(fits["n"]):
        L, params = comparable(design, n)
        valid = (fits["n"] == n) & fits["valid"]
        print(f"n={n}: {valid.sum()}/{np.sum(fits['n'] == n)} valid MLE fits")
        selected = valid
        if mcmc:
            converged = (mcmc["n"] == n) & mcmc["valid"]
            print(f"n={n}: {converged.sum()}/{np.sum(mcmc['n'] == n)} converged MCMC fits")
            reps = np.intersect1d(fits["rep"][valid], mcmc["rep"][converged])
            print(f"n={n}: {len(reps)} repetitions shared by both methods")
            converged = converged & np.isin(mcmc["rep"], reps)
            if len(reps) >= 3:
                selected = valid & np.isin(fits["rep"], reps)
            else:
                print(f"n={n}: too few shared repetitions; MLE scored on all valid fits, MCMC omitted")
        estimates = [
            (
                method,
                fits["runtime"][valid],
                fits["params"][selected] @ L.T,
                np.sqrt(np.einsum("ij,rjk,ik->ri", L, fits["cov"][selected, m], L)),  # diag(L cov L')
            )
            for m, method in enumerate(METHODS)
        ]
        if mcmc:
            shared = slice(len(QUANTITIES))
            estimates.append(
                (
                    "mcmc",
                    mcmc["runtime"][converged],
                    *(mcmc[k][converged, shared] for k in ["params", "sd"]),
                )
            )
        for method, runtime, params_, uncs_ in estimates:
            if len(params_) >= 3:
                rows.extend(summarize_method(n, method, runtime, params, params_, uncs_))
    return pd.DataFrame.from_records(rows)


def plot_benchmark(df):
    """paired rRMSE; reference lines are descriptive, not acceptance cutoffs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    sizes = sorted(df.n.unique())
    x = np.arange(len(sizes))
    offsets = np.linspace(-0.24, 0.24, len(QUANTITIES))
    runtime = df.groupby(["method", "n"]).runtime.first()
    axes[0].plot(x, runtime[METHODS[0]].loc[sizes], "o-", color="0.72", markerfacecolor="0.3", markeredgecolor="0.3")
    axes[0].plot(x, runtime.get("mcmc", pd.Series()).reindex(sizes), "o-", color=MCMC_COLOR)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    for name, marker, offset in zip(QUANTITIES, MARKERS, offsets):
        group = df[df.param == name]
        xx = x + offset
        point = group[(group.method == METHODS[0]) & (group.estimate == "param")].set_index("n").loc[sizes]
        axes[1].plot(
            xx, point.rrmse, marker=marker, color="0.72", markerfacecolor="0.3", markeredgecolor="0.3", lw=1.05
        )
        paired = []
        for method, color in zip(METHODS + ["mcmc"], COLORS + [MCMC_COLOR]):
            unc = group[(group.method == method) & (group.estimate == "unc")].set_index("n").reindex(sizes)
            paired.append(unc.rrmse.to_numpy())
            axes[2].scatter(xx, unc.rrmse, color=color, marker=marker, s=46, edgecolor="white", lw=0.35, zorder=3)
        axes[2].vlines(xx, np.nanmin(paired, axis=0), np.nanmax(paired, axis=0), color="0.72", lw=1.1)
        point = group[(group.method == "mcmc") & (group.estimate == "param")].set_index("n").reindex(sizes)
        axes[1].scatter(xx, point.rrmse, color=MCMC_COLOR, marker=marker, s=46, edgecolor="white", lw=0.35)
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
        ax.set(xlabel="n Trials", ylabel=label, xticks=x, xticklabels=[f"{n:,}" for n in sizes])
        ax.set(xlim=(-0.45, len(sizes) - 0.55))
        if ax is not axes[0]:
            ax.set_ylim(0, None)
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    handles = [
        Line2D([], [], color=color, marker="o", ls="none", label=label)
        for color, label in zip(
            COLORS + [MCMC_COLOR], ["Sample Hessian", "Autocorrelation robust (60 lags)", "full_ddm MCMC"]
        )
    ]
    handles += [
        Line2D([], [], color="0.3", marker=marker, ls="none", label=label)
        for marker, label in zip(MARKERS, QUANTITY_LABELS)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    fig.savefig(ROOT / "results/fig08.svg")
    plt.close(fig)


def run_mcmc_checkpoint(n, rep, params_mle, chains, tune, draws):
    """one MCMC fit, saved to data/fig08-mcmc/ so an interrupted run resumes where it stopped."""
    path = CHECKPOINTS / f"n{n}-rep{rep}-c{chains}-t{tune}-d{draws}.npz"
    if not path.exists():
        X, y = simulate(simulation_design(), n, rep)
        seed = int(np.random.SeedSequence([SEED, 3, n, rep]).generate_state(1)[0])
        fit = fit_mcmc(X, y, params_mle, chains, tune, draws, seed)
        np.savez(path, **dict(zip(MCMC_FIELDS, fit)))
        print(f"MCMC n={n} rep={rep}: {fit[0]:.0f} s, converged={fit[-1]}", flush=True)
    with np.load(path) as saved:
        return n, rep, *(saved[name] for name in MCMC_FIELDS)


def run_mcmc(fits, n_samples, n_repeats, n_jobs, chains, tune, draws):
    """chains run in parallel within a fit; fits run in parallel across the remaining cpus.

    Fits run in spawned processes: their chain processes cannot nest in loky workers or threads.
    """
    CHECKPOINTS.mkdir(exist_ok=True)
    n_fits = max(1, effective_n_jobs(n_jobs) // chains)
    tasks = [(n, rep) for n in sorted(n_samples, reverse=True) for rep in range(n_repeats)]
    print(f"MCMC: {len(tasks)} fits, {n_fits} at a time with {chains} chains each", flush=True)
    with ProcessPoolExecutor(n_fits, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                run_mcmc_checkpoint,
                n,
                rep,
                fits["params"][(fits["n"] == n) & (fits["rep"] == rep)][0],
                chains,
                tune,
                draws,
            )
            for n, rep in tasks
        ]
        records = [future.result() for future in futures]
    return dict(zip(["n", "rep", *MCMC_FIELDS], map(np.array, zip(*records))))


def main(n_samples, n_repeats, n_repeats_mcmc, n_jobs, chains, tune, draws, csv):
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
        mcmc = run_mcmc(fits, n_samples, n_repeats_mcmc, n_jobs, chains, tune, draws) if n_repeats_mcmc else {}
        df = summarize(fits, mcmc, design)
        np.savez_compressed(
            ROOT / "data/fig08-fits.npz",
            methods=METHODS,
            param_names=PARAM_NAMES,
            hssm_names=MCMC_NAMES,
            **fits,
            **{f"mcmc_{key}": value for key, value in mcmc.items()},
        )
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
    parser.add_argument("--n-repeats", type=int, default=900, help="number of independent simulation repeats")
    parser.add_argument(
        "--n-repeats-mcmc", type=int, default=100, help="first repeats also fit with full_ddm MCMC (0 skips MCMC)"
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs (cpus)")
    parser.add_argument("--chains", type=int, default=4, help="MCMC chains per fit, run in parallel")
    parser.add_argument("--tune", type=int, default=1000, help="MCMC tuning iterations per chain")
    parser.add_argument("--draws", type=int, default=12000, help="MCMC draws per chain")
    parser.add_argument("--csv", action="store_true", help="replot saved data/fig08.csv")
    args = parser.parse_args()
    if args.n_repeats < 3 or args.n_jobs == 0 or len(set(args.n_samples)) != len(args.n_samples):
        parser.error("need at least three repeats, nonzero n-jobs and unique sample sizes")
    if args.n_repeats_mcmc and not 3 <= args.n_repeats_mcmc <= args.n_repeats or args.chains < 2:
        parser.error("n-repeats-mcmc must be 0 or between 3 and n-repeats; chains must be at least 2")
    main(
        args.n_samples,
        args.n_repeats,
        args.n_repeats_mcmc,
        args.n_jobs,
        args.chains,
        args.tune,
        args.draws,
        args.csv,
    )
