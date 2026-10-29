#!/usr/bin/env python

"""figure 08: MLE with session trends, pupil effects, and hidden engagement states"""

# simulate one 1000-trial session per repeat: the pupil trace is fixed, hidden states and stimuli are redrawn
# fit the covariate 4p-DDM MLE (sample-Hessian and autocorrelation-robust SEs) and the seven-parameter HSSM
# full_ddm MCMC to every repeat; score both against pseudo-true targets; --replot replots saved fits

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
from hddm_wfpt import wfpt
from joblib import Parallel, delayed, effective_n_jobs
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from pseudotrue import pseudotrue, sampler_grid
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from drift_diffusion.model import DriftDiffusionModel, pdf
from drift_diffusion.sim import sample_from_pdf

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261020
N = 1000  # trials per repeat (one session)
FITS = ROOT / f"data/fig08-n{N}.npz"
TARGETS = ROOT / f"data/fig08-n{N}-pseudotrue.npz"
CHECKPOINTS = ROOT / "data/fig08-mcmc"  # MCMC ignores pupil, so fits from the five-session calibration still apply
METHODS = ["sample-hessian", "autocorrelation-robust"]
PARAM_NAMES = ["a_intercept", "a_time", "t0", "v_stim", "v_stim:pupil", "z"]
# rescale HSSM z -> 2*z - 1 and sz -> 2*sz
MCMC_NAMES = ["a", "t", "v_stim", "z", "sz", "sv", "st"]
MCMC_SCALE, MCMC_OFFSET = np.array([1, 1, 1, 2, 2, 1, 1]), np.array([0, 0, 0, -1, 0, 0, 0])
MCMC_FIELDS = ["runtime", "params", "sd", "r_hat", "ess", "valid"]
# shared targets: first four MCMC_NAMES and contrasts of MLE coefficients
QUANTITIES = ["a_mid", "t0", "v_avg", "z"]
QUANTITY_LABELS = [r"$\bar{a}$", r"$t_0$", r"$\bar{v}$", r"$z$"]
# fig07 palette: 4p-DDM in reds (dark for point estimates and AR, light for SH), 7p-DDM in blue
COLORS = {"sample-hessian": "#f19c9d", "autocorrelation-robust": "#8c1515", "mcmc": "#3b6fb6"}
PALE = {key: 0.3 * np.array(to_rgb(color)) + 0.7 for key, color in [("mle", "r"), ("mcmc", "#3b6fb6")]}


def simulation_design():
    """fixed pupil trace for one session, with drift calibrated to span 0.1 to the fig07 maximum"""

    rng = np.random.default_rng(np.random.SeedSequence([SEED, 17]))
    pupil = gaussian_filter1d(rng.normal(size=N), 20, mode="wrap")
    pupil = 0.5 + 0.15 * pupil / pupil.std()
    centered = pupil - pupil.mean()
    beta_p = (2.23 + 1.33 * np.sqrt(3) - 0.1) / np.ptp(pupil)
    beta_v = 0.1 - beta_p * centered.min()
    session_time = np.linspace(-0.5, 0.5, N)
    z = 0.008 + np.array([0, 2]) * (2 * 0.037 / 0.125) / np.sqrt(12)
    return dict(
        time=session_time,
        pupil_centered=centered,
        a=0.63 + 0.252 * session_time,
        t0=0.435,
        magnitude=beta_v + beta_p * centered,
        z=z,
    )


def comparable(design):
    """contrasts from the MLE coefficients to QUANTITIES, and their generating values"""

    L = np.zeros((len(QUANTITIES), len(PARAM_NAMES)))
    L[0, [0, 1]] = 1, design["time"].mean()  # a at mean session time
    L[1, 2] = 1
    L[2, [3, 4]] = 1, design["pupil_centered"].mean()  # drift at mean pupil
    L[3, 5] = 1
    truth = [design["a"].mean(), design["t0"], design["magnitude"].mean(), design["z"].mean()]
    return L, np.array(truth)


def simulate(design, rep):
    """sample RTs with persistent states and independent stimulus directions"""

    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, N, rep]))
    state = np.empty(N, dtype=int)
    state[0] = rng.integers(2)
    state[1:] = (state[0] + np.cumsum(rng.random(N - 1) < 0.05)) % 2
    stim = 2 * rng.integers(2, size=N) - 1
    y = sample_from_pdf(
        a=design["a"],
        t0=design["t0"],
        v=stim * design["magnitude"],
        z=design["z"][state],
        n_samples=N,
        random_state=int(rng.integers(2**32)),
    )
    X = pd.DataFrame(dict(time=design["time"], pupil_centered=design["pupil_centered"], stim=stim))
    return X, y, state


def pseudotrue_targets(design, cut=1e-12):
    """4p coefficients and 7p parameters maximizing each expected working log-likelihood under the sampler pmf"""

    # sampler pmf per stimulus and trial on the RT grid, averaged over the two equally likely states
    y = sampler_grid([design["t0"]])
    a, magnitude, yy = np.repeat(design["a"], len(y)), np.repeat(design["magnitude"], len(y)), np.tile(y, N)
    w = np.zeros((2, N, len(y)))
    for s, stim in enumerate((-1, 1)):
        for z in design["z"]:
            p = pdf(yy, a, design["t0"], stim * magnitude, z).reshape(N, -1)
            w[s] += 0.25 * p / p.sum(axis=1, keepdims=True)

    s, i, j = np.nonzero(w > cut)
    stim, time_, pupil = 2 * s - 1, design["time"][i], design["pupil_centered"][i]
    slope, intercept = np.polyfit(design["pupil_centered"], design["magnitude"], 1)
    mle = pseudotrue(
        y[j],
        w[s, i, j],
        lambda b, y: pdf(y, b[0] + b[1] * time_, b[2], stim * (b[3] + b[4] * pupil), b[5]),
        [0.63, 0.252, design["t0"], intercept, slope, design["z"].mean()],
    )

    # 7p-DDM has no covariates, so trials pool; HSSM's full_ddm likelihood with z, sz on the [0, 1] scale
    pooled = w.sum(axis=1)

    def loss(q):
        a, t, v, z, sz, sv, st = q
        if min(a, sv, sz, st) < 0 or not 0 < z - sz / 2 < z + sz / 2 < 1 or st / 2 >= t:
            return 1e10
        total = 0
        for s, stim in enumerate((-1, 1)):
            keep = pooled[s] > cut
            one = np.ones(keep.sum())
            logp = wfpt.wiener_logp_array(
                x=y[keep], v=stim * v * one, sv=sv * one, a=2 * a * one, z=z * one, sz=sz * one, t=t * one,
                st=st * one, err=1e-8, simps_err=1e-4,
            )  # fmt: skip
            total -= pooled[s][keep] @ np.where(np.isfinite(logp), logp, -66.1)
        return total / N

    q = np.array([0.63, 0.43, 2.27, 0.59, 0.15, 0.4, 0.02])
    for xatol, fatol in [(1e-7, 1e-12), (1e-8, 1e-13)]:
        options = dict(xatol=xatol, fatol=fatol, maxiter=20000, maxfev=20000, adaptive=True)
        q = minimize(loss, q, method="Nelder-Mead", options=options).x
    return mle, MCMC_SCALE * q + MCMC_OFFSET


def fit_mle(X, y):
    """fit from the package's default start (a=1, all else 0); time optimization plus covariance estimation"""

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
    """fit seven-parameter full_ddm with stimulus-coded drift and no other covariates; start chains at the MLE"""

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
    # st mixes slowly at zero; check convergence on the other six
    valid = bool(np.all(r_hat[:-1] < 1.05) and np.all(ess[:-1] >= 100))
    mean = MCMC_SCALE * summary["mean"].to_numpy() + MCMC_OFFSET
    sd = MCMC_SCALE * summary["sd"].to_numpy()
    return runtime, mean, sd, r_hat, ess, valid


def run_mcmc_checkpoint(rep, params_mle, chains, tune, draws):
    """save one MCMC fit to CHECKPOINTS so an interrupted run resumes where it stopped"""

    path = CHECKPOINTS / f"n{N}-rep{rep}-c{chains}-t{tune}-d{draws}.npz"
    if not path.exists():
        X, y, _ = simulate(simulation_design(), rep)
        seed = int(np.random.SeedSequence([SEED, 3, N, rep]).generate_state(1)[0])
        fit = fit_mcmc(X, y, params_mle, chains, tune, draws, seed)
        np.savez(path, **dict(zip(MCMC_FIELDS, fit)))
        print(f"MCMC rep={rep}: {fit[0]:.0f} s, converged={fit[-1]}", flush=True)
    with np.load(path) as saved:
        return rep, *(saved[name] for name in MCMC_FIELDS)


def run_mcmc(params_mle, n_jobs, chains, tune, draws):
    """run chains in parallel within a fit and fits in parallel across the remaining CPUs"""

    # spawn fits: chain processes cannot nest in loky workers or threads
    CHECKPOINTS.mkdir(exist_ok=True)
    n_fits = max(1, effective_n_jobs(n_jobs) // chains)
    print(f"MCMC: {len(params_mle)} fits, {n_fits} at a time with {chains} chains each", flush=True)
    with ProcessPoolExecutor(n_fits, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(run_mcmc_checkpoint, rep, params_mle[rep], chains, tune, draws)
            for rep in range(len(params_mle))
        ]
        records = [future.result() for future in futures]
    return dict(zip(["rep", *MCMC_FIELDS], map(np.array, zip(*records))))


def scores(fits, L, targets):
    """point and SE relative bias and 95% CI coverage (%), with 95% Monte Carlo half-widths from the CLT"""

    def metrics(params_, uncs_, reference):
        # the empirical SD of the point estimates serves as the true SE; each metric's Monte Carlo SE is the SD of
        # its per-repeat influence over sqrt(repeats), by the delta method for the SE ratio
        error, sd, unc = params_ - reference, params_.std(ddof=1, axis=0), uncs_.mean(axis=0)
        covered = abs(error) <= 1.96 * uncs_
        influence = {
            "bias": 100 * error / reference,
            "se_bias": 100 * ((uncs_ - unc) / sd - unc * ((params_ - params_.mean(axis=0)) ** 2 - sd**2) / (2 * sd**3)),
            "coverage": 100 * covered,
        }
        values = {
            "bias": 100 * error.mean(axis=0) / reference,
            "se_bias": 100 * (unc - sd) / sd,
            "coverage": 100 * covered.mean(axis=0),
        }
        return {key: (values[key], 1.96 * influence[key].std(ddof=1, axis=0) / np.sqrt(len(params_))) for key in values}

    mle, mcmc = fits["valid"], fits["mcmc_valid"]
    params_ = fits["params"][mle] @ L.T
    estimates = {
        method: (params_, np.sqrt(np.einsum("ij,rjk,ik->ri", L, fits["cov"][mle, m], L)), L @ targets["mle"])
        for m, method in enumerate(METHODS)
    }
    estimates["mcmc"] = fits["mcmc_params"][mcmc, :4], fits["mcmc_sd"][mcmc, :4], targets["mcmc"][:4]
    result = {}
    for method, (params_, uncs_, reference) in estimates.items():
        result[method] = metrics(params_, uncs_, reference)
        print(
            f"{method} ({len(params_)} repeats):", {key: np.round(value[0], 1) for key, value in result[method].items()}
        )
    return result


def plot_example(fig, spec, fits, design, targets):
    """panel A: one repeat's generating states, pseudo-true 4p trace, 4p fit with 95% AR band, and 7p iid draws"""

    # converged repeat whose 4p and 7p estimates lie jointly closest to their targets
    L, _ = comparable(design)
    estimates = [(fits["params"] @ L.T, L @ targets["mle"]), (fits["mcmc_params"][:, :4], targets["mcmc"][:4])]
    distance = sum(np.sum(((e - ref) / np.nanstd(e, axis=0)) ** 2, axis=1) for e, ref in estimates)
    rep = int(np.argmin(np.where(fits["valid"] & fits["mcmc_valid"], distance, np.inf)))
    beta, cov = fits["params"][rep], fits["cov"][rep, METHODS.index("autocorrelation-robust")]
    a, t0, v, z, sz, sv, st = fits["mcmc_params"][rep]

    one, zero = np.ones(N), np.zeros(N)
    X = {
        "a": np.column_stack([one, design["time"], zero, zero, zero, zero]),
        "t0": np.column_stack([zero, zero, one, zero, zero, zero]),
        "v": np.column_stack([zero, zero, zero, one, design["pupil_centered"], zero]),
        "z": np.column_stack([zero, zero, zero, zero, zero, one]),
    }
    rng = np.random.default_rng(0)
    draws = {
        "a": np.full(N, a),
        "t0": rng.uniform(t0 - st / 2, t0 + st / 2, N),
        "v": np.abs(rng.normal(v, sv, N)),
        "z": rng.uniform(z - sz / 2, z + sz / 2, N),
    }
    trial = np.arange(1, N + 1)
    axes = [fig.add_subplot(row) for row in spec.subgridspec(4, 1)]
    for ax, key, label in zip(axes, X, [r"$a$", r"$t_0$", r"$|v|$", r"$z$"]):
        mean, half = X[key] @ beta, 1.96 * np.sqrt(np.einsum("ij,jk,ik->i", X[key], cov, X[key]))
        ax.plot(trial, draws[key], color=PALE["mcmc"], lw=0.65, zorder=1)
        ax.fill_between(trial, mean - half, mean + half, color=PALE["mle"], zorder=2)
        ax.plot(trial, X[key] @ targets["mle"], "--", color="k", lw=1.2, zorder=4)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)
        ax.set_xlim(1, N)
        ax.tick_params(labelbottom=ax is axes[-1])
    axes[-1].step(trial, design["z"][simulate(design, rep)[2]], color="0.25", lw=0.6, where="post", zorder=3)
    axes[1].set_ylim(0.33, 0.53)
    axes[-1].set_xlabel("Trial")
    axes[0].set_title("A. Example Simulation and Model Fits")
    return axes


def plot(fits, design, targets):
    """A: example fit; B: runtime; C-E: point and SE relative bias and coverage, with 95% Monte Carlo intervals"""

    with (ROOT / "code/config.json").open() as f:
        plt.rcParams.update(json.load(f)["rc-params"])
    plt.rcParams["font.size"] = 12
    L, _ = comparable(design)
    result = scores(fits, L, targets)
    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    axes = plot_example(fig, grid[0, :2], fits, design, targets)
    axes += [fig.add_subplot(grid[0, 2])] + [fig.add_subplot(grid[1, k]) for k in range(3)]

    # B: runtime from the earlier across-n run on the same machine; only n=1000 feeds C-E, larger n faded
    with np.load(ROOT / "data/fig08-fits.npz") as old:
        sizes = [1000, 5000, 10000]
        runtimes = {
            "autocorrelation-robust": [old["runtime"][(old["n"] == n) & old["valid"]].mean() for n in sizes],
            "mcmc": [old["mcmc_runtime"][(old["mcmc_n"] == n) & old["mcmc_valid"]].mean() for n in sizes],
        }
    for method, runtime in runtimes.items():
        axes[4].plot(range(3), runtime, "o-", color=COLORS[method], ms=6, alpha=0.3)
        axes[4].plot(0, runtime[0], "o", color=COLORS[method], ms=6)
    axes[4].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[4].set(xlabel="n Trials", ylabel="Seconds per fit", xticks=range(3), xticklabels=[f"{n:,}" for n in sizes])
    axes[4].set_xlim(-0.45, 2.45)

    panels = [
        ("bias", ["autocorrelation-robust", "mcmc"], "Point relative bias (%)", (-3, 3)),
        ("se_bias", METHODS + ["mcmc"], "SE relative bias (%)", (-45, 45)),
        ("coverage", METHODS + ["mcmc"], "95% CI coverage (%)", (60, 101)),
    ]
    x = np.arange(len(QUANTITIES))
    for ax, (key, methods, label, ylim) in zip(axes[5:], panels):
        for method, offset in zip(methods, np.linspace(-0.2, 0.2, len(methods))):
            value, mcse = result[method][key]
            ax.errorbar(x + offset, value, yerr=mcse, fmt="o", color=COLORS[method], ms=6, mec="white", mew=0.5)
        ax.axhline(95 if key == "coverage" else 0, color="0.35", ls="--" if key == "coverage" else "-", lw=1, zorder=1)
        ax.set(ylabel=label, ylim=ylim, xticks=x, xticklabels=QUANTITY_LABELS, xlim=(-0.5, len(x) - 0.5))
        ax.tick_params(axis="x", labelsize=18)
    half = 196 * np.sqrt(0.95 * 0.05 / fits["mcmc_valid"].sum())  # 95% Monte Carlo band around nominal coverage
    axes[7].axhspan(95 - half, 95 + half, color="0.92", zorder=0)
    titles = ["B. Runtime", "C. Point Estimate Bias", "D. Standard Error Bias", "E. Confidence Interval Coverage"]
    for ax, title in zip(axes[4:], titles):
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    labels = ["4p-DDM MLE (SH SE)", "4p-DDM MLE (point, AR SE)", "7p-DDM MCMC"]
    handles = [
        Line2D([], [], color=color, marker="o", ms=6, ls="none", label=l) for color, l in zip(COLORS.values(), labels)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(ROOT / "results/fig08.svg")
    plt.close(fig)


def main(n_repeats, n_jobs, chains, tune, draws, replot):
    """fit all repeats and save them to data/, then plot"""

    design = simulation_design()
    print(design)
    if not replot:

        @delayed
        def run_simulation(rep):
            X, y, _ = simulate(design, rep)
            return rep, *fit_mle(X, y)

        with Parallel(n_jobs=n_jobs) as parallel:
            records = parallel(run_simulation(rep) for rep in range(n_repeats))
        fits = dict(zip(["rep", "runtime", "params", "cov", "valid"], map(np.array, zip(*records))))
        print(f"MLE: {fits['valid'].sum()}/{n_repeats} valid fits", flush=True)
        mcmc = run_mcmc(fits["params"], n_jobs, chains, tune, draws)
        print(f"MCMC: {mcmc['valid'].sum()}/{n_repeats} converged fits")
        np.savez_compressed(FITS, **fits, **{f"mcmc_{key}": value for key, value in mcmc.items()})
    if not TARGETS.exists():
        np.savez(TARGETS, **dict(zip(["mle", "mcmc"], pseudotrue_targets(design))))  # ~15 min
    with np.load(FITS) as saved, np.load(TARGETS) as targets:
        plot({key: saved[key] for key in saved}, design, dict(targets))


if __name__ == "__main__":
    # set script defaults

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-repeats", type=int, default=500, help="number of simulation repeats, each fit by MLE and MCMC"
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs (cpus)")
    parser.add_argument("--chains", type=int, default=4, help="MCMC chains per fit, run in parallel")
    parser.add_argument("--tune", type=int, default=1000, help="MCMC tuning iterations per chain")
    parser.add_argument("--draws", type=int, default=12000, help="MCMC draws per chain")
    parser.add_argument("--replot", action="store_true", help="replot saved data/fig08-n1000.npz")
    args = parser.parse_args()
    if args.n_repeats < 3 or args.n_jobs == 0 or args.chains < 2:
        parser.error("need at least three repeats, nonzero n-jobs and at least two chains")
    main(args.n_repeats, args.n_jobs, args.chains, args.tune, args.draws, args.replot)
