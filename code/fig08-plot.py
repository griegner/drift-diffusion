#!/usr/bin/env python

"""figure 08: model fits, runtimes, and estimate/SE distributions"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fig08 import (
    COLORS,
    METHODS,
    QUANTITY_LABELS,
    ROOT,
    SESSION,
    collect_results,
    contrasts,
    trial_inputs,
)
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

LABELS = {
    "sample-hessian": "(10d) Sample Hessian",
    "outer-product": "(10c) Outer Product",
    "misspecification-robust": "(10b) Misspec. Robust",
    "autocorrelation-robust": "(10a) Autocorr. Robust",
    "mcmc": "7p-DDM MCMC",
}


def fits_for(records, n, method, valid=False):
    return [
        fit
        for (reference, size, rep, name), fit in sorted(records.items())
        if not reference and size == n and name == method and (not valid or fit["valid"])
    ]


def estimates(records, n, targets):
    L = contrasts(n)
    mle, mcmc = (fits_for(records, n, method, valid=True) for method in ("mle", "mcmc"))
    points = np.array([f["params"] for f in mle]).reshape(-1, 6) @ L.T
    cov = np.array([f["cov"] for f in mle]).reshape(-1, len(METHODS), 6, 6)
    result = {
        method: (points, np.sqrt(np.einsum("ij,rjk,ik->ri", L, cov[:, m], L)), L @ targets["mle"])
        for m, method in enumerate(METHODS)
    }
    result["mcmc"] = (
        np.array([f["params"] for f in mcmc]).reshape(-1, 7)[:, :4],
        np.array([f["sd"] for f in mcmc]).reshape(-1, 7)[:, :4],
        targets["mcmc"][:4],
    )
    return result


def scores(estimates):
    """compute point/SE bias and coverage with 95% Monte Carlo intervals"""

    result = {}
    for method, (params, uncs, reference) in estimates.items():
        if len(params) < 2:
            result[method] = {key: (np.full(4, np.nan), np.full(4, np.nan)) for key in ("bias", "se_bias", "coverage")}
            continue
        error, sd, unc = params - reference, params.std(ddof=1, axis=0), uncs.mean(axis=0)
        covered = abs(error) <= 1.96 * uncs
        with np.errstate(divide="ignore", invalid="ignore"):
            # influence functions give Monte Carlo SEs via the delta method
            influence = dict(
                bias=100 * error / reference,
                se_bias=100 * ((uncs - unc) / sd - unc * ((params - params.mean(axis=0)) ** 2 - sd**2) / (2 * sd**3)),
                coverage=100 * covered,
            )
            values = dict(
                bias=100 * error.mean(axis=0) / reference,
                se_bias=100 * (unc - sd) / sd,
                coverage=100 * covered.mean(axis=0),
            )
            result[method] = {
                key: (values[key], 1.96 * value.std(ddof=1, axis=0) / np.sqrt(len(params)))
                for key, value in influence.items()
            }
    return result


def plot_example(fig, spec, records, n, rep, config, targets):
    design, state, _, _ = trial_inputs(n, rep, config)
    mle, mcmc = (records[(False, n, rep, method)] for method in ("mle", "mcmc"))
    a, t0, v, z, sz, sv, st = mcmc["params"]
    rng = np.random.default_rng(0)
    draws = [
        np.full(n, a),
        rng.uniform(t0 - st / 2, t0 + st / 2, n),
        np.abs(rng.normal(v, sv, n)),
        rng.uniform(z - sz / 2, z + sz / 2, n),
    ]
    X = np.tile(np.eye(6)[[0, 2, 3, 5]][:, None, :], (1, n, 1))
    X[0, :, 1], X[2, :, 4] = design["time"], design["pupil_centered"]
    cov = mle["cov"][METHODS.index("autocorrelation-robust")]
    pale = {key: 0.3 * np.array(to_rgb(color)) + 0.7 for key, color in [("mle", "r"), ("mcmc", COLORS["mcmc"])]}
    trial = np.arange(1, n + 1)
    axes = [fig.add_subplot(row) for row in spec.subgridspec(4, 1)]
    for ax, B, samples, label in zip(axes, X, draws, [r"$a$", r"$t_0$", r"$|v|$", r"$z$"]):
        mean, half = B @ mle["params"], 1.96 * np.sqrt(np.einsum("ij,jk,ik->i", B, cov, B))
        ax.plot(trial, samples, color=pale["mcmc"], lw=0.65, zorder=1)
        ax.fill_between(trial, mean - half, mean + half, color=pale["mle"], zorder=2)
        ax.plot(trial, B @ targets["mle"], "--", color="k", lw=1.2, zorder=4)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)
        ax.set_xlim(1, n)
        ax.tick_params(labelbottom=ax is axes[-1])
    for ax, key in zip(axes[1::2], ["t0", "z"]):
        ax.step(trial, design[key][state], color=".25", lw=0.6, where="post", zorder=3)
    axes[1].set_ylim(0.33, 0.53)
    axes[-1].set_xlabel("Trial")
    axes[0].set_title(f"A. Example Simulation and Model Fits (repeat {rep})")


def plot_runtime(fig, spec, records, sizes):
    machines = sorted({str(f["machine"]) for key, f in records.items() if not key[0] and key[1] in sizes})
    grid = spec.subgridspec(len(machines), 1)
    rng = np.random.default_rng(0)
    for row, machine in enumerate(machines):
        ax = fig.add_subplot(grid[row])
        for n in sizes:
            for method, color in [("mle", COLORS["autocorrelation-robust"]), ("mcmc", COLORS["mcmc"])]:
                times = [f["runtime"] for f in fits_for(records, n, method, valid=True) if str(f["machine"]) == machine]
                ax.plot(
                    n + rng.uniform(-0.08 * min(sizes), 0.08 * min(sizes), len(times)),
                    times,
                    "o",
                    color=color,
                    ms=3,
                    mec="white",
                    mew=0.3,
                )
        ax.set(
            title=f"{'B. Runtime: ' if row == 0 else ''}{machine}",
            ylabel="Seconds",
            xticks=sizes,
            xlim=(0, max(sizes) + SESSION),
        )
        ax.title.set_fontsize(9)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(labelbottom=row == len(machines) - 1, labelsize=8)
        if row == len(machines) - 1:
            ax.set_xlabel("n Trials")


def plot_composite(records, n, rep, sizes, config, targets, estimates, output):
    result = scores(estimates)
    counts = [(len(fits_for(records, n, m, valid=True)), len(fits_for(records, n, m))) for m in ("mle", "mcmc")]
    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    fig.suptitle(f"n = {n:,} trials; MLE {counts[0][0]}/{counts[0][1]}, MCMC {counts[1][0]}/{counts[1][1]} fits pass")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    plot_example(fig, grid[0, :2], records, n, rep, config, targets)
    plot_runtime(fig, grid[0, 2], records, sizes)
    panels = [
        ("bias", ["autocorrelation-robust", "mcmc"], "Point relative bias (%)", (-3, 3), "C. Point Estimate Bias"),
        ("se_bias", METHODS + ["mcmc"], "SE relative bias (%)", (-45, 45), "D. Standard Error Bias"),
        ("coverage", METHODS + ["mcmc"], "95% CI coverage (%)", (60, 101), "E. Confidence Interval Coverage"),
    ]
    for col, (key, methods, label, ylim, title) in enumerate(panels):
        ax = fig.add_subplot(grid[1, col])
        bounds = []
        for method, offset in zip(methods, np.linspace(-0.2, 0.2, len(methods))):
            value, half = result[method][key]
            mask = np.isfinite(value) & np.isfinite(half)
            ax.errorbar(
                np.arange(4)[mask] + offset,
                value[mask],
                yerr=half[mask],
                fmt="o",
                color=COLORS[method],
                ms=6,
                mec="white",
                mew=0.5,
            )
            bounds.extend((value[mask] - half[mask]).tolist() + (value[mask] + half[mask]).tolist())
        if bounds:
            margin = 0.05 * max(max(bounds) - min(bounds), 1)
            ylim = min(ylim[0], min(bounds) - margin), max(ylim[1], max(bounds) + margin)
        ax.axhline(95 if key == "coverage" else 0, color=".35", ls="--" if key == "coverage" else "-", lw=1)
        if key == "coverage":
            half = 196 * np.sqrt(0.95 * 0.05 / counts[1][0])
            ax.axhspan(95 - half, 95 + half, color=".92", zorder=0)
        ax.set(ylabel=label, ylim=ylim, xticks=np.arange(4), xticklabels=QUANTITY_LABELS, xlim=(-0.5, 3.5), title=title)
        ax.tick_params(axis="x", labelsize=18)
        ax.grid(axis="y", alpha=0.15)
    fig.legend(
        handles=[
            Line2D([], [], color=color, marker="o", ms=6, ls="none", label=LABELS[method])
            for method, color in COLORS.items()
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(output / ("fig08.svg" if n == SESSION else f"fig08-n{n}.svg"))
    plt.close(fig)
    for method, metrics in result.items():
        print(
            f"n={n} {method} ({len(estimates[method][0])} valid):",
            {key: np.round(value[0], 1).tolist() for key, value in metrics.items()},
        )


def plot_distributions(groups, config, output):
    sizes = sorted(groups)
    s = config["simulation"]
    truth = [s["A"], s["T0"], s["V_PUPIL_MEAN"], np.mean(s["Z_STATES"])]
    fig, axes = plt.subplots(2 * len(sizes), 4, figsize=(14, 5 * len(sizes)), layout="constrained", squeeze=False)
    for value, index in [("point estimate", 0), ("SE", 1)]:
        methods = ["autocorrelation-robust", "mcmc"] if index == 0 else METHODS + ["mcmc"]
        for k in range(4):
            pooled = np.concatenate([groups[n][m][index][:, k] for n in sizes for m in methods])
            bins = np.histogram_bin_edges(pooled, bins=79)
            for row, n in enumerate(sizes):
                ax = axes[row + index * len(sizes), k]
                for method in methods:
                    points, uncs, reference = groups[n][method]
                    samples = (points if index == 0 else uncs)[:, k]
                    if len(samples):
                        ax.hist(
                            samples,
                            bins=bins,
                            density=True,
                            histtype="step",
                            color=COLORS[method],
                            label=LABELS[method],
                        )
                    line = reference[k] if index == 0 else points[:, k].std(ddof=1) if len(points) > 1 else np.nan
                    if np.isfinite(line):
                        ax.axvline(line, color=COLORS[method], ls="--", lw=1.5)
                if index == 0:
                    ax.axvline(truth[k], color="k", lw=1.5)
                ax.set(xlabel=f"{QUANTITY_LABELS[k]} {value}", ylabel=f"n={n}" if k == 0 else "")
                if row == 0 and k == 0:
                    ax.legend(fontsize=9)
    fig.suptitle("solid: true; dashed: pseudo-true (point rows) or empirical SD of point estimates (SE rows)")
    fig.savefig(output / "fig08-distributions.svg")
    plt.close(fig)


def validate_selection(records, sizes, examples):
    if not sizes:
        raise ValueError("No simulation results found")
    for n in sizes:
        eligible = sorted(
            rep
            for reference, size, rep, method in records
            if not reference
            and size == n
            and method == "mle"
            and records[(False, n, rep, "mle")]["valid"]
            and (False, n, rep, "mcmc") in records
            and records[(False, n, rep, "mcmc")]["valid"]
        )
        if n not in examples or examples[n] not in eligible:
            raise ValueError(f"Need --example {n}:REP with both fits valid; available IDs: {eligible}")


def main(args):
    config, records = collect_results(args.inputs)
    sizes = sorted(set(args.n_trials) if args.n_trials else {key[1] for key in records if not key[0]})
    examples = dict(args.example)
    if len(examples) != len(args.example):
        raise ValueError("Supply only one example per sample size")
    validate_selection(records, sizes, examples)
    targets = {}
    for method in ("mle", "mcmc"):
        reference = [fit for key, fit in records.items() if key[0] and key[3] == method]
        if len(reference) != 1 or not reference[0]["valid"]:
            raise ValueError(f"Need a valid {method} reference; run fig08-run.py --pseudo-true and include its folder")
        targets[method] = reference[0]["params"]
    with (ROOT / "code/config.json").open() as f:
        plt.rcParams.update(json.load(f)["rc-params"])
    plt.rcParams["font.size"] = 12
    groups = {n: estimates(records, n, targets) for n in sizes}
    args.output.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        for method in ("mle", "mcmc"):
            fits = fits_for(records, n, method)
            statuses = Counter(str(f["status"]) for f in fits)
            passed = sum(bool(f["valid"]) for f in fits)
            timed_out = statuses["timeout"]
            print(
                f"n={n} {method}: attempted={len(fits)}, passed={passed}, "
                f"failed={len(fits) - passed - timed_out}, timed_out={timed_out}; "
                f"statuses={dict(statuses)}"
            )
        plot_composite(records, n, examples[n], sizes, config, targets, groups[n], args.output)
    plot_distributions(groups, config, args.output)


def example_pair(value):
    try:
        n, rep = map(int, value.split(":"))
        if n >= SESSION and n % SESSION == 0 and rep >= 0:
            return n, rep
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("need N:REP with N a positive multiple of 1000 and REP >= 0")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=[ROOT / "data/fig08"])
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--n-trials", type=int, nargs="+", help="sample sizes; default: all discovered sizes")
    parser.add_argument("--example", type=example_pair, nargs="+", required=True, metavar="N:REP")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        main(parse_args())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
