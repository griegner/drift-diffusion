#!/usr/bin/env python

"""figure 08: model fits, runtimes, and estimate/SE distributions"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fig08 import COLORS, METHODS, QUANTITY_LABELS, ROOT, SESSION, collect_results, contrasts, trial_inputs
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

LABELS = {
    "sample-hessian": "4p MLE (10d) Sample Hessian",
    "outer-product": "4p MLE (10c) Outer Product",
    "misspecification-robust": "4p MLE (10b) Misspec. Robust",
    "autocorrelation-robust": "4p MLE (10a) Autocorr. Robust",
    "mcmc": "7p MCMC",
}
PANELS = [
    ("bias", ["autocorrelation-robust", "mcmc"], "C. Point Estimate Bias", "Point relative bias (%)"),
    ("se_bias", METHODS + ["mcmc"], "D. Standard Error Bias", "SE relative bias (%)"),
    ("coverage", METHODS + ["mcmc"], "E. Confidence Interval Coverage", "95% CI coverage (%)"),
]


def fits_for(records, n, method, valid=False):
    return [
        fit
        for (reference, size, _, name), fit in sorted(records.items())
        if not reference and size == n and name == method and (fit["valid"] or not valid)
    ]


def estimates(records, n, targets):
    """(points, SEs, pseudo-true) of the four reported quantities per method"""

    L = contrasts(n)
    mle, mcmc = (fits_for(records, n, method, valid=True) for method in ("mle", "mcmc"))
    points = np.array([f["params"] for f in mle]) @ L.T
    cov = np.array([f["cov"] for f in mle])
    result = {
        method: (points, np.sqrt(np.einsum("ij,rjk,ik->ri", L, cov[:, m], L)), L @ targets["mle"])
        for m, method in enumerate(METHODS)
    }
    result["mcmc"] = (
        np.array([f["params"] for f in mcmc])[:, :4],
        np.array([f["sd"] for f in mcmc])[:, :4],
        targets["mcmc"][:4],
    )
    return result


def scores(estimates):
    """point/SE relative bias and coverage (%), each with a 95% Monte Carlo half-width"""

    result = {}
    for method, (params, uncs, reference) in estimates.items():
        error, sd, unc = params - reference, params.std(ddof=1, axis=0), uncs.mean(axis=0)
        # influence functions give Monte Carlo SEs via the delta method
        influence = dict(
            bias=100 * error / reference,
            se_bias=100 * ((uncs - unc) / sd - unc * ((params - params.mean(axis=0)) ** 2 - sd**2) / (2 * sd**3)),
            coverage=100 * (abs(error) <= 1.96 * uncs),
        )
        values = {key: value.mean(axis=0) for key, value in influence.items()} | {"se_bias": 100 * (unc - sd) / sd}
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


def plot_runtime(ax, records, machines):
    """runtime per fit on a log scale; medians are printed for the caption"""

    rng = np.random.default_rng(0)
    for method, name, color in [
        ("mle", "4p MLE", COLORS["autocorrelation-robust"]),
        ("mcmc", "7p MCMC", COLORS["mcmc"]),
    ]:
        n, minutes = np.array(
            [
                (key[1], fit["runtime"] / 60)
                for key, fit in records.items()
                if not key[0] and key[3] == method and fit["valid"] and str(fit["machine"]) in machines
            ]
        ).T
        sizes = np.unique(n)
        jitter = rng.uniform(-0.025, 0.025, len(n)) * sizes.max()
        ax.plot(n + jitter, minutes, "o", color=color, ms=3, mec="white", mew=0.3, alpha=0.5)
        print(f"{name} median runtime (min):", {int(s): round(float(np.median(minutes[n == s])), 2) for s in sizes})
    ax.set(
        title="B. Runtime",
        xlabel="n Trials",
        ylabel="Minutes",
        xticks=sizes,
        xlim=(0, 1.05 * sizes.max()),
        yscale="log",
    )
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,g}"))


def plot_summary(spec, result, counts, key, methods, title, ylabel):
    """one row per sample size; methods dodged within each quantity"""

    axes = spec.subgridspec(len(result), 1).subplots(sharex=True, sharey=True, squeeze=False)[:, 0]
    for ax, (n, scores_n) in zip(axes, result.items()):
        for offset, method in zip(0.15 * (np.arange(len(methods)) - (len(methods) - 1) / 2), methods):
            value, half = scores_n[method][key]
            ax.errorbar(
                np.arange(4) + offset, value, half, fmt="o", color=COLORS[method], ms=5, mec="white", mew=0.4, lw=1
            )
        if key == "coverage":
            band = 196 * np.sqrt(0.95 * 0.05 / counts[n])
            ax.axhspan(95 - band, 95 + band, color=".92", zorder=0)
        ax.axhline(95 if key == "coverage" else 0, color=".35", ls="--" if key == "coverage" else "-", lw=1)
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_title(title)
    axes[len(axes) // 2].set_ylabel(ylabel)
    axes[-1].set(xticks=np.arange(4), xticklabels=QUANTITY_LABELS, xlim=(-0.5, 3.5))
    axes[-1].tick_params(axis="x", labelsize=18)


def plot_figure(records, groups, example, config, targets, machines, output):
    result = {n: scores(estimates) for n, estimates in groups.items()}
    for n, scores_n in result.items():
        for method, metrics in scores_n.items():
            print(f"n={n} {method}:", {key: np.round(value, 1).tolist() for key, (value, _) in metrics.items()})
    fig = plt.figure(figsize=(12, 8.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    plot_example(fig, grid[0, :2], records, *example, config, targets)
    plot_runtime(fig.add_subplot(grid[0, 2]), records, machines)
    counts = {n: len(estimates["mcmc"][0]) for n, estimates in groups.items()}
    for col, panel in enumerate(PANELS):
        plot_summary(grid[1, col], result, counts, *panel)
    handles = [Line2D([], [], color=color, marker="o", ms=6, ls="none", label=LABELS[m]) for m, color in COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "fig08.svg")
    plt.close(fig)


def plot_distributions(groups, config, output):
    """review only: histograms of point estimates and SEs per sample size"""

    s = config["simulation"]
    truth = [s["A"], s["T0"], s["V_PUPIL_MEAN"], np.mean(s["Z_STATES"])]
    fig, axes = plt.subplots(2 * len(groups), 4, figsize=(14, 5 * len(groups)), layout="constrained", squeeze=False)
    for index, (value, methods) in enumerate([("point estimate", PANELS[0][1]), ("SE", PANELS[1][1])]):
        for k in range(4):
            pooled = [estimates[m][index][:, k] for estimates in groups.values() for m in methods]
            bins = np.histogram_bin_edges(np.concatenate(pooled), bins=79)
            for row, (n, estimates) in enumerate(groups.items()):
                ax = axes[row + index * len(groups), k]
                for method in methods:
                    points, uncs, reference = estimates[method]
                    color = COLORS[method]
                    ax.hist(
                        (points, uncs)[index][:, k],
                        bins=bins,
                        density=True,
                        histtype="step",
                        color=color,
                        label=LABELS[method],
                    )
                    ax.axvline(reference[k] if index == 0 else points[:, k].std(ddof=1), color=color, ls="--", lw=1.5)
                if index == 0:
                    ax.axvline(truth[k], color="k", lw=1.5)
                ax.set(xlabel=f"{QUANTITY_LABELS[k]} {value}", ylabel=f"n={n}" if k == 0 else "")
    for ax in axes[:: len(groups), 0]:  # first point and first SE row
        ax.legend(fontsize=9)
    fig.suptitle("solid: true; dashed: pseudo-true (point rows) or empirical SD of point estimates (SE rows)")
    fig.savefig(output / "fig08-distributions.svg")
    plt.close(fig)


def main(args):
    config, records = collect_results(args.inputs)
    n, rep = args.example
    eligible = sorted(
        r
        for ref, size, r, m in records
        if (ref, size, m) == (False, n, "mle")
        and all(records.get((False, n, r, method), {}).get("valid") for method in ("mle", "mcmc"))
    )
    if rep not in eligible:
        raise ValueError(f"Need --example {n}:REP with both fits valid; available IDs: {eligible}")
    targets = {}
    for method in ("mle", "mcmc"):
        reference = [fit for key, fit in records.items() if key[0] and key[3] == method]
        if len(reference) != 1 or not reference[0]["valid"]:
            raise ValueError(f"Need a valid {method} reference; run fig08-run.py --pseudo-true and include its folder")
        targets[method] = reference[0]["params"]
    sizes = sorted(set(args.n_trials))
    for size in sizes:
        for method in ("mle", "mcmc"):
            fits = fits_for(records, size, method)
            passed = sum(bool(f["valid"]) for f in fits)
            print(f"n={size} {method}: {passed}/{len(fits)} passed; {dict(Counter(str(f['status']) for f in fits))}")
            if passed < 2:
                raise ValueError(f"Need at least two valid {method} fits for n={size}")
    with (ROOT / "code/config.json").open() as f:
        plt.rcParams.update(json.load(f)["rc-params"])
    plt.rcParams["font.size"] = 12
    groups = {size: estimates(records, size, targets) for size in sizes}
    args.output.mkdir(parents=True, exist_ok=True)
    plot_figure(records, groups, args.example, config, targets, args.runtime_machines, args.output)
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
    parser.add_argument("--n-trials", type=int, nargs="+", default=[1000, 3000, 5000], help="panels C-E sizes")
    parser.add_argument("--example", type=example_pair, required=True, metavar="N:REP", help="panel A example")
    parser.add_argument("--runtime-machines", nargs="+", default=["mac1", "mac2"], help="panel B machines")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        main(parse_args())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
