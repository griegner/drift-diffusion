#!/usr/bin/env python

"""figure 08 variant: panels C-E with one row per quantity and sample size on the x-axis (writes fig08-n.svg)"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fig08 import COLORS, QUANTITY_LABELS, ROOT, SESSION, collect_results, reference_targets, trial_inputs
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, StrMethodFormatter

# reuse the scores, method lists, and labels from fig08-plot.py (hyphenated, so not importable by name)
spec = importlib.util.spec_from_file_location("fig08_plot", Path(__file__).with_name("fig08-plot.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
EXAMPLE_N, EXAMPLE_REP = 3000, 0  # panel A: the first n=3000 repeat (three sessions)
ORDER = ["autocorrelation-robust", "misspecification-robust", "outer-product", "sample-hessian", "mcmc"]  # 10a-10d
MARKERS = dict(zip(ORDER, "os^Do"))  # the MLE SE methods share point estimates, so shape tells them apart
TITLES = {
    "bias": "C. Point relative bias (%)",
    "se_bias": "D. Standard error relative bias (%)",
    "coverage": "E. 95% CI coverage (%)",
}


def plot_example(axes, config):
    """generative trial-level parameters of the example repeat; dashed lines separate sessions"""

    design, state, _, _ = trial_inputs(EXAMPLE_N, EXAMPLE_REP, config)
    values = [design["a"], design["t0"][state], np.abs(design["magnitude"]), design["z"][state]]
    for ax, value, label in zip(axes, values, [r"$a$", r"$t_0$", r"$|v|$", r"$z$"]):
        ax.step(np.arange(EXAMPLE_N), value, color="k", lw=0.5, where="post")
        for boundary in range(SESSION, EXAMPLE_N, SESSION):
            ax.axvline(boundary, color=".5", ls="--", lw=1)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)
        ax.margins(y=0.2)
    axes[-1].set(xlim=(0, EXAMPLE_N), xlabel="Trial")
    axes[0].set_title(f"A. Example Simulation (n = {EXAMPLE_N:,}, {EXAMPLE_N // SESSION} sessions)")


def plot_runtime(ax, records, machines):
    """runtime per fit on log-log axes (slope = scaling with n); medians are printed for the caption"""

    rng = np.random.default_rng(0)
    for method, color in [("mle", COLORS["autocorrelation-robust"]), ("mcmc", COLORS["mcmc"])]:
        n, minutes = np.array(
            [
                (key[1], fit["runtime"] / 60)
                for key, fit in records.items()
                if not key[0] and key[3] == method and fit["valid"] and str(fit["machine"]) in machines
            ]
        ).T
        jitter = np.exp(rng.uniform(-0.08, 0.08, len(n)))  # multiplicative, so equal width on the log axis
        ax.plot(n * jitter, minutes, "o", color=color, ms=3, mec="white", mew=0.3, alpha=0.5)
        print(
            f"{method} median runtime (min):",
            {int(s): round(float(np.median(minutes[n == s])), 2) for s in np.unique(n)},
        )
    sizes = np.unique(n)
    ax.set(title="B. Runtime (minutes)", xlabel="n Trials", xscale="log", yscale="log", xticks=sizes)
    ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,g}"))


def plot_summary(axes, result, counts, key, methods):
    """one row per quantity; sample sizes along x with methods dodged and connected across sizes"""

    sizes = list(result)
    x = np.arange(len(sizes))
    methods = sorted(methods, key=ORDER.index)
    offsets = 0.15 * (np.arange(len(methods)) - (len(methods) - 1) / 2)
    for k, ax in enumerate(axes):
        for offset, method in zip(offsets, methods):
            value, half = np.array([[result[n][method][key][i][k] for n in sizes] for i in (0, 1)])
            style = dict(color=COLORS[method], marker=MARKERS[method], ms=5, mec="white", mew=0.4)
            ax.plot(x + offset, value, color=COLORS[method], lw=0.5)
            ax.errorbar(x + offset, value, half, ls="none", lw=1, **style)
        if key == "coverage":  # 95% Monte Carlo band around nominal coverage for each size's repeat count
            for i, n in enumerate(sizes):
                band = 196 * np.sqrt(0.95 * 0.05 / counts[n])
                ax.fill_between([i - 0.4, i + 0.4], 95 - band, 95 + band, color=".92", lw=0, zorder=0)
        ax.axhline(95 if key == "coverage" else 0, color=".35", ls="--" if key == "coverage" else "-", lw=1)
        ax.grid(axis="y", alpha=0.15)
        for boundary in x[:-1] + 0.5:  # separate sample sizes instead of tick marks
            ax.axvline(boundary, color=".8", lw=0.8, zorder=0)
        ax.tick_params(axis="x", length=0, labelbottom=ax is axes[-1])
    axes[0].set_title(TITLES[key])
    if key == "coverage":
        axes[0].set_yticks([50, 95])  # label the nominal level (shared across the column)
    axes[-1].set(xticks=x, xticklabels=[f"{n:,}" for n in sizes], xlim=(-0.5, len(sizes) - 0.5), xlabel="n Trials")


def plot_figure(records, groups, config, machines, output):
    # one flat 8 x 3 grid (no nesting) so the outer axis edges line up between the top and bottom halves
    fig = plt.figure(figsize=(12, 11), layout="constrained")
    grid = fig.add_gridspec(8, 3, height_ratios=[1] * 4 + [1.4] * 4)
    example = [fig.add_subplot(grid[0, :2])]
    example += [fig.add_subplot(grid[k, :2], sharex=example[0]) for k in range(1, 4)]
    for ax in example[:-1]:
        ax.tick_params(labelbottom=False)
    plot_example(example, config)
    plot_runtime(fig.add_subplot(grid[:4, 2]), records, machines)  # unit in the title: no y label gap
    axes = np.array([[fig.add_subplot(grid[4 + k, col]) for col in range(3)] for k in range(4)])
    for col in range(3):  # share y within each panel column
        for ax in axes[1:, col]:
            ax.sharey(axes[0, col])
    result = {n: base.scores(estimates) for n, estimates in groups.items()}
    counts = {n: len(estimates["mcmc"][0]) for n, estimates in groups.items()}
    for col, (key, methods, _, _) in enumerate(base.PANELS):
        plot_summary(axes[:, col], result, counts, key, methods)
    for ax, label in zip(axes[:, 0], QUANTITY_LABELS):
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)
    # columns fill first: 10a-10b, 10c-10d, then MCMC
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=6, lw=0.5, label=base.LABELS[m]) for m in ORDER]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "fig08-n.svg")
    plt.close(fig)


def main(args):
    config, records = collect_results(args.inputs)
    targets = {method: reference_targets(records, method) for method in ("mle", "mcmc")}
    if any(target is None for target in targets.values()):
        raise ValueError("Need the pseudo-true references (fig08-run.py --pseudo-true)")
    plt.rcParams.update(json.loads((ROOT / "code/config.json").read_text())["rc-params"])
    plt.rcParams["font.size"] = 12
    groups = {n: base.estimates(records, n, targets) for n in sorted(set(args.n_trials))}
    args.output.mkdir(parents=True, exist_ok=True)
    plot_figure(records, groups, config, args.runtime_machines, args.output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    folders = [ROOT / f"data/fig08-v1-{machine}" for machine in ("mac1", "mac2", "win")] + [
        ROOT / "data/fig08-reference"
    ]
    parser.add_argument("--inputs", type=Path, nargs="+", default=folders)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--n-trials", type=int, nargs="+", default=[1000, 3000, 5000, 10000], help="panels C-E sizes")
    parser.add_argument("--runtime-machines", nargs="+", default=["mac1", "mac2"], help="panel B machines")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        main(parse_args())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
