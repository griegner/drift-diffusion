#!/usr/bin/env python

"""figure 08 with sample size on the x-axis (writes fig08.svg)"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fig08 import (
    COLORS,
    LABELS,
    QUANTITY_LABELS,
    ROOT,
    SESSION,
    checkpoint_path,
    collect_results,
    estimates,
    reference_targets,
    scores,
    trial_inputs,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

EXAMPLE_N, EXAMPLE_REP = 3000, 0
ORDER = ["autocorrelation-robust", "misspecification-robust", "outer-product", "sample-hessian", "mcmc"]
MARKERS = dict(zip(ORDER, "os^Do"))
DODGE = 10  # trials between neighbouring methods in panels C-F
SUMMARY = [  # panels C-F: (score, methods, title)
    ("bias", ["autocorrelation-robust", "mcmc"], "C. Point relative bias (%)"),
    ("sd", ["autocorrelation-robust", "mcmc"], "D. Point relative SD (%)"),
    ("se_bias", ORDER, "E. SE relative bias (%)"),
    ("coverage", ORDER, "F. 95% CI coverage (%)"),
]


def plot_example(axes, config):
    """panel A: parameter trajectories of one example simulation"""

    design, state, _, _ = trial_inputs(EXAMPLE_N, EXAMPLE_REP, config)
    values = [design["a"], design["t0"][state], np.abs(design["magnitude"]), design["z"][state]]
    for ax, value, label in zip(axes, values, [r"$a$", r"$t_0$", r"$|v|$", r"$z$"]):
        ax.step(np.arange(EXAMPLE_N), value, color="k", lw=0.5, where="post")
        for boundary in range(SESSION, EXAMPLE_N, SESSION):
            ax.axvline(boundary, color=".5", ls="--", lw=1)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)
        ax.margins(y=0.2)
    axes[-1].set(xlim=(0, EXAMPLE_N), xlabel="Trial")
    panel_title(axes[0], f"A. Example Simulation (n = {EXAMPLE_N:,}, {EXAMPLE_N // SESSION} sessions)")


def plot_runtime(ax, records, runtime_records, machines, per_n, stuck=3):
    """MLE: the first per_n valid benchmark fits per n; MCMC: the first per_n valid fits per n on the given machines. MCMC fit line and
    solid bars exclude stuck fits (> stuck x median at n) and fits that ran at the same time as one on that machine"""

    runs = {}  # every MCMC run (also failed ones): (machine, n, rep) -> (start, end), end = file time
    for (reference, n, rep, method), fit in records.items():
        if not reference and method == "mcmc" and (machine := str(fit["machine"])) in machines:
            end = checkpoint_path(ROOT / f"data/fig08-v1-{machine}", method, n, rep, False).stat().st_mtime
            runs[machine, n, rep] = (end - fit["runtime"], end)
    median = {n: np.median([e - s for (_, m, _), (s, e) in runs.items() if m == n]) for _, n, _ in runs}
    slow = {k: (s, e) for k, (s, e) in runs.items() if e - s > stuck * median[k[1]]}
    excluded = {
        k for k, (s, e) in runs.items() if any(k[0] == j[0] and s < e2 and s2 < e for j, (s2, e2) in slow.items())
    }
    mcmc = {}
    for (reference, n, rep, method), fit in sorted(records.items()):  # sorted: lowest repeat IDs first
        if not reference and method == "mcmc" and fit["valid"] and str(fit["machine"]) in machines:
            mcmc.setdefault(n, []).append((fit["runtime"], (str(fit["machine"]), n, rep) not in excluded))
    mle = {}
    for (reference, n, rep, method), fit in sorted(runtime_records.items()):  # sorted: lowest repeat IDs first
        if not reference and method == "mle" and fit["valid"]:
            mle.setdefault(n, []).append((fit["runtime"], True))
    runtimes = {
        method: [(n, *run) for n, values in fits.items() for run in values[:per_n]]
        for method, fits in [("mle", mle), ("mcmc", mcmc)]
    }
    rng = np.random.default_rng(0)
    for method, color, label in [
        ("mle", COLORS["autocorrelation-robust"], "4p MLE"),
        ("mcmc", COLORS["mcmc"], "7p MCMC"),
    ]:
        n, seconds, kept = np.array(runtimes[method]).T
        kept = kept.astype(bool)
        sizes, x = np.unique(n), np.linspace(n.min(), n.max(), 100)
        slope, intercept = np.polyfit(n[kept], seconds[kept], 1)  # line with an intercept, fit to the kept fits
        means = np.array([seconds[kept & (n == s)].mean() for s in sizes])
        ax.plot(
            n + rng.uniform(-250, 250, len(n)), seconds / 60, "o", color=color, ms=4, mec="white", mew=0.3, alpha=0.5
        )
        ax.plot(x, (intercept + slope * x) / 60, color=color, lw=1)
        label = f"{label}: {significant(intercept, 1)} s + {significant(1000 * slope, 2)} s/1000 trials"
        ax.plot([], [], "o", color=color, ms=5, label=label)
        ax.hlines(means / 60, sizes - 400, sizes + 400, color="k", lw=1.2)
        if not kept.all():  # dashed: mean of all fits, also the excluded ones
            ax.hlines(
                [seconds[n == s].mean() / 60 for s in sizes], sizes - 400, sizes + 400, color="k", lw=1.2, ls="--"
            )
        print(
            f"{method} runtime fits (kept):", {int(s): f"{np.sum(n == s)} ({np.sum(kept & (n == s))})" for s in sizes}
        )
    panel_title(ax, "B. Runtime (minutes)")
    ax.set(yscale="log", ylabel="log(Minutes)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,g}"))
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    sample_size_axis(ax, sizes)


def panel_title(ax, title):
    """title with a bold panel letter: bold "A." followed by the plain text"""
    letter, text = title.split(" ", 1)
    bold = ax.set_title(letter, fontweight="bold")
    ax.annotate(f" {text}", (1, 0), xycoords=bold, va="bottom", fontsize=bold.get_fontsize())


def sample_size_axis(ax, sizes):
    """format a shared n-trials x-axis"""

    ax.set(xticks=sizes, xlim=(0, sizes[-1] * 1.08), xlabel="n Trials")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def significant(value, digits):
    """round to significant figures without an exponent (30, not 3e+01)"""
    return f"{float(f'{value:.{digits}g}'):g}"


def plot_summary(axes, result, counts, key, methods, title):
    """one column of panels C-F: a score per parameter row against sample size"""

    sizes = np.array(list(result))
    offsets = DODGE * (np.arange(len(methods)) - (len(methods) - 1) / 2)
    for k, ax in enumerate(axes):
        for offset, method in zip(offsets, methods):
            value, half = np.array([[result[n][method][key][i][k] for n in sizes] for i in (0, 1)])
            style = dict(color=COLORS[method], marker=MARKERS[method], ms=5, mec="white", mew=0.4)
            ax.plot(sizes + offset, value, color=COLORS[method], lw=0.5)
            ax.errorbar(sizes + offset, value, half, ls="none", lw=1, **style)
        ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))  # round steps
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.15)
        if key == "sd":  # spread from zero, no reference line
            ax.set_ylim(bottom=0)
        elif key == "coverage":  # nominal 95: Monte Carlo band, dashed line, bold top tick
            for n in sizes:
                band = 196 * np.sqrt(0.95 * 0.05 / counts[n])
                ax.fill_between([n - 3 * DODGE, n + 3 * DODGE], 95 - band, 95 + band, color=".92", lw=0, zorder=0)
            ax.axhline(95, color=".35", ls="--", lw=1)
            low, high = ax.get_ylim()
            ticks = ax.yaxis.get_major_locator().tick_values(low, high)
            ticks = [t for t in ticks if low <= t <= 95 - 0.8 * (ticks[1] - ticks[0])] + [95]  # no tick crowding 95
            ax.set_yticks(ticks, labels=[f"{t:g}" for t in ticks])
            ax.get_yticklabels()[-1].set_fontweight("bold")
        else:  # relative biases: zero line
            ax.axhline(0, color=".35", lw=1)
    panel_title(axes[0], title)
    sample_size_axis(axes[-1], sizes)


def plot_figure(records, runtime_records, groups, config, machines, runtime_b):
    """assemble panels A-F"""

    fig = plt.figure(figsize=(15, 10), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.4])
    left, right = top.subfigures(1, 2, width_ratios=[2.5, 1.5])  # top widths independent of the bottom columns
    plot_example(left.subplots(4, 1, sharex=True), config)
    plot_runtime(right.subplots(), records, runtime_records, machines, runtime_b)

    axes = bottom.subplots(4, len(SUMMARY), sharex=True)
    result = {n: scores(group) for n, group in groups.items()}
    counts = {n: min(len(points) for points, _, _ in group.values()) for n, group in groups.items()}
    for column, (key, methods, title) in zip(axes.T, SUMMARY):
        plot_summary(column, result, counts, key, methods, title)
    for ax, label in zip(axes[:, 0], QUANTITY_LABELS):
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)

    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=6, lw=0.5, label=LABELS[m]) for m in ORDER]
    bottom.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    return fig


def main(args):
    """load fits, select paired repeats, and write fig08.svg"""

    config, records = collect_results(args.inputs)
    runtime_config, runtime_records = collect_results(args.runtime_inputs)
    if runtime_config != config:
        raise ValueError("Runtime benchmark settings differ from the fit settings")
    targets = {method: reference_targets(records, method) for method in ("mle", "mcmc")}
    if any(target is None for target in targets.values()):
        raise ValueError("Need the pseudo-true references (fig08-run.py --pseudo-true)")

    groups = {}
    b = None if args.b == "all" else args.b  # first b paired repeats (lowest repeat IDs), or every valid fit
    for n in sorted(set(args.n_trials)):
        group = estimates(records, n, targets, paired=b is not None)
        groups[n] = {method: (points[:b], ses[:b], ref) for method, (points, ses, ref) in group.items()}
        print(
            f"n={n}: using {len(groups[n]['autocorrelation-robust'][0])} MLE and {len(groups[n]['mcmc'][0])} MCMC fits"
        )

    plt.rcParams.update(json.loads((ROOT / "code/config.json").read_text())["rc-params"])
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlelocation"] = "left"
    fig = plot_figure(records, runtime_records, groups, config, args.runtime_machines, args.runtime_b)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "fig08.svg")
    plt.close(fig)


def parse_args(argv=None):
    """command-line options"""

    parser = argparse.ArgumentParser(description=__doc__)
    folders = [ROOT / f"data/fig08-v1-{machine}" for machine in ("mac1", "mac2", "win")] + [
        ROOT / "data/fig08-reference"
    ]
    parser.add_argument("--inputs", type=Path, nargs="+", default=folders)
    parser.add_argument(
        "--runtime-inputs", type=Path, nargs="+", default=[ROOT / "data/fig08-v1-runtime"], help="panel B MLE"
    )
    parser.add_argument("--runtime-machines", nargs="+", default=["mac1", "mac2"], help="panel B MCMC")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--n-trials", type=int, nargs="+", default=[1000, 3000, 5000, 10000])
    parser.add_argument(
        "--b",
        type=lambda value: value if value == "all" else int(value),
        default=780,
        help="repeats per n: the first B paired repeats, or 'all' for every valid fit of each method (unpaired)",
    )
    parser.add_argument(
        "--runtime-b", type=int, default=280, help="panel B fits per n and method: the first valid fits per n"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
