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
from matplotlib.ticker import StrMethodFormatter

EXAMPLE_N, EXAMPLE_REP = 3000, 0
ORDER = ["autocorrelation-robust", "misspecification-robust", "outer-product", "sample-hessian", "mcmc"]
MARKERS = dict(zip(ORDER, "os^Do"))
PARAMS = ["a", "t0", "v", "z"]
MOSAIC = [
    ["A_a", "A_a", "B"],
    ["A_t0", "A_t0", "B"],
    ["A_v", "A_v", "B"],
    ["A_z", "A_z", "B"],
    ["C_a", "D_a", "E_a"],
    ["C_t0", "D_t0", "E_t0"],
    ["C_v", "D_v", "E_v"],
    ["C_z", "D_z", "E_z"],
]
DODGE = 10  # trials between neighbouring methods in panels C-E
SHARE_Y = ["C", "D", "E"]  # panels whose four rows share a y-axis; remove a letter to free that panel's rows
SUMMARY = {  # panel: (score, methods, title)
    "C": ("bias", ["autocorrelation-robust", "mcmc"], "C. Point relative bias (%)"),
    "D": ("se_bias", ORDER, "D. Standard error relative bias (%)"),
    "E": ("coverage", ORDER, "E. 95% CI coverage (%)"),
}


def plot_example(axes, config):
    design, state, _, _ = trial_inputs(EXAMPLE_N, EXAMPLE_REP, config)
    values = [design["a"], design["t0"][state], np.abs(design["magnitude"]), design["z"][state]]
    for ax, value, label in zip(axes, values, [r"$a$", r"$t_0$", r"$|v|$", r"$z$"]):
        ax.step(np.arange(EXAMPLE_N), value, color="k", lw=0.5, where="post")
        for boundary in range(SESSION, EXAMPLE_N, SESSION):
            ax.axvline(boundary, color=".5", ls="--", lw=1)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)
        ax.margins(y=0.2)
        ax.tick_params(labelbottom=ax is axes[-1])
    axes[-1].set(xlim=(0, EXAMPLE_N), xlabel="Trial")
    axes[0].set_title(f"A. Example Simulation (n = {EXAMPLE_N:,}, {EXAMPLE_N // SESSION} sessions)")


def plot_runtime(ax, records, runtime_records, machines, per_n=250, stuck=3):
    """MLE: every benchmark fit; MCMC: the first per_n valid fits per n on the given machines. MCMC fit line and
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
    runtimes = {
        "mle": [
            (n, f["runtime"], True)
            for (ref, n, _, m), f in runtime_records.items()
            if not ref and m == "mle" and f["valid"]
        ],
        "mcmc": [(n, *run) for n, values in mcmc.items() for run in values[:per_n]],
    }
    rng = np.random.default_rng(0)
    for method, color, label in [
        ("mle", COLORS["autocorrelation-robust"], "4p MLE"),
        ("mcmc", COLORS["mcmc"], "7p MCMC"),
    ]:
        n, seconds, kept = np.array(runtimes[method]).T
        kept = kept.astype(bool)
        sizes, x = np.unique(n), np.linspace(n.min(), n.max(), 100)
        slope = n[kept] @ seconds[kept] / (n[kept] @ n[kept])  # line through the origin, fit to the kept fits
        means = np.array([seconds[kept & (n == s)].mean() for s in sizes])
        ax.plot(
            n + rng.uniform(-250, 250, len(n)), seconds / 60, "o", color=color, ms=3, mec="white", mew=0.3, alpha=0.5
        )
        ax.plot(x, slope * x / 60, color=color, lw=1)
        ax.plot([], [], "o", color=color, ms=5, label=f"{label}: {1000 * slope:.3g} s/1000 trials")
        ax.hlines(means / 60, sizes - 400, sizes + 400, color="k", lw=1.2)
        if not kept.all():  # dashed: mean of all fits, also the excluded ones
            ax.hlines(
                [seconds[n == s].mean() / 60 for s in sizes], sizes - 400, sizes + 400, color="k", lw=1.2, ls="--"
            )
        print(
            f"{method} mean runtime of kept fits (min):",
            dict(zip(sizes.astype(int).tolist(), (means / 60).round(3).tolist())),
        )
        print(
            f"{method} runtime fits (kept):", {int(s): f"{np.sum(n == s)} ({np.sum(kept & (n == s))})" for s in sizes}
        )
    ax.set(title="B. Runtime (minutes)", yscale="log", xlim=(0, sizes[-1] * 1.08), xticks=sizes, xlabel="n Trials")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,g}"))


def plot_summary(axes, result, counts, key, methods, title):
    sizes = np.array(list(result))
    offsets = DODGE * (np.arange(len(methods)) - (len(methods) - 1) / 2)
    for k, ax in enumerate(axes):
        for offset, method in zip(offsets, methods):
            value, half = np.array([[result[n][method][key][i][k] for n in sizes] for i in (0, 1)])
            style = dict(color=COLORS[method], marker=MARKERS[method], ms=5, mec="white", mew=0.4)
            ax.plot(sizes + offset, value, color=COLORS[method], lw=0.5)
            ax.errorbar(sizes + offset, value, half, ls="none", lw=1, **style)
        if key == "coverage":  # 95% Monte Carlo band around nominal coverage
            for n in sizes:
                band = 196 * np.sqrt(0.95 * 0.05 / counts[n])
                ax.fill_between([n - 3 * DODGE, n + 3 * DODGE], 95 - band, 95 + band, color=".92", lw=0, zorder=0)
        ax.axhline(95 if key == "coverage" else 0, color=".35", ls="--" if key == "coverage" else "-", lw=1)
        ax.grid(axis="y", alpha=0.15)
        ax.tick_params(axis="x", labelbottom=ax is axes[-1])
    axes[0].set_title(title)
    # if key == "coverage":
    #     axes[0].set_yticks([50, 95])
    axes[-1].set(xticks=sizes, xlim=(0, sizes[-1] * 1.08), xlabel="n Trials")
    axes[-1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def plot_figure(records, runtime_records, groups, config, machines):
    fig, ax = plt.subplot_mosaic(MOSAIC, figsize=(12, 11), layout="constrained", height_ratios=[1] * 4 + [1.4] * 4)
    for p in PARAMS[1:]:  # share x within panel A and within each panel of C-E; y within the SHARE_Y panels
        ax[f"A_{p}"].sharex(ax["A_a"])
        for panel in SUMMARY:
            ax[f"{panel}_{p}"].sharex(ax[f"{panel}_a"])
            # if panel in SHARE_Y:
            #     ax[f"{panel}_{p}"].sharey(ax[f"{panel}_a"])

    plot_example([ax[f"A_{p}"] for p in PARAMS], config)
    plot_runtime(ax["B"], records, runtime_records, machines)
    result = {n: scores(group) for n, group in groups.items()}
    counts = {n: min(len(points) for points, _, _ in group.values()) for n, group in groups.items()}
    for panel, (key, methods, title) in SUMMARY.items():
        plot_summary([ax[f"{panel}_{p}"] for p in PARAMS], result, counts, key, methods, title)
    for p, label in zip(PARAMS, QUANTITY_LABELS):
        ax[f"C_{p}"].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)

    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=6, lw=0.5, label=LABELS[m]) for m in ORDER]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    return fig


def main(args):
    config, records = collect_results(args.inputs)
    runtime_config, runtime_records = collect_results(args.runtime_inputs)
    if runtime_config != config:
        raise ValueError("Runtime benchmark settings differ from the fit settings")
    targets = {method: reference_targets(records, method) for method in ("mle", "mcmc")}
    if any(target is None for target in targets.values()):
        raise ValueError("Need the pseudo-true references (fig08-run.py --pseudo-true)")

    groups = {}
    for n in sorted(set(args.n_trials)):  # first b paired repeats (lowest repeat IDs), or every valid fit
        b = None if args.b == "all" else args.b
        group = estimates(records, n, targets, paired=b is not None)
        groups[n] = {method: (points[:b], ses[:b], ref) for method, (points, ses, ref) in group.items()}
        print(
            f"n={n}: using {len(groups[n]['autocorrelation-robust'][0])} MLE and {len(groups[n]['mcmc'][0])} MCMC fits"
        )

    plt.rcParams.update(json.loads((ROOT / "code/config.json").read_text())["rc-params"])
    plt.rcParams["font.size"] = 12
    plt.rcParams["ytick.labelsize"] = 8
    fig = plot_figure(records, runtime_records, groups, config, args.runtime_machines)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "fig08.svg")
    plt.close(fig)


def parse_args(argv=None):
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
        default=600,
        help="repeats per n: the first B paired repeats, or 'all' for every valid fit of each method (unpaired)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
