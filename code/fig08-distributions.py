#!/usr/bin/env python

"""distributions of fig08 point estimates and standard errors"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["a", "t0", "v", "z"]
TRUE = [0.630, 0.435, 2.317, 0.179]  # fig08 means
MLE_COLS = [0, 2, 3, 5]  # a_intercept, t0, v_stim, z: session time and pupil have mean 0, so these are the quantities
COLORS = {"MLE (SH)": "#f19c9d", "MLE (AR)": "#8c1515", "MCMC": "#3b6fb6"}

targets = np.load(ROOT / "data/fig08-pseudotrue.npz")
pseudo = {"MLE (AR)": targets["mle"][MLE_COLS], "MCMC": targets["mcmc"][:4]}

rows = []
for n in [1000, 5000]:
    mle = np.load(ROOT / f"data/fig08-mle-n{n}.npz")
    valid = mle["valid"]
    se = np.sqrt(np.diagonal(mle["cov"][valid], axis1=2, axis2=3)[..., MLE_COLS])
    for k, name in enumerate(NAMES):
        rows += [
            dict(n=n, quantity=name, method="MLE (AR)", point=p, se=s)
            for p, s in zip(mle["params"][valid][:, MLE_COLS[k]], se[:, 1, k])
        ]
        rows += [dict(n=n, quantity=name, method="MLE (SH)", point=np.nan, se=s) for s in se[:, 0, k]]
    for path in (ROOT / "data/fig08-mcmc").glob(f"n{n}-rep*.npz"):
        fit = np.load(path)
        if fit["valid"]:
            rows += [
                dict(n=n, quantity=name, method="MCMC", point=fit["params"][k], se=fit["sd"][k])
                for k, name in enumerate(NAMES)
            ]
df = pd.DataFrame(rows)
print(df.groupby(["n", "method"]).size() // len(NAMES))

fig, axes = plt.subplots(4, 4, figsize=(14, 10), layout="constrained")
for r, (value, n) in enumerate([("point", 1000), ("point", 5000), ("se", 1000), ("se", 5000)]):
    for k, name in enumerate(NAMES):
        ax = axes[r, k]
        if r % 2:
            ax.sharex(axes[r - 1, k])
        data = df[(df.n == n) & (df.quantity == name)].dropna(subset=value)
        bins = np.linspace(*df.loc[df.quantity == name, value].agg(["min", "max"]), 80)  # same bins for both n
        sns.histplot(
            data,
            x=value,
            hue="method",
            palette=COLORS,
            element="step",
            fill=False,
            stat="density",
            common_norm=False,
            bins=bins,
            legend=k == 0 and r in (0, 2),
            ax=ax,
        )
        if value == "point":
            ax.axvline(TRUE[k], color="k", lw=1.5)
            for method, target in pseudo.items():
                ax.axvline(target[k], color=COLORS[method], ls="--", lw=1.5)
        else:  # empirical SD of the point estimates as the true SE
            for method in ["MLE (AR)", "MCMC"]:
                ax.axvline(data.loc[data.method == method, "point"].std(), color=COLORS[method], ls="--", lw=1.5)
        ax.set(xlabel=f"{name} {'point estimate' if value == 'point' else 'SE'}", ylabel=f"n={n}" if k == 0 else "")
fig.suptitle("solid: true; dashed: pseudo-true (point rows) or empirical SD of point estimates (SE rows)")
fig.savefig(ROOT / "results/fig08-distributions.png", dpi=150)
