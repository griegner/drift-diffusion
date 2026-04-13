#!/usr/bin/env python

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from drift_diffusion.model import DriftDiffusionModel, pdf
from drift_diffusion.sim import sim_ddm


def _mat_to_pd(mat):
    """load matlab file and select trial variables"""
    mat = loadmat(mat)
    mat = {k: v.squeeze() for k, v in mat.items() if isinstance(v, np.ndarray) and v.shape == mat["RT"].shape}
    return pd.DataFrame(mat)


def preproc_df(path="./code_ocean/data/Rat195Vectors_241025.mat"):
    """load and preprocess the dataframe for rat195 from Reinagel 2013"""
    R, L = 0.0, np.pi  # dot movement
    df = (
        _mat_to_pd(path)
        .query("Valid == 1 and RT == RT")  # keep valid trials and non-null RT
        .assign(trialDate=lambda x: pd.to_datetime(x["trialDate"] - 719529, unit="D"))
        .set_index("trialDate")
        .sort_index()
        .loc["2008-12-03":"2009-03-12"]  # select 24h sessions with constant |coh|
    )

    shifted_index = (df.index - pd.Timedelta(hours=18)).floor("D")  # 6pm-to-6pm days
    all_days = pd.date_range(shifted_index.min(), shifted_index.max(), freq="D")
    day_map = {day: i + 1 for i, day in enumerate(all_days)}

    df = df.assign(
        LR=lambda x: x["dotDirection"].map({R: +1, L: -1}) * x["correct"].map({1: 1, 0: -1}),  # +1 R, -1 L choice
        y=lambda x: x["RT"] * x["LR"],  # signed RT
        coherence=lambda x: x["dotDirection"].map({R: +1, L: -1}) * x["coherence"],  # +coh R, -coh L dot movement
        day=shifted_index.map(day_map),  # 6pm-to-6pm day
        hour=lambda x: x.index.hour + 1,  # 1 to 24
        trial=lambda x: ((x.groupby("day").cumcount() + 1) // 20) * 20,  # trials in day
        reward=lambda x: x["correct"] * x["proposedReward"],
    )

    return df


def fig01a(path="./code_ocean/results/fig01a.pdf"):
    """plot and save fig01a"""
    a, t0, v, z = 1, 0.3, 1, 0.3
    fig, axs = plt.subplots(3, 1, figsize=(6, 3), sharex=True, height_ratios=[0.4, 1, 0.1], layout="constrained")
    for s in [5, 7]:
        z_t = sim_ddm(dt=0.001, t=t0, z=z, v=v, a=a * 2, seed=s)
        axs[1].plot(np.arange(len(z_t)) * 0.001, z_t, "r", alpha=1 if s == 5 else 0.5, lw=1)
    axs[1].set(ylim=(-a - 0.2, a + 0.2), xlim=(0, 1.5), ylabel=r"$Z_t$", xlabel=r"time ($t$)")
    axs[1].tick_params(labelbottom=True)
    [axs[1].axhline(val, ls="--", c="k", lw=1) for val in (a, -a)]
    y = np.linspace(0, 1.5, 1000)
    axs[0].plot(y, pdf(y, a, t0, v, z), "r")
    axs[2].plot(y, pdf(-y, a, t0, v, z), "r", alpha=0.5)
    axs[2].invert_yaxis()
    [ax.axis("off") for ax in (axs[0], axs[2])]
    fig.savefig(path)


def fig01b(ddms, blocks, path="./code_ocean/results/fig01b.pdf"):
    """plot and save fig01b"""
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = {"R": "r", "L": "b"}
    y = np.linspace(-3, +3, 1000)
    for label, color in colors.items():
        a, t0, beta_v, z = ddms[label].params_
        v = beta_v * (0.85 if label == "R" else -0.85)
        ax.hist(
            blocks[label]["y"],
            bins=150,
            histtype="step",
            density=True,
            color=color,
            alpha=0.5,
            lw=2,
            label=r"$\hat{f}_0$",
        )
        ax.plot(y, pdf(y, a=a, t0=t0, v=v, z=z), color=color, label=r"$\hat{f}^*$")
    ax.legend()
    ax.set(xlim=[-3, +3], ylabel=r"$f$", xlabel=r"$RT$")
    fig.savefig(path)


def main(n_samples):
    """fig01a,b"""
    # fit ddm to block of rightward and leftward trials
    df195 = preproc_df()
    ddms = {}
    blocks = {}
    dir_labels = {0: "R", np.pi: "L"}
    min_len = min([len(df195.query("dotDirection == @d")) for d in dir_labels])
    rng = np.random.default_rng(seed=0)
    idx = rng.integers(0, min_len - n_samples + 1)
    for d, label in dir_labels.items():
        block = df195.query("dotDirection == @d").iloc[idx : idx + n_samples]
        blocks[label] = block
        ddms[label] = DriftDiffusionModel(a="+1", t0="+1", v="-1 + coherence", z="+1")
        ddms[label].fit(block, block["y"])

    fig01a("./code_ocean/results/fig01a.pdf")
    fig01b(ddms, blocks, "./code_ocean/results/fig01b.pdf")


if __name__ == "__main__":
    """set script defaults"""

    # input arguments
    parser = argparse.ArgumentParser(description="Generate Figure 01 panels.")
    parser.add_argument("--n-samples", type=int, default=2000, help="number of trials sampled per direction.")
    args = parser.parse_args()

    # figure defaults
    with open("./code_ocean/code/config.json") as f:
        rc_params = json.load(f)["rc-params"]
        plt.rcParams.update(rc_params)

    main(args.n_samples)
