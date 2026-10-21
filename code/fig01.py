#!/usr/bin/env python

"""Figure 01: Drift Diffusion Model"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from drift_diffusion.model import pdf
from drift_diffusion.sim import sim_ddm

with open("./config.json") as f:
    rc_params = json.load(f)["rc-params"]
    plt.rcParams.update(rc_params)


def fig01a(params, path):
    """plot and save fig01a"""

    a, t0, v, z = params
    fig, axs = plt.subplots(3, 1, figsize=(6, 3), sharex=True, height_ratios=[0.4, 1, 0.1])
    for s in [5, 7]:
        z_t = sim_ddm(dt=0.001, t=t0, z=z, v=v, a=a * 2, seed=s)
        axs[1].plot(np.arange(len(z_t)) * 0.001, z_t, "r", alpha=1 if s == 5 else 0.5, lw=1)
    axs[1].set(ylim=(-a - 0.2, a + 0.2), xlim=(0, 1.5), ylabel=r"$Z_t$", xlabel=r"time ($t$)")
    axs[1].tick_params(labelbottom=True)
    [axs[1].axhline(val, ls="--", c="k", lw=1) for val in (a, -a)]
    axs[1].axhline(0, c="gray", lw=0.5)
    y = np.linspace(0, 1.5, 1000)
    axs[0].plot(y, pdf(y, a, t0, v, z), "r")
    axs[2].plot(y, pdf(-y, a, t0, v, z), "r", alpha=0.5)
    axs[2].invert_yaxis()
    [ax.axis("off") for ax in (axs[0], axs[2])]
    fig.savefig(path)


def fig01b(params, path):
    """plot and save fig01b"""

    y = np.linspace(-3, +3, 1000)
    a, t0, v, z = params
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(y, pdf(y, a=a, t0=t0, v=v, z=z), color="r", label="rightward dot motion")
    ax.plot(y, pdf(y, a=a, t0=t0, v=-v, z=z), color="b", label="leftward dot motion")
    ax.set(xlim=[-3, +3], ylabel=r"$f$", xlabel=r"leftward choices $\qquad Y \qquad$ rightward choices")
    ax.legend()
    fig.savefig(path)


def main(params):
    """fig01a,b"""

    fig01a(params, "../results/fig01a")
    fig01b(params, "../results/fig01b")


if __name__ == "__main__":
    """set script defaults"""

    parser = argparse.ArgumentParser(description="Figures 02-04: Simulation Results")
    parser.add_argument("--params", nargs=4, type=str, default=[1, 0.3, 1, 0.3], help="a, t0, v, z parameter values")
    args = parser.parse_args()

    main(params=args.params)
