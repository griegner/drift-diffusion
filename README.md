### drift-diffusion
> Quantifying Uncertainty in Drift Diffusion Models of Decision Making under Temporal Dependence and Misspecification

> <img src="./readme.png" width="800"/>

[![project poster](https://img.shields.io/badge/poster-PDF-red?style=flat&logo=google-docs&logoColor=white)](https://griegner.github.io/drift-diffusion/poster/poster-15NOV2025.pdf)
[![project slides](https://img.shields.io/badge/slides-PDF-red?style=flat&logo=google-slides&logoColor=white)](https://griegner.github.io/drift-diffusion/slides/slides-12NOV2025.pdf)
[![codecov](https://codecov.io/gh/griegner/drift-diffusion/graph/badge.svg?token=VP43QWD2NS)](https://codecov.io/gh/griegner/drift-diffusion)

**Project Organization**
```
.
├── drift_diffusion/
│   ├── model/              <- drift diffusion model class
│   ├── sim/                <- simulation functions
│   └── tests/              <- unit tests
├── docs/
│   └── ...                 <- latex/pdf documentation files
├── figures/
│   ├── datasets/           <- Reinagel 2013 rats 195 and 196
│   ├── results/            <- precomputed simulation and analysis results
│   ├── fig01.ipynb         <- drift diffusion model example
│   ├── fig01.py            <- ...
│   ├── fig02to04.ipynb     <- validation by simulation
│   ├── fig02to04.py        <- ...
│   ├── fig05to06.ipynb     <- application to rat decision making
│   ├── fig05to06.py        <- ...
│   └── fig06.ipynb         <- ...

├── LICENSE                 <- MIT license
├── pyproject.toml          <- python configuration and dependencies
└── README.md               <- this readme file
```

**Data Availability**

Sequences of choices and reaction times from two rats trained on a random dot motion task, originally published in [Reinagel (2013)](https://pubmed.ncbi.nlm.nih.gov/23840856/), are at `./figures/datasets/*.mat`; the function for loading and preprocessing is at `./figures/fig01.py/preproc_df()`. Figures 1 and 5-6 use a contiguous sequence of 42,754 trials from rat 195, selected after the rat reached 85% correct performance, spanning 100 days at constant 85% motion coherence.

Intermediate results for Figures 2–6 are stored at `./figures/results/*.{csv,npy}`. In each figure notebook, the flag `REFIT=False` (default) loads these precomputed results to reproduce the figures, while setting `REFIT=True` will recompute the results, requiring more time and compute.

**Code Installation**

Clone this repository:
```
git clone https://github.com/griegner/drift-diffusion.git
cd drift-diffusion
```

Create virtual environment using `pip` or `conda` (or `pixi`):
```
# pip
python3 -m venv .drift-diffusion
source .drift-diffusion/bin/activate
```

```
# conda
conda create --name drift-diffusion python
conda activate drift-diffusion
```

Install `drift-diffusion` and dependencies:
```
pip install --editable ".[notebooks]"
```

[Optional] Install development environment with `pixi`:
```
pixi install -e dev
pixi shell -e dev
```

[Optional] Run development tasks with `pixi`:
```
pixi run -e dev lint
pixi run -e dev test
```
