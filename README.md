### drift-diffusion
> Quantifying Uncertainty in Drift Diffusion Models of Decision Making under Temporal Dependence and Misspecification

<img src="./readme.png" width="800"/>

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

**Usage Examples**

`DriftDiffusionModel` fits decision making parameters by maximum likelihood estimation.
The four decision making parameters ($a, t_0, v, z$) can each be linear functions of coefficients and
sample-by-sample covariate columns in $X$; and each can be *fixed*, *free*, or *mixed*.
*Free* parameters/coefficients are estimated during `fit` and defined with Wilkinson notation to specify linear
relationships with covariates (e.g. `v = "+1 + coherence"`; see [formulaic](https://matthewwardrop.github.io/formulaic)).
*Fixed* parameters are set at initialization by passing a float. *Mixed* coefficients can be defined with a dict,
for example `v={"formula": "+1 + coherence", "fixed": {"coherence": 1.0}}`, where *fixed* coefficients are
excluded from optimization but included in likelihood evaluation.

```python
>>> import numpy as np, pandas as pd
>>> from drift_diffusion.model import DriftDiffusionModel
>>> from drift_diffusion.sim import sample_from_pdf
>>> n = 1000; stim = np.linspace(-1, +1, n); X = pd.DataFrame({"stim": stim})
```

(i) *Fixed*: fix `t0, v, z`; fit `a`
```python
>>> y = sample_from_pdf(a=1.0, t0=0.2, v=0.3, z=0, n_samples=n, random_state=0)
>>> ddm = DriftDiffusionModel(a="+1", t0=0.2, v=0.3, z=0).fit(X, y)  # intercept/constant `a`
>>> ddm.params_, ddm.covariance_ # parameter/standard error estimates for `a`
(array([0.99537521]), array([[0.0001443]]))
```

(ii) *Free*: set `v = beta * stim`; fit `a, beta, t0, z`
```python
>>> beta_v = 0.8; v = beta_v * stim # v as linear function of stimulus
>>> y = sample_from_pdf(a=1.0, t0=0.2, v=v, z=0, n_samples=n, random_state=1)
>>> ddm = DriftDiffusionModel(a="+1", t0="+1", v="-1 + stim", z="+1").fit(X, y)
>>> ddm.params_, np.sqrt(np.diag(ddm.covariance_))  # parameter/standard error estimates for `a, t0, beta_v, z`
(array([ 0.98992872,  0.19523826,  0.81235863, -0.00421529]),
 array([0.01535874, 0.00801454, 0.05959234, 0.0175263 ]))
```

(iii) *Mixed*: set `v = beta0 + beta1*stim + beta2*stim**2`; fix `beta1, beta2`; fit `beta0`
```python
>>> intercept = -0.5; v = intercept + stim + stim**2  # v as quadratic function of stimulus
>>> y = sample_from_pdf(a=1.0, t0=0.2, v=v, z=0, n_samples=n, random_state=2)
>>> ddm = DriftDiffusionModel(a=1.0, t0=0.2, v={"formula": "+1 + stim + {stim ** 2}", "fixed": {"stim": 1, "stim ** 2": 1}}, z=0).fit(X, y)
>>> ddm.params_, np.sqrt(ddm.covariance_)  # parameter/standard error estimates for intercept
(array([-0.54279923]), array([[0.03392439]]))
```

(iv) *Covariance Estimators*: `"sample-hessian"`-`"autocorrelation-robust"` are valid under increasingly general conditions on the sequence of choice and reaction times `y`
```python
>>> ddm.set_params(cov_estimator="all").fit(X, y).covariance_
{
    'sample-hessian': array([[0.00115086]]),
    'outer-product': array([[0.00110442]]),
    'misspecification-robust': array([[0.00119926]]),
    'autocorrelation-robust': array([[0.00119337]])
}
```
