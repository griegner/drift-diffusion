#!/usr/bin/env bash
set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# catch-up to ~1000 converged MCMC per n; MCMC only on repeats whose MLE already exists
pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 811:900 --methods mcmc --tune 1000 \
    --timeout-per-1000 2000 --output data/fig08-v1-win --machine win  # 89 fits
pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 358:479 --methods mcmc --tune 1000 \
    --timeout-per-1000 2000 --output data/fig08-v1-win --machine win  # 121 fits (MLE in fig08-v1-mac2)
