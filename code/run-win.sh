#!/usr/bin/env bash
# fig08 pseudo-true

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# Three MCMC fits at a time, four chains each: 12 chain workers; completed fits are skipped.
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --repeats 1000000:1000015 \
    --n-jobs 12 --chains 4 --tune 1000 --timeout-per-1000 inf --output data/fig08-reference-win --machine win
