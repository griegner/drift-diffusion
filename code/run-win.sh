#!/usr/bin/env bash

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# free-t 7p MCMC pseudo-true references at 100k (resumable). Samples are read from data/pseudotrue-samples, so
# each fit needs only ~2.5 GB; pull those first or each fit simulates in place (~19 GB)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 100000 --repeats 1000000:1000004 --n-jobs 16 --machine win  # 4 fits, four at a time, ~14 h
