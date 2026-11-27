#!/usr/bin/env bash

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# free-t 7p MCMC pseudo-true references (resumable). Samples are read from data/pseudotrue-samples, so each fit
# needs only ~2.5 GB; pull those first or 50k+ fits simulate in place (~0.18 GB per 1000 trials)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 50000 --repeats 1000007:1000008 --n-jobs 24 --machine win  # the missing rep of the original ten (1000005 was fitted elsewhere)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 50000 --repeats 1000026:1000034 --n-jobs 24 --machine win  # 8 fits
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 75000 --repeats 1000000:1000015 --n-jobs 24 --machine win  # 15 fits, six at a time, ~10 h each; total ~36-40 h
