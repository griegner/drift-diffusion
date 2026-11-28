#!/usr/bin/env bash

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# more free-t 7p MCMC pseudo-true references at 100k (resumable); can start alongside the previous 100k run, whose
# reps differ. Samples are read from data/pseudotrue-samples (~2.5 GB per fit); pull those first
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 100000 --repeats 1000010:1000014 --n-jobs 16 --machine win  # 4 fits, four at a time, ~14-16 h (slower until the previous four finish)
