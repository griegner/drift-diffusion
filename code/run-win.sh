#!/usr/bin/env bash
# fits machine: n=5000

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"


pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 0:600 --tune 1000 --timeout-per-1000 2000 \
    --output data/fig08-win --machine win
