#!/usr/bin/env bash
# fits machine: n=10000; mac2 takes 0:500, mac1 900:1300

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"


pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 500:900 --tune 1000 --timeout-per-1000 2000 \
    --output data/fig08-v1-win --machine win
