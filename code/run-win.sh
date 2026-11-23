#!/usr/bin/env bash

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 1000:1280 --tune 1000 --timeout-per-1000 2000 \
    --output data/fig08-v1-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 1300:1470 --tune 1000 --timeout-per-1000 2000 \
    --output data/fig08-v1-win --machine win
