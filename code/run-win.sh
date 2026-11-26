#!/usr/bin/env bash
set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# fixed-t 7p MCMC references, 50k trials (~11 GB each, four at a time; drop to --n-jobs 12 if memory exceeds ~55 GB)
pixi run -e code-ocean python code/fig08-reference-tfix.py --n-trials 50000 --repeats 1000000:1000010 --n-jobs 16 \
    --machine win  # 10 fits, ~20 h
