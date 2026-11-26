#!/usr/bin/env bash
set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# fixed-t 7p MCMC references (resumable; stop any time)
# 100k first: ~23 GB each, two at a time (watch memory in the first round)
pixi run -e code-ocean python code/fig08-reference-tfix.py --n-trials 100000 --repeats 1000000:1000006 --n-jobs 8 \
    --machine win  # 6 fits, 3 rounds of ~9 h
# then 25k: ~6 GB each, six at a time
pixi run -e code-ocean python code/fig08-reference-tfix.py --n-trials 25000 --repeats 1000010:1000034 --n-jobs 24 \
    --machine win  # 24 fits, 4 rounds of ~3-4 h
