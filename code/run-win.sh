#!/usr/bin/env bash
set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"

# free-t 7p MCMC pseudo-true references (resumable)
# 25k: ~6 GB each, six at a time
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --n-trials 25000 --repeats 1000000:1000030 \
    --n-jobs 24 --tune 1000 --timeout-per-1000 inf --output data/fig08-reference --machine win  # 30 fits, ~17 h
# 50k: ~11 GB each, four at a time; 1000007 is the missing rep of the original ten (data are not synced, so name it
# explicitly: 1000005 was fitted on another machine and refitting it here would conflict)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --n-trials 50000 --repeats 1000007:1000008 \
    --n-jobs 16 --tune 1000 --timeout-per-1000 inf --output data/fig08-reference --machine win
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --n-trials 50000 --repeats 1000026:1000034 \
    --n-jobs 16 --tune 1000 --timeout-per-1000 inf --output data/fig08-reference --machine win  # 9 fits total, ~14-20 h
