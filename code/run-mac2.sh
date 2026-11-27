#!/usr/bin/env bash

set -e

# free-t 7p MCMC pseudo-true references at 100k (resumable). Samples are read from data/pseudotrue-samples, so
# each fit needs only ~2.5 GB; pull those first or each fit simulates in place (~19 GB)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 100000 --repeats 1000007:1000010 --n-jobs 4 --machine mac2  # 3 fits, one at a time, ~4.3 h each
