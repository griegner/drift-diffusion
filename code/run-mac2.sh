#!/usr/bin/env bash

set -e

# free-t 7p MCMC pseudo-true references (resumable). Samples are read from data/pseudotrue-samples, so each fit
# needs only ~2.5 GB; pull those first or 50k+ fits simulate in place (~0.18 GB per 1000 trials)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 50000 --repeats 1000018:1000026 --n-jobs 4 --machine mac2  # 3 left (1000023-25), ~2.2 h each
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 100000 --repeats 1000005:1000010 --n-jobs 4 --machine mac2  # 5 fits, ~4.3 h each
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 75000 --repeats 1000017:1000020 --n-jobs 4 --machine mac2  # 3 fits, ~3.2 h each; total ~38 h
