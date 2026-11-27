#!/usr/bin/env bash

set -e

# free-t 7p MCMC pseudo-true references (resumable). Samples are read from data/pseudotrue-samples, so each fit
# needs only ~2.5 GB; pull those first or 50k+ fits simulate in place (~0.18 GB per 1000 trials)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 50000 --repeats 1000010:1000018 --n-jobs 4 --machine mac1  # 3 left (1000015-17), ~2.2 h each
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 100000 --repeats 1000000:1000005 --n-jobs 4 --machine mac1  # 5 fits, ~4.3 h each
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 75000 --repeats 1000015:1000017 --n-jobs 4 --machine mac1  # 2 fits, ~3.2 h each; total ~35 h
