#!/usr/bin/env bash

set -e

# more free-t 7p MCMC pseudo-true references at 100k (resumable); can start alongside the previous 100k run, whose
# reps differ. Samples are read from data/pseudotrue-samples (~2.5 GB per fit); pull those first
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --tune 1000 --timeout-per-1000 inf --output data/fig08-reference \
    --n-trials 100000 --repeats 1000014:1000018 --n-jobs 4 --machine mac1  # 4 fits, one at a time, ~4.3-5 h each
