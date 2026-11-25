#!/usr/bin/env bash

set -e

# catch-up to ~1000 converged MCMC per n; MCMC only on repeats whose MLE already exists
pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 254:358 --methods mcmc --tune 1000 \
    --timeout-per-1000 inf --output data/fig08-v1-mac2 --machine mac2  # 100 fits (255:259 done)
