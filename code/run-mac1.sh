#!/usr/bin/env bash

set -e

# free-t 7p MCMC pseudo-true references, 50k trials (~11 GB each, one at a time; resumable)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --methods mcmc --n-trials 50000 --repeats 1000010:1000018 \
    --n-jobs 4 --tune 1000 --timeout-per-1000 inf --output data/fig08-reference --machine mac1  # 8 fits, ~3.5 h each
