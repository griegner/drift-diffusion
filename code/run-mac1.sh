#!/usr/bin/env bash

set -e

# catch-up to ~1000 converged MCMC per n; MCMC only on repeats whose MLE already exists (some in fig08-v1-win)
pixi run -e code-ocean python code/fig08-run.py --n-trials 3000 --repeats 900:1260 --methods mcmc --tune 1000 \
    --timeout-per-1000 inf --output data/fig08-v1-mac1 --machine mac1  # 255 fits (1100:1205 done)
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 800:1000 --methods mcmc --tune 1000 \
    --timeout-per-1000 inf --output data/fig08-v1-mac1 --machine mac1  # 107 fits (900:993 done)
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 364:462 --methods mcmc --tune 1000 \
    --timeout-per-1000 inf --output data/fig08-v1-mac1 --machine mac1  # 98 fits (MLE in fig08-v1-win)
