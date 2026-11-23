#!/usr/bin/env bash
# fig08 v1: top up converged MCMC fits to at least 600 per sample size (n = 5000 and 10000 already have 624 and 686)
# n = 1000 had 521/600 converged (88% rate on mac1): 100 new repeats, MLE and MCMC (no MLE exists for 100:200)
# n = 3000 had 570/594 converged (95% rate on mac1): 40 new repeats (MLE already in this folder for 855:895)
# default --n-jobs (all CPUs) runs two 4-chain MCMC fits at a time, as in the earlier mac1 v1 runs

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 100:200 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-v1-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 3000 --repeats 855:895 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-v1-mac1 --machine mac1
