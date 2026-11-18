#!/usr/bin/env bash
# mac1 overnight: pseudo-true references averaged over 5 x 50,000-trial fits (fig04, fig08), then resume n=10000

set -e

# fig08: reference reps 1000000:1000005 per method; finished reps are skipped
# --n-jobs 4 runs one MCMC fit at a time (each about 4 h and 11 GB peak memory)
pixi run -e code-ocean python code/fig08-run.py --pseudo-true --n-jobs 4 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-reference --machine mac1

bash code/run-mac1.sh
