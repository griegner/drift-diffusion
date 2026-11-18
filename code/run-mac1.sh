#!/usr/bin/env bash
# runtime machine: n=10000, two 4-core fits at a time (as in v1); mac2 takes 0:500, win 500:900

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 900:1300 --n-jobs 8 --tune 1000 \
    --timeout-per-1000 inf --output data/fig08-v1-mac1 --machine mac1
