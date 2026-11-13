#!/usr/bin/env bash
# runtime machine: n=10000

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 0:150 --tune 1000 --timeout-per-1000 3600 \
    --output data/fig08-mac2 --machine mac2
