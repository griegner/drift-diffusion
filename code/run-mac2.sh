#!/usr/bin/env bash

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 1470:1640 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-v1-mac2 --machine mac2
