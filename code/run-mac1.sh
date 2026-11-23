#!/usr/bin/env bash

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 1100:1530 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-v1-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 3000 --repeats 1100:1500 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-v1-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 1280:1400 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-v1-mac1 --machine mac1
