#!/usr/bin/env bash
# runtime machine: n=10000; resumes v1 (MCMC 0:127 minus rep 115 done), win takes 500:900, mac1 900:1300

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 0:500 --tune 1000 --timeout-per-1000 inf \
    --output data/fig08-v1-mac2 --machine mac2
