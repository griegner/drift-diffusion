#!/usr/bin/env bash

set -e

# fixed-t 7p MCMC references, 25k trials (~6 GB each, one at a time)
pixi run -e code-ocean python code/fig08-reference-tfix.py --n-trials 25000 --repeats 1000005:1000010 --n-jobs 4 \
    --machine mac2  # 5 fits, ~15 h
