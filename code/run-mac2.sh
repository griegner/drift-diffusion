#!/usr/bin/env bash

set -e

# more fixed-t 7p MCMC references, 50k trials (~11 GB each, one at a time; 25k took 0.75 h here)
pixi run -e code-ocean python code/fig08-reference-tfix.py --n-trials 50000 --repeats 1000026:1000042 --n-jobs 4 \
    --machine mac2  # 16 fits, ~1.5 h each
