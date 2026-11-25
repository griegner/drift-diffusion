#!/usr/bin/env bash

set -e

git merge-base --is-ancestor a8dd3c9 HEAD || { echo "pull the pdf change (a8dd3c9) first" >&2; exit 1; }

for n in 1000 3000 5000 10000; do
    pixi run -e code-ocean python code/fig08-run.py --n-trials "$n" --repeats 0:500 --methods mle --n-jobs 3 \
        --tune 1000 --timeout-per-1000 inf --output data/fig08-v1-runtime-mac1 --machine mac1
done
