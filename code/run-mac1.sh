#!/usr/bin/env bash
# runtime machine: n=10000 in two streams (one fit each); mac2 takes 0:500, win 500:900

set -e

run() {
    pixi run -e code-ocean python code/fig08-run.py "$@" --n-trials 10000 --n-jobs 4 --tune 1000 --timeout-per-1000 inf \
        --output data/fig08-v1-mac1 --machine mac1
}

run --repeats 900:1100 &
run --repeats 1100:1300 &
wait
