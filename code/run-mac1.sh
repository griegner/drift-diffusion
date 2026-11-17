#!/usr/bin/env bash
# runtime machine: n=1000 in two streams (one fit each)

set -e

run() {
    pixi run -e code-ocean python code/fig08-run.py "$@" --n-trials 1000 --n-jobs 4 --tune 1000 --draws 1000 \
        --timeout-per-1000 inf --output data/fig08-v2-mac1 --machine mac1
}

run --repeats 0:250 &
run --repeats 250:500 &
wait
