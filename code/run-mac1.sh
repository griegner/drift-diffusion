#!/usr/bin/env bash
# runtime machine: n=1000 and n=5000 side by side (one fit each); n=1000 cores move to n=5000 when done

set -e

run() {
    pixi run -e code-ocean python code/fig08-run.py "$@" --n-jobs 4 --tune 1000 --timeout-per-1000 inf \
        --output data/fig08-mac1 --machine mac1
}

(run --n-trials 1000 --repeats 600:1100 && run --n-trials 5000 --repeats 900:1000) &
run --n-trials 5000 --repeats 600:900 &
wait
