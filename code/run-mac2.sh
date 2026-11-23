#!/usr/bin/env bash
# fig08 v1 runtime benchmark (fig08-plot-n.py panel B, MLE): rerun MLE fits at every n on this machine only,
# four fits at a time, into a separate folder so the production records (and their runtimes) stay unchanged.
# The estimates reproduce the production fits; only the runtimes are new.

set -e

for n in 1000 3000 5000 10000; do
    pixi run -e code-ocean python code/fig08-run.py --n-trials "$n" --repeats 0:200 --methods mle --n-jobs 3 \
        --tune 1000 --timeout-per-1000 inf --output data/fig08-v1-runtime --machine mac2
done
