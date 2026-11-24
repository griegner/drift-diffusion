#!/usr/bin/env bash
# fig08 v1 runtime benchmark (fig08-plot.py panel B, MLE) with the current pdf (per-series term counts, a8dd3c9):
# rerun MLE fits at every n on this machine only, three at a time, into a separate folder so the production records
# (and their runtimes) stay unchanged. Run after mac1's production fits finish, with nothing else running.
# Plot with: fig08-plot.py --runtime-inputs data/fig08-v1-runtime-mac1

set -e

git merge-base --is-ancestor a8dd3c9 HEAD || { echo "pull the pdf change (a8dd3c9) first" >&2; exit 1; }

for n in 1000 3000 5000 10000; do
    pixi run -e code-ocean python code/fig08-run.py --n-trials "$n" --repeats 0:200 --methods mle --n-jobs 3 \
        --tune 1000 --timeout-per-1000 inf --output data/fig08-v1-runtime-mac1 --machine mac1
done
