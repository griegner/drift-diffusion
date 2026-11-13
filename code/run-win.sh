#!/usr/bin/env bash
# fits machine: n=5000

set -e

# Application Control blocks PyTensor's generated DLLs on this Windows machine.
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}cxx=,linker=py"

pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 0:600 --tune 1000 --timeout-per-1000 2000 \
    --output data/fig08-win --machine win
