#!/usr/bin/env bash

set -e

# PyTensor compiles C modules; keep the cache in one place
export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:+${PYTENSOR_FLAGS},}base_compiledir=C:/pytensor"


pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 0:100 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 0:100 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 100:200 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 100:200 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 200:300 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 200:300 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 300:400 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 300:400 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 400:500 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 400:500 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 500:600 --output data/fig08-win --machine win
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 500:600 --output data/fig08-win --machine win
