#!/usr/bin/env bash

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 0:100 --tune 1000 --output data/fig08-mac2 --machine mac2
pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 100:200 --tune 1000 --output data/fig08-mac2 --machine mac2
pixi run -e code-ocean python code/fig08-run.py --n-trials 10000 --repeats 200:300 --tune 1000 --output data/fig08-mac2 --machine mac2
