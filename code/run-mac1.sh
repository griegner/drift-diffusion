#!/usr/bin/env bash

set -e

pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 600:700 --tune 1000 --output data/fig08-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 600:700 --tune 1000 --output data/fig08-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 700:800 --tune 1000 --output data/fig08-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 700:800 --tune 1000 --output data/fig08-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 1000 --repeats 800:900 --tune 1000 --output data/fig08-mac1 --machine mac1
pixi run -e code-ocean python code/fig08-run.py --n-trials 5000 --repeats 800:900 --tune 1000 --output data/fig08-mac1 --machine mac1
