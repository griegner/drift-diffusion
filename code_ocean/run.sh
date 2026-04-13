#!/usr/bin/env zsh
set -e

./code_ocean/code/fig01.py --n-samples 10

./code_ocean/code/fig02to04.py --setting constant --n-samples 10 --n-repeats 8
./code_ocean/code/fig02to04.py --setting coherence --n-samples 10 --n-repeats 8
./code_ocean/code/fig02to04.py --setting iid --n-samples 10 --n-repeats 8
