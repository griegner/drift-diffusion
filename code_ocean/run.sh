#!/usr/bin/env zsh
set -e

# fig01
echo "-> fig01"
./code_ocean/code/fig01.py --n-samples 100

# fig02-04
echo "-> fig02"
./code_ocean/code/fig02to04.py --setting constant --n-samples 100 --n-repeats 20
echo "-> fig03"
./code_ocean/code/fig02to04.py --setting coherence --n-samples 100 --n-repeats 20
echo "-> fig04"
./code_ocean/code/fig02to04.py --setting iid --n-samples 100 --n-repeats 20
