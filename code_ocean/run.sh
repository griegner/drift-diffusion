#!/usr/bin/env zsh
set -e
start_seconds=$SECONDS

mode="${1:-paper}"

if [[ "$mode" == "test" ]]; then
    echo "\nrunning TEST mode ...\n"

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

    # fig05-06
    echo "-> fig05-06"
    ./code_ocean/code/fig05to06.py --subset

    # fig 06
    echo "-> fig06"
    ./code_ocean/code/fig06.py --subset

    # fig 07
    echo "-> fig07"
    ./code_ocean/code/fig07.py --n-sample 200 500 --n-repeats 1

elif [[ "$mode" == "paper" ]]; then
    echo "\nrunning PAPER mode ...\n"

    # fig01
    echo "-> fig01"
    ./code_ocean/code/fig01.py

    # fig02-04
    echo "-> fig02"
    ./code_ocean/code/fig02to04.py --setting constant
    echo "-> fig03"
    ./code_ocean/code/fig02to04.py --setting coherence
    echo "-> fig04"
    ./code_ocean/code/fig02to04.py --setting iid

    # fig05-06
    echo "-> fig05-06"
    ./code_ocean/code/fig05to06.py

    # fig 06
    echo "-> fig06"
    ./code_ocean/code/fig06.py

    # fig 07
    echo "-> fig07"
    ./code_ocean/code/fig07.py

else
    echo "Usage: ./code_ocean/run.sh [test|paper]"
    exit 1
fi

echo "Total runtime: $((SECONDS - start_seconds))s"
