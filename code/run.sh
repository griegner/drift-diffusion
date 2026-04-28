#!/usr/bin/env zsh
set -e
start_seconds=$SECONDS

mode="${1:-paper}"

if [[ "$mode" == "test" ]]; then
    echo "\nrunning TEST mode ...\n"

    # fig01
    echo "-> fig01"
    ./fig01.py

    # fig02-04
    echo "-> fig02"
    ./fig02to04.py --setting constant --n-samples 100 --n-repeats 20
    echo "-> fig03"
    ./fig02to04.py --setting coherence --n-samples 100 --n-repeats 20
    echo "-> fig04"
    ./fig02to04.py --setting iid --n-samples 100 --n-repeats 20

    # fig05-06
    echo "-> fig05-06"
    ./fig05to06.py --subset

    # fig 06
    echo "-> fig06"
    ./fig06.py --subset

    # fig 07
    echo "-> fig07"
    ./fig07.py --n-samples 200 500 --n-repeats 1

elif [[ "$mode" == "paper" ]]; then
    echo "\nrunning PAPER mode ...\n"

    # fig01
    echo "-> fig01"
    ./fig01.py

    # fig02-04
    echo "-> fig02"
    ./fig02to04.py --setting constant
    echo "-> fig03"
    ./fig02to04.py --setting coherence
    echo "-> fig04"
    ./fig02to04.py --setting iid

    # fig05-06
    echo "-> fig05-06"
    ./fig05to06.py

    # fig 06
    echo "-> fig06"
    ./fig06.py

    # fig 07
    echo "-> fig07"
    ./fig07.py

else
    echo "Usage: ./run.sh [test|paper]"
    exit 1
fi

echo "Total runtime: $((SECONDS - start_seconds))s"
