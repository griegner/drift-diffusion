#!/usr/bin/env bash
set -e

start_seconds=$SECONDS

mode=${1:-paper}

# download csv from dryad DOI: 10.5061/dryad.hhmgqnkxb

if [[ "$mode" == "test" ]]; then
    echo "\nrunning TEST mode ...\n"

    # fig01
    echo "-> fig01"
    python fig01.py

    # fig02-04
    echo "-> fig02"
    python fig02to04.py --setting constant --n-samples 100 --n-repeats 20
    echo "-> fig03"
    python fig02to04.py --setting time --n-samples 100 --n-repeats 20
    echo "-> fig04"
    python fig02to04.py --setting persistent --n-samples 100 --n-repeats 20

    # fig05-06
    echo "-> fig05-06"
    python fig05to06.py --subset

    # fig06
    echo "-> fig06"
    python fig06.py --subset

    # fig07
    echo "-> fig07"
    python fig07.py --n-trials 200 --tune 100 --draws 200

    # fig08
    echo "-> fig08"
    python fig08.py --n-samples 500 1000 --n-repeats 3 --n-repeats-mcmc 3 --tune 100 --draws 200

elif [[ "$mode" == "paper" ]]; then
    echo "\nrunning PAPER mode ...\n"

    # fig01
    echo "-> fig01"
    python fig01.py

    # fig02-04
    echo "-> fig02"
    python fig02to04.py --setting constant
    echo "-> fig03"
    python fig02to04.py --setting time
    echo "-> fig04"
    python fig02to04.py --setting persistent

    # fig05-06
    echo "-> fig05-06"
    python fig05to06.py

    # fig06
    echo "-> fig06"
    python fig06.py

    # fig07
    echo "-> fig07"
    python fig07.py

    # fig08
    echo "-> fig08"
    python fig08.py --csv

else
    echo "Usage: ./run.sh [test|paper]"
    exit 1
fi

echo "Total runtime: $((SECONDS - start_seconds))s"
