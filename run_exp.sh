#!/bin/bash
NAME="$1"
ENV="$2"
DFTAS="automatic manual"
DFTA_SUFFIX="_dfta_filter_control.py"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

function abort_on_failure(){
    out=$?
    if [ $out != 0 ]; then
        echo "An error has occured"
        exit 1
    fi
}

function run_exp(){
    seed=$1
    dfta=$2
    filename="$NAME/${dfta}_$seed.csv"
    if [ ! -f "$filename" ]; then
        arg="--filter ${dfta}$DFTA_SUFFIX"
        if [[ -z "$dfta" ]]; then
            arg=""
        fi
        python examples/rl/solve.py --seed $seed -o $filename $arg @$ENV
    fi
}

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

for SEED in $SEEDS
do
    run_exp $SEED
    for DFTA in $DFTAS
    do
        run_exp $SEED $DFTA
    done
done
