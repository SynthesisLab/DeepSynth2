#!/bin/bash
ENV="$1"
filename=$(basename -- "$ENV")
NAME="${filename%.*}"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
SOLVERS="sampling none"
OPTI="mab none"

function run_exp(){
    seed=$1
    dfta=$2
    solver=$3
    opti=$4
    filename="$NAME/${dfta}_$seed_${solver}_${opti}.csv"
    # echo "file:$filename"
    if [ ! -f "$filename" ]; then
        arg="--automaton 'control.grape'"
        if [[ -z $dfta ]]; then
            arg=""
        fi
        if [[ ! $solver == "none" ]]; then
            arg="$arg --$solver"
        fi
        if [[ ! $opti == "none" ]]; then
            arg="$arg --$opti"
        fi
        echo "python examples/rl/solve.py --seed $seed -o $filename $arg @$ENV"
    fi
}

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

for SEED in $SEEDS
do
    for solver in $SOLVERS; do
        for opti in $OPTI; do
            run_exp $SEED "automatic" $solver $opti
            run_exp $SEED "" $solver $opti
        done
    done
done
