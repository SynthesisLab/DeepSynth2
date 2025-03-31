#!/bin/bash
ENV="$1"
filename=$(basename -- "$ENV")
NAME="${filename%.*}"
DFTA_SUFFIX="_dfta_filter_control"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

function gen_dfta(){
    dst="automatic_dfta_filter_control_$1.py"
    json="equivalent_classes_control_$1.json"
    if [ ! -f "$json" ]; then
        python examples/rl/dsl_equation_generator.py -s $1 --n 1000 > /dev/null
        mv equivalent_classes_control.json $json
    fi
    if [ ! -f "$dst" ]; then
        python examples/rl/equivalence_classes_to_filter.py $json > /dev/null
        mv dfta_filter_control.py $dst
    fi
}

function run_exp(){
    seed=$1
    dfta=$2
    filename="$NAME/${dfta}_$seed.csv"
    if [ ! -f "$filename" ]; then
        arg="--automaton 'control.txt'"
        # arg="--filter ${dfta}$DFTA_SUFFIX"
        if [[ -z "$dfta" ]]; then
            arg=""
        fi
        echo "python examples/rl/solve.py --seed $seed -o $filename $arg @$ENV"
    fi
}

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

for SEED in $SEEDS
do
    # gen_dfta $SEED
    run_exp $SEED "automatic"
    run_exp $SEED
done
