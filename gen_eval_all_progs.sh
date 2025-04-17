#!/bin/bash
ENV="$1"
SEEDS="1"
filename=$(basename -- "$ENV")
NAME="${filename%.*}"
BASIC="./control_basic15.grape"
FILTER="./inf_control9.grape"
SIZE=10

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

for SEED in $SEEDS
do
    basic_progs="${NAME}_basic_seed$seed.txt"
    filter_progs="${NAME}_filter_seed$seed.txt"
    echo "python examples/rl/gen_programs.py @$ENV --size $SIZE --automaton $BASIC --seed $seed -o $basic_progs"
    echo "python examples/rl/gen_programs.py @$ENV --size $SIZE --automaton $FILTER --seed $seed -o $filter_progs"
    echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $basic_progs -o c${NAME}_seed${seed}_basic.csv"
    echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $filter_progs -o c${NAME}_seed${seed}_filter.csv"
done