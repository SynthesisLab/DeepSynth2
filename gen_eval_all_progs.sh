#!/bin/bash
ENV="$1"
SEEDS="1"
filename=$(basename -- "$ENV")
NAME="${filename%.*}"
BASIC="./control_basic15.grape"
FILTER="./inf_control9.grape"
SIZE=10
if [[ $NAME == "lunar_lander" || $NAME == "acrobot" ]]; then
    SIZE=9
fi

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

for SEED in $SEEDS
do
    basic_progs="./$NAME/progs_${NAME}_basic_seed$SEED.txt"
    filter_progs="./$NAME/progs_${NAME}_filter_seed$SEED.txt"
    echo "python examples/rl/gen_programs.py @$ENV --size $SIZE --automaton $BASIC --seed $SEED -o $basic_progs"
    echo "python examples/rl/gen_programs.py @$ENV --size $SIZE --automaton $FILTER --seed $SEED -o $filter_progs"
    echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $basic_progs -o ./$NAME/progs_eval_${NAME}_seed${SEED}_basic.csv"
    echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $filter_progs -o ./$NAME/progs_eval_${NAME}_seed${SEED}_filter.csv --warm ./$NAME/progs_eval_${NAME}_seed${SEED}_basic.csv"
done