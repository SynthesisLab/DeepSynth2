#!/bin/bash
ENV="$1"
SEEDS="1"
filename=$(basename -- "$ENV")
NAME="${filename%.*}"
BASIC="./grammars/basic.grape"
BASIC_CST="./grammars/basic_with_csts.grape"
FILTER_CST="./grammars/filter_with_csts.grape"
FILTER="./grammars/filter.grape"
SIZE=10
if [[ $NAME == "lunar_lander" ]]; then
    SIZE=9
elif [[ $NAME == "acrobot" ]]; then
    SIZE=8
fi

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

for SEED in $SEEDS
do
    basic_progs="./$NAME/progs_${NAME}_basic_seed$SEED.txt"
    basic_progs_cst="./$NAME/progs_${NAME}_basic_with_csts_seed$SEED.txt"
    filter_progs="./$NAME/progs_${NAME}_filter_seed$SEED.txt"
    missing="./$NAME/missing.txt"
    echo "python examples/rl/gen_programs.py @$ENV --size $SIZE --automaton $BASIC --seed $SEED -o $basic_progs"
    echo "python examples/rl/gen_programs.py @$ENV --size $SIZE --automaton $BASIC_CST --seed $SEED -o $basic_progs_cst"
    echo "python examples/rl/gen_programs.py @$ENV --size $SIZE --automaton $FILTER --seed $SEED -o $filter_progs"
    echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $basic_progs -o ./$NAME/progs_eval_${NAME}_seed${SEED}_basic.csv"
    echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $basic_progs_cst -o ./$NAME/progs_eval_${NAME}_seed${SEED}_basic_with_csts.csv --warm ./$NAME/progs_eval_${NAME}_seed${SEED}_basic.csv"
    echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $filter_progs -o ./$NAME/progs_eval_${NAME}_seed${SEED}_filter.csv --warm ./$NAME/progs_eval_${NAME}_seed${SEED}_basic.csv"
    if [ -f "$missing" ]; then
        echo "python examples/rl/eval_all.py @$ENV --seed $SEED -f $missing -o ./$NAME/progs_eval_${NAME}_seed${SEED}_missing.csv --warm ./$NAME/progs_eval_${NAME}_seed${SEED}_basic_with_csts.csv"
    fi
done