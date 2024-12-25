#!/bin/bash
NAME="karel"
DFTAS="automatic"
DFTA_SUFFIX="_dfta_filter_karel.py"
TASKS=100

function gen_dfta(){
    python examples/pbe/karel/equivalence_classes_to_filter.py equivalent_classes_karel.json -c -v > /dev/null
    mv dfta_filter_karel.py automatic_dfta_filter_karel.py
}
function gen_dataset(){
    file="./$NAME/dataset.pickle"
    if [ ! -f "$file" ]; then
        echo "python examples/pbe/karel/karel_task_generator.py -o $file -s $1 -w 10 --height 10 -g 4 --size $TASKS --max-depth 10 --min-operations 9 --uniform --filter automatic_dfta_filter_karel.py"

    fi
}

function run_exp(){
    seed=$1
    dfta=$2
    filename="$NAME/dataset_seed_${seed}_beap_search_uniform_${dfta}.csv"
    if [ ! -f "$filename" ]; then
        arg="--filter ${dfta}$DFTA_SUFFIX"
        if [[ -z "$dfta" ]]; then
            arg=""
        fi
        echo "python examples/pbe/solve.py --dsl karel -d karel/dataset.pickle --solver cutoff --search beap_search -t 60 -o karel/ $arg"
        echo "mv karel/dataset_beap_search_uniform_cutoff.csv $filename"
    fi
}

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

gen_dfta
gen_dataset 1
for DFTA in $DFTAS
do
    run_exp 1 $DFTA
done
run_exp 1
