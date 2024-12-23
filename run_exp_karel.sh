#!/bin/bash
NAME="karel"
DFTAS="automatic"
DFTA_SUFFIX="_dfta_filter_karel.py"
TASKS=500


function abort_on_failure(){
    out=$?
    if [ $out != 0 ]; then
        echo "An error has occured"
        exit 1
    fi
}

function gen_dfta(){
    python examples/pbe/karel/equivalence_classes_to_filter.py equivalent_classes_karel.json
    mv dfta_filter_karel.py automatic_dfta_filter_karel.py
}

function gen_dataset(){
    file="./$NAME/dataset.pickle"
    if [ ! -f "$file" ]; then
        python examples/pbe/karel/karel_task_generator.py -o $file -s $1 -w 10 --height 10 -g 3 --size $TASKS --max-operations 5 --uniform

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
        python examples/pbe/solve.py --dsl karel -d karel/dataset.pickle --solver cutoff -o karel/ $arg
        mv karel/dataset_cd_search_uniform_cutoff.csv $filename
    fi
}

if [ ! -d "./$NAME" ]; then
    mkdir "./$NAME"
fi

gen_dfta
gen_dataset 1
for DFTA in $DFTAS
do
    run_exp 1 $DFTA &
done
run_exp 1 &
wait
