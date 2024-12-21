#!/bin/bash
python examples/rl/dsl_equation_generator.py -s 1 --n 500
python examples/rl/equivalence_classes_to_filter.py equivalent_classes_control.json -c -v
mv dfta_filter_control.py automatic_dfta_filter_control.py