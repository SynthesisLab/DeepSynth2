from glob import glob
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import pltpublish as pub
import csv
import tqdm


from multiprocessing import Pool


import numpy as np


def preprocess(file: str):
    basename = os.path.basename(file)
    preprocessed_file = file.replace(basename, f"preprocessed_{basename}")
    if not os.path.exists(preprocessed_file):
        values = []
        last_program = None
        last_time = -1
        count = 0
        with open(file) as fd:
            lines = fd.readlines()
            lines.pop(0)
            for line in lines:
                elems = line.split(",")
                time = float(elems.pop(0))
                program = elems.pop().strip(" \n")
                if program != last_program:
                    if count > 1:
                        values.append((last_time, last_program))
                    values.append((time, program))
                    last_program = program
                    last_time = time
                count += 1
        with open(preprocessed_file, "w") as fd:
            fd.writelines([f"{time},{program}\n" for time, program in values])
        print("done preprocessing")

def load_evals(folder: str, env: str):
    eval_file = f"progs_eval_{env}_seed1_basic.csv"
    rewards = {}
    with open(os.path.join(folder, eval_file)) as fd:
        line = fd.readline()
        line = fd.readline()
        while line:
            elems = line.split(",")
            prog = elems.pop(0)
            mean_reward = sum(map(float, elems)) / len(elems)
            rewards[prog] = mean_reward
            line = fd.readline()

    eval_file = f"progs_eval_{env}_seed1_filter.csv"
    with open(os.path.join(folder, eval_file)) as fd:
        line = fd.readline()
        line = fd.readline()
        while line:
            elems = line.split(",")
            prog = elems.pop(0)
            mean_reward = sum(map(float, elems)) / len(elems)
            if prog in rewards:
                assert np.isclose(rewards[prog], mean_reward)
            else:
                assert False, "program is possible in filtered but not in basic!"
            line = fd.readline()
    return rewards

def plot(folder: str):
    env = os.path.basename(folder[:-1] if folder[-1] == "/" else folder)
    # Load evals
    rewards = load_evals(folder, env)
    print("loaded evals of:", len(rewards), "programs")
    # Preprocessing
    all_to_preprocess = []
    for file in glob(f"{folder}/*.csv"):
        basename = os.path.basename(file)
        if not (basename.startswith("_") or basename.startswith("automatic")):
            continue
        all_to_preprocess.append(file)
    print(f"{len(all_to_preprocess)} file(s) to preprocess")
    with Pool() as p:
        p.map_async(preprocess, all_to_preprocess).wait()


    # Map traces to rewards
    methods = {"basic": [], "filter": []}
    for file in tqdm.tqdm(glob(f"{folder}/*.csv")):
        basename = os.path.basename(file)
        if not basename.startswith("preprocessed"):
            continue
        values = []
        with open(file) as fd:
            lines = fd.readlines()
            for line in lines:
                elems = line.split(",")
                time = float(elems.pop(0))
                program = elems.pop().strip(" \n")
                if program not in rewards:
                    print("FATAL ERROR in:", file, "with:", program)
                    break
                values.append((time, rewards[program]))
        if basename.startswith("preprocessed__"):
            methods["basic"].append(values)
        elif basename.startswith("preprocessed_automatic"):
            methods["filter"].append(values)

            
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument(
        "folder",
        type=str,
        help="data folder to load",
    )
    parameters = parser.parse_args()

    # Load data
    pub.setup()
    methods = plot(parameters.folder)
