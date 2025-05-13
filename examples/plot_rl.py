from glob import glob
import os
import matplotlib.pyplot as plt
import pltpublish as pub
import tqdm

import numpy as np

from examples.plot_helper import plot_y_wrt_x


def load_eval_file(file: str, rewards: dict):
    with open(file) as fd:
        line = fd.readline()
        line = fd.readline()
        while line:
            elems = line.split(",")
            prog = elems.pop(0).replace(".0", "")
            mean_reward = sum(map(float, elems)) / len(elems)
            if prog in rewards:
                assert np.isclose(rewards[prog], mean_reward)
            else:
                rewards[prog] = mean_reward
            line = fd.readline()


def load_evals(folder: str, env: str):
    eval_file = f"progs_eval_{env}_seed1_basic.csv"
    rewards = {}
    load_eval_file(os.path.join(folder, eval_file), rewards)
    eval_file = f"progs_eval_{env}_seed1_basic_with_csts.csv"
    load_eval_file(os.path.join(folder, eval_file), rewards)
    return rewards


def plot(folder: str):
    env = os.path.basename(folder[:-1] if folder[-1] == "/" else folder)
    # Load evals
    rewards = load_evals(folder, env)
    print("loaded evals of:", len(rewards), "programs")

    # Map traces to rewards
    methods = {}
    missing = []
    for file in tqdm.tqdm(glob(f"{folder}/*.csv")):
        basename = os.path.basename(file)
        # if not basename.startswith(prefix):
        #     continue
        # basename = basename[len(prefix):]
        if not (basename.startswith("_") or basename.startswith("automatic_")):
            continue

        values = []
        with open(file) as fd:
            lines = fd.readlines()
            lines.pop(0)
            for lineno, line in enumerate(lines):
                elems = line.split(",")
                time = float(elems.pop(0))
                program = elems.pop().strip(" \n")
                if program not in rewards:
                    print("FATAL ERROR in:", file, "with:", program, "line n°:", lineno)
                    missing.append(program)
                    # break
                    continue
                values.append((time, rewards[program]))
        if basename.startswith("_"):
            i = len("_")
            seed = int(basename[i : basename.find("_", i)])
            name = "basic"
        elif basename.startswith("automatic_"):
            i = len("automatic_")
            seed = int(basename[i : basename.find("_", i)])
            name = "filter"
        else:
            assert False
        name += basename[basename.find("_", i) : -len(".csv")]
        name = name.replace("none", "").replace("__", "_").strip("_").replace("_", " ")
        if len(values) == 0:
            continue
        if name not in methods:
            methods[name] = {}
        methods[name][seed] = values
    plot_y_wrt_x(
        plt.gca(), methods, (0, "Time (in s)"), (1, "Mean Reward"), cumulative=False
    )
    plt.show()
    with open(f"./missing_{env}.txt", "w") as fd:
        fd.write("\n".join(set(missing)))


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
