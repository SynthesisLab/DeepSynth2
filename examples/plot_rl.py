from glob import glob
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import pltpublish as pub
import csv

import numpy as np

__RENAME = {
    "": "no filter",
    "automatic": "filter",
}


def load_data(output_folder: str, verbose: bool = False) -> Dict[str, Dict[int, List]]:
    # Dict[name, data]
    methods = {}

    for file in glob(os.path.join(output_folder, "*.csv")):
        filename = os.path.relpath(file, output_folder)
        name = filename[:-4]
        seed = int(name[name.rindex("_") + 1 :])
        method = name[: name.rindex("_")]
        method = __RENAME.get(method, method)

        trace = []
        with open(file, "r") as fd:
            reader = csv.reader(fd)
            trace = [tuple(row) for row in reader]
            # Pop columns names
            columns = {name: ind for ind, name in enumerate(trace.pop(0))}
            indices = [
                columns["programs"],
                columns["removed"],
                columns["time"],
                columns["score-best-mean"],
                columns["score-best-min"],
                columns["score-best-max"],
            ]
            data = [
                tuple(float(row[k]) if k >= 0 else 0 for k in indices) for row in trace
            ]
            if len(data) == 0:
                if verbose:
                    print(f"filename:{output_file} is empty!")
                return {}
            if method not in methods:
                methods[method] = {}
            if seed not in methods[method]:
                methods[method][seed] = []
            for row in data:
                methods[method][seed].append(row)
    all_seeds = set(methods[__RENAME.get("")].keys()).union(
        set(methods[__RENAME.get("automatic")].keys())
    )

    filter_points = [[], []]
    basic_points = [[], []]
    new_method = {"cmp": {}}
    for seed in all_seeds:
        basic = methods[__RENAME.get("")][seed]
        filter = methods[__RENAME.get("automatic")][seed]

        # Rescale time in [0;1]
        max_time = max(max(x[2] for x in basic), max(x[2] for x in filter))
        basic = [(x[0], x[1], x[2] / max_time, x[3]) for x in basic]
        filter = [(x[0], x[1], x[2] / max_time, x[3]) for x in filter]

        # Interpolate Reward w.r.t. time
        # min_time = min(max(x[2] for x in basic), max(x[2] for x in filter))
        min_time = max_time
        n_points = min(100, max(len(basic), len(filter)))
        step = (min_time - 0) / n_points
        time_range = np.arange(0, min_time + step / 2, step=step) / min_time

        basic_data = np.interp(time_range, [x[2] for x in basic], [x[3] for x in basic])
        filter_data = np.interp(
            time_range, [x[2] for x in filter], [x[3] for x in filter]
        )

        # Compare Reward at same time
        cmp_data = np.cumsum(filter_data >= basic_data) / (np.arange(n_points + 1) + 1)
        new_method["cmp"][seed] = [(x, y) for x, y in zip(time_range, cmp_data)]

        # Produce scatter points
        # X = % better reward
        # Y = % time saved at max reward
        min_reward = min(basic[-1][3], filter[-1][3])
        time_saved = 0
        filter_better = 0
        equals = 0
        reached = [False, False]
        reach_time = [1, 1]
        for a, b in zip(basic_data, filter_data):
            time_saved += 1
            filter_better += b > a
            equals += a == b
            if not reached[0] and a >= min_reward:
                reached[0] = True
                reach_time[0] = time_saved / n_points
            if not reached[1] and b >= min_reward:
                reached[1] = True
                reach_time[1] = time_saved / n_points

        reach_time = np.array(reach_time) / max(reach_time)

        filter_points[0].append(filter_better / n_points)
        basic_points[0].append((n_points - filter_better - equals) / n_points)
        basic_points[1].append(reach_time[0])
        filter_points[1].append(reach_time[1])

    plt.subplot(1, 2, 1)
    from plot_helper import plot_y_wrt_x

    plot_y_wrt_x(
        plt.gca(),
        new_method,
        (0, "% of Time"),
        (1, r"% Filter $\geq$ Basic"),
        ylim=(0, 1.025),
        xlim=(0, 1),
        cumulative=False,
    )
    plt.hlines(0.5, 0, 1, linestyles="dashed", colors="red")
    plt.gca().get_legend().set_visible(False)
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.scatter(
        basic_points[0], basic_points[1], label=__RENAME.get(""), alpha=0.7, c="black"
    )
    plt.scatter(
        filter_points[0], filter_points[1], label=__RENAME.get("automatic"), alpha=0.7
    )
    plt.legend()
    plt.xlabel("% Better Reward")
    plt.ylabel("% of Time Used")
    plt.xlim(left=max(plt.xlim()[0], 0), right=min(plt.xlim()[1], 1.025))
    plt.ylim(bottom=max(plt.ylim()[0], 0), top=min(plt.ylim()[1], 1.025))
    plt.show()
    return methods


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument(
        "folder",
        type=str,
        help="data folder to load",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose mode",
    )
    parameters = parser.parse_args()
    output_file: str = parameters.folder
    verbose: bool = parameters.verbose

    # Load data
    pub.setup()
    methods = load_data(output_file, verbose)
