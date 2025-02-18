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

    rewards_per_seed = {}

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
                columns["program"],
                columns["programs"],
                columns["removed"],
                columns["time"],
                columns["score-best-mean"],
                columns["score-best-min"],
                columns["score-best-max"],
                columns["samples"],
            ]
            if seed not in rewards_per_seed:
                rewards_per_seed[seed] = {}
            for row in trace:
                prog = row[indices[0]]
                scores = list(map(float, [row[k] for k in indices[4:]]))
                if prog not in rewards_per_seed[seed]:
                    rewards_per_seed[seed][prog] = scores
                n = rewards_per_seed[seed][prog][-1]
                if n < scores[-1]:
                    rewards_per_seed[seed][prog] = scores
            data = [
                [row[0]] + list(float(row[k]) for k in indices[1:4]) for row in trace
            ]
            if len(data) == 0:
                if verbose:
                    print(f"filename:{output_file} is empty!")
                return {}
            if method not in methods:
                methods[method] = {}
            if seed not in methods[method]:
                methods[method][seed] = []
            methods[method][seed] = data
    
    
    # POST PROCESSING NOW THAT ALL FILES ARE PROCESSED
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
        max_time = max(max(x[3] for x in basic), max(x[3] for x in filter))
        basic = [(x[0], x[1], x[2], x[3] / max_time, x[4]) for x in basic]
        filter = [(x[0], x[1], x[2], x[3] / max_time, x[4]) for x in filter]

        # Interpolate Reward w.r.t. time
        # min_time = min(max(x[2] for x in basic), max(x[2] for x in filter))
        min_time = max_time
        n_points = min(100, max(len(basic), len(filter)))
        step = (min_time - 0) / n_points
        time_range = np.arange(0, min_time + step / 2, step=step) / min_time

        basic_data = np.interp(
            time_range,
            [x[2] for x in basic],
            [rewards_per_seed[seed][x[0]][0] for x in basic],
        )
        filter_data = np.interp(
            time_range,
            [x[2] for x in filter],
            [rewards_per_seed[seed][x[0]][0] for x in filter],
        )

        # Compare Reward at same time
        cmp_data = np.cumsum(filter_data >= basic_data) / (np.arange(n_points + 1) + 1)
        new_method["cmp"][seed] = [
            (x * 100, y * 100) for x, y in zip(time_range, cmp_data)
        ]

        # Produce scatter points
        # X = % better reward
        # Y = % time saved at max reward
        min_reward = min(basic[-1][3], filter[-1][3])
        time_saved = 0
        filter_better = 0
        basic_better = 0
        equals = 0
        reached = [False, False]
        reach_time = [1, 1]
        for a, b in zip(basic_data, filter_data):
            time_saved += 1
            filter_better += (b / a) > 1.01
            basic_better += (a / b) > 1.01
            equals += abs(b - a) / abs(a) <= 0.01
            if not reached[0] and a >= min_reward:
                reached[0] = True
                reach_time[0] = time_saved / n_points
            if not reached[1] and b >= min_reward:
                reached[1] = True
                reach_time[1] = time_saved / n_points

        reach_time = (np.array(reach_time) / max(reach_time)) * 100

        filter_points[0].append(rewards_per_seed[seed][filter[-1][0]])
        basic_points[0].append(rewards_per_seed[seed][basic[-1][0]])
        basic_points[1].append(reach_time[0])
        filter_points[1].append(reach_time[1])

    plt.subplot(1, 2, 1)
    from plot_helper import plot_y_wrt_x

    # width = 100
    # n = 15
    # unit = (width + 50) / n
    # for i in range(n):
    #     end_point = unit * (i + 1)
    #     # y=-x
    #     plt.plot([end_point - 50, end_point], [50, 0], c='red')
    plt.fill_between([0, 100], 0, 50, alpha=0.5, color="red")
    plt.hlines(50, 0, 100, linestyles="dashed", colors="red")
    plot_y_wrt_x(
        plt.gca(),
        new_method,
        (0, "% of Time"),
        (1, r"% Filter $\geq$ Basic"),
        ylim=(0, 102.5),
        xlim=(0, 100),
        cumulative=False,
    )
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
    plt.xlabel("Average Reward")
    plt.ylabel("% of Time Used to Reach Worst Method's Reward")
    # plt.xlim(left=max(plt.xlim()[0], 0), right=min(plt.xlim()[1], 1.025))
    plt.ylim(bottom=max(plt.ylim()[0], 0), top=min(plt.ylim()[1], 102.5))
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
