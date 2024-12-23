from collections import OrderedDict, defaultdict
from glob import glob
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import pltpublish as pub
import csv

from plot_helper import (
    plot_y_wrt_x,
    make_plot_wrapper,
)


__RENAME = {
    "": "basic",
    "automatic": "filter",
}

__DATA__ = {
    "programs": (0, "Programs Enumerated"),
    "skipped": (1, "Programs Skipped"),
    "time": (2, "Time (in s)"),
    "reward": (3, "Mean Reward"),
    "min-rew": (4, "Min Reward"),
    "maw-rew": (5, "Max Reward"),
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
    return methods


# Generate all possible combinations
__PLOTS__ = {}
for ydata in list(__DATA__.keys()):
    for xdata in list(__DATA__.keys()):
        if xdata == ydata:
            continue
        __PLOTS__[f"{ydata}_wrt_{xdata}"] = make_plot_wrapper(
            plot_y_wrt_x,
            __DATA__[xdata],
            __DATA__[ydata],
            cumulative=False,
            logy=xdata == "non_terminals",
        )

if __name__ == "__main__":
    import argparse
    import sys

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
    parser.add_argument("plots", nargs="+", choices=list(__PLOTS__.keys()))
    parameters = parser.parse_args()
    output_file: str = parameters.folder
    verbose: bool = parameters.verbose
    plots: List[str] = parameters.plots

    # Load data
    pub.setup()
    methods = load_data(output_file, verbose)
    # Check we have at least one file
    if len(methods) == 0:
        print("Error: no performance file was found!", file=sys.stderr)
        sys.exit(1)
    # Order by name so that it is always the same color for the same methods if diff. DSL
    ordered_methods = OrderedDict()
    for met in sorted(methods.keys()):
        ordered_methods[met] = methods[met]
    # Plotting
    for count, to_plot in enumerate(plots):
        ax = plt.subplot(1, len(plots), count + 1)
        __PLOTS__[to_plot](ax, ordered_methods)
    plt.show()
