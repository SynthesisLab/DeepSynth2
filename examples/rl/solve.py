import atexit
import argparse
import json
import csv

from typing import Dict

import numpy as np
from synth.syntax.program import Program


from control_dsl import get_dsl
from optim.constant_optimizer import ConstantOptimizer
import stats.chronometer as chronometer
import stats.counter as counter
from bandits.topk_manager import TopkManager
from rl.rl_utils import type_for_env
from program_evaluator import ProgramEvaluator

from synth.syntax import (
    auto_type,
)
from gpoe_automaton_to_grammar import parse, size_constraint

from synth.syntax.grammars.enumeration.sbsur import enumerate_uniform_dfta

import gymnasium as gym


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

parser.add_argument("--env", type=str, help="name of the environment")
parser.add_argument(
    "-g",
    "--goal",
    type=float,
    default=500,
    help="target score after which we automatically stop (default: 500)",
)
# parser.add_argument(
#     "-d", "--derivative", type=int, default=-1, help="number of derivatives steps"
# )
parser.add_argument(
    "--env-build-arg",
    dest="env_arg",
    type=str,
    default="{}",
    help="dictionnary of arguments to pass to the env",
)
parser.add_argument("-s", "--seed", type=int, default=1, help="seed")
parser.add_argument(
    "-o", "--output", type=str, default="./search_data.csv", help="CSV file name"
)
parser.add_argument(
    "--automaton",
    type=str,
    help="path to the automaton",
)

params = parser.parse_args()
SEED: int = params.seed
# DERIVATIVE_TIMESTEP: int = params.derivative
TARGET_RETURN: float = params.goal
automaton: str = params.automaton
output_file: str = params.output
env_args: Dict = json.loads(params.env_arg)
env_name: str = params.env
env = gym.make(env_name, **env_args)

# =========================================================================
# GLOBAL PARAMETERS
# max number of episodes that should be done at most to compare two possiby equal (optimised) candidates
MAX_BUDGET: int = 50
MAX_TO_CHECK_SOLVED: int = 2 * MAX_BUDGET
BATCH_SIZE: int = 1
SIZE: int = 15

if "Pong" in env_name:
    from pong_wrapper import make_pong


def build_env():
    if "Pong" in env_name:
        env = make_pong()
    else:
        env = gym.make(env_name, **env_args)
    env.reset(seed=SEED)
    return env


env = build_env()

# if DERIVATIVE_TIMESTEP > 0:
#     env = DerivativeObsWrapper(env, DERIVATIVE_TIMESTEP)
type_request = type_for_env(build_env())
print("Requested type:", type_request)
(
    dsl,
    prog_evaluator,
) = get_dsl(
    type_request,
    env.action_space,
)
evaluator = ProgramEvaluator(build_env, prog_evaluator)


constant_types = set()
if "float" in str(type_request) and "Pong" not in env_name:
    constant_types.add(auto_type("float"))
# Filter
auto = size_constraint(dsl, type_request, SIZE)
try:
    with open(automaton) as fd:
        content = fd.read()
    G = parse(
        dsl,
        content,
        type_request,
        constant_types,
        env.action_space.n if type_request.ends_with(auto_type("action")) else 0,
    )
    dfta = auto.read_product(G)
except FileNotFoundError:
    print("No automaton found, using base")
    dfta = auto
enumerator = enumerate_uniform_dfta(dfta, BATCH_SIZE)

topk: TopkManager = TopkManager(evaluator, c=abs(TARGET_RETURN) / 2, k=2)
const_opti = ConstantOptimizer(SEED)


stats = [
    (
        "programs",
        "removed",
        "time",
        "score-best-mean",
        "score-best-min",
        "score-best-max",
        "program",
        "samples",
    )
]


def log_data():
    best_program, q_value, incertitude, mini, maxi = topk.get_best_stats()

    stats.append(
        (
            counter.get("programs.iterated").total,
            enumerator.filtered,
            chronometer.total_time(),
            q_value,
            mini,
            maxi,
            best_program,
            incertitude,
        )
    )
    with open(output_file, "w") as fd:
        csv.writer(fd).writerows(stats)


def print_search_state():
    total = counter.get("programs.iterated").total
    print()
    print(
        "[SEARCH]",
        "programs:",
        ", ".join(
            [
                f"{key}:{value.total} ({value.total * 100 / total:.1f}%)"
                for key, value in counter.items("programs")
            ]
        ),
    )
    print(
        "[SEARCH]",
        "skipped:",
        f"{enumerator.filtered} ({enumerator.filtered * 100 / total:.1f}%)",
    )
    total = max(1, counter.total("episodes"))
    print(
        "[SEARCH]",
        f"episodes (total:{total}):",
        ", ".join(
            [
                f"{key}:{value.total} ({value.total * 100 / total:.1f}%)"
                for key, value in counter.items("episodes")
            ]
        ),
    )
    total = chronometer.total_time()
    print(
        "[SEARCH]",
        "total times:",
        ", ".join(
            [
                f"{key}:{value.total:.2f}s ({value.total * 100 / total:.1f}%)"
                for key, value in chronometer.items()
            ]
        ),
    )
    print(
        "[SEARCH]",
        "mean times:",
        ", ".join(
            [
                f"{key}:{value.total * 1000 / max(1, value.count):.2f}ms"
                for key, value in chronometer.items()
            ]
        ),
    )


def print_best_program():
    if topk.num_candidates() < 1:
        print("NO PROGRAM FOUND")
        return
    best_program, q_value, samples, mini, maxi = topk.get_best_stats()
    print("[BEST PROGRAM]\t", best_program)
    # print("\tprobability=", toy_PCFG.probability_program(toy_PCFG.start, original_program))
    print(
        f"\treturns: {q_value:.2f}(n={samples}, inc={topk.uncertainty(best_program):.2f}) [{mini:.2f}; {maxi:.2f}]"
    )


def is_solved() -> bool:
    if topk.num_candidates() < 1:
        return False
    current_best_return = topk.get_best_stats()[1]
    if current_best_return >= TARGET_RETURN:
        with chronometer.clock("evaluation.confirm"):
            budget_used = topk.run_at_least(MAX_TO_CHECK_SOLVED, TARGET_RETURN)
            counter.count("episodes.confirm", budget_used)
        current_best_return = topk.get_best_stats()[1]
        if current_best_return >= TARGET_RETURN:
            print("[SUCCESSFULLY SOLVED]")
            return True
    return False


def at_exit():
    print_search_state()
    print("=" * 60)
    print_best_program()

    with open(output_file, "w") as fd:
        csv.writer(fd).writerows(stats)


atexit.register(at_exit)


def make_eval(program: Program) -> float:
    def f():
        program.refresh_hash()
        evaluator.eval(program)
        return evaluator.returns(program)[-1]

    return f


current_best_return: float = -np.inf
gen = enumerator.generator()
while True:
    with chronometer.clock("enumeration"):
        program = next(gen)
    # print("PROGRAM:", program)
    counter.count("programs.iterated", 1)
    iterated = counter.get("programs.iterated").total
    if iterated % 20 == 0 and iterated > 0:
        log_data()
        if iterated % 100 == 0 and iterated > 0:
            print_search_state()
    copy = program.clone()
    tiles = None
    returns = []
    if copy.count_constants() > 0:
        np.random.seed(SEED)

        # print("\tprogram:", copy)
        # print("Constant domains:", cst_domains)
        evaluator.record(False)
        with chronometer.clock("mab"):
            tiles, returns = const_opti.optimize(
                make_eval(copy),
                list(copy.constants()),
            )
        counter.count("programs.optimised", 1)
        counter.count("episodes.constant_opti", const_opti.budget_used)
        for constant, tile in zip(copy.constants(), tiles):
            constant.assign(tile.map(np.random.uniform(0, 1)))
        evaluator.record(True)
        copy.refresh_hash()
    counter.count("programs.evaluated", 1)
    with chronometer.clock("evaluation.comparison"):
        ejected, budget_used = topk.challenge_with(
            copy, MAX_BUDGET, prior_experience=returns
        )
        const_opti.best_return = topk.get_best_stats()[1]
        counter.count("episodes.comparison", budget_used)
        if ejected and ejected != copy:
            print_best_program()
            if is_solved():
                log_data()
                print("SOLVED" + "=" * 60)
                break
