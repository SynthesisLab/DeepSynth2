import os
import atexit
import argparse
import json
import time
from typing import Dict

import numpy as np

from control_dsl import get_dsl
from examples.rl.optim.constant_optimizer import ConstantOptimizer
from rl.rl_utils import type_for_env
from program_evaluator import ProgramEvaluator


from gpoe_automaton_to_grammar import parse
import gymnasium as gym

from synth.filter.dfta_filter import DFTAFilter
from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.enumeration.beap_search import enumerate_prob_grammar
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.program import Program
from synth.syntax.type_helper import auto_type


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

parser.add_argument("--env", type=str, help="name of the environment")

parser.add_argument("--size", type=int, help="max size of programs")
parser.add_argument("--automaton", type=str, help="automaton file")
parser.add_argument(
    "--env-build-arg",
    dest="env_arg",
    type=str,
    default="{}",
    help="dictionnary of arguments to pass to the env",
)
parser.add_argument("--seed", type=int, default=1, help="seed")
parser.add_argument(
    "-o", "--output", type=str, default="./search_data.csv", help="CSV file name"
)
parser.add_argument(
    "-g",
    "--goal",
    type=float,
    help="target score after which we automatically stop",
)
parser.add_argument(
    "--with-target",
    action="store_true",
    help="Consider only programs where all runs are above threshold",
)


params = parser.parse_args()
SEED: int = params.seed
output_file: str = params.output
env_args: Dict = json.loads(params.env_arg)
env_name: str = params.env
env = gym.make(env_name, **env_args)

# =========================================================================
# GLOBAL PARAMETERS
# max number of episodes that should be done at most to compare two possibly equal (optimised) candidates
MAX_BUDGET: int = 500
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
dfta = parse(
    dsl,
    params.automaton,
    type_request,
    constant_types,
    env.action_space.n if type_request.ends_with(auto_type("action")) else 0,
)
cfg = CFG.infinite(dsl, type_request, constant_types=constant_types)
probcfg = ProbDetGrammar.uniform(cfg)
enumerator = enumerate_prob_grammar(probcfg)
filter = DFTAFilter(dfta)
enumerator.filter = filter
const_opti = ConstantOptimizer(SEED)


def make_eval(program: Program) -> float:
    def f():
        program.refresh_hash()
        evaluator.eval(program)
        return evaluator.returns(program)[-1]

    return f


already_loaded = []
progs = []

if os.path.exists(params.output):
    with open(params.output) as fd:
        already_loaded = [x.strip("\n") for x in fd.readlines() if len(x.strip()) > 1]
    print("loaded", len(already_loaded), "programs")


def save():
    print("saved", len(progs), "new programs")
    with open(params.output, "w") as fd:
        fd.writelines(list(map(lambda x: x + "\n", already_loaded)))
        for prog in progs:
            str_prog = str(prog)
            # Put all constants in << >>
            fd.write(f"{str_prog}\n")


atexit.register(save)
g = enumerator.generator()
i = 0
size = 0
last_save = time.time()
try:
    while size <= params.size:
        program = next(g)
        if program.size() > size:
            size = program.size()
            print("size:", size, "/", params.size)
            if size > params.size:
                continue
        if i < len(already_loaded):
            i += 1
            continue
        if time.time() - last_save >= 300:
            save()
            last_save = time.time()
        copy = program.clone()
        tiles = None
        returns = []
        if copy.count_constants() > 0:
            np.random.seed(SEED)
            evaluator.record(False)
            tiles, returns = const_opti.optimize(
                make_eval(copy),
                list(copy.constants()),
                max_total_budget=10 * MAX_BUDGET,
            )
            for constant, tile in zip(copy.constants(), tiles):
                constant.assign(tile.map(np.random.uniform(0, 1)))
            evaluator.record(True)
            copy.refresh_hash()

        progs.append(copy)
except StopIteration:
    pass
save()
