import argparse
import json
from typing import Dict

from control_dsl import get_dsl
from rl.rl_utils import type_for_env
from program_evaluator import ProgramEvaluator
from evaluate_tocsv import evaluate_programs_to_csv


import gymnasium as gym

from synth.syntax.program import Program
from synth.syntax.type_helper import auto_type


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

parser.add_argument("--env", type=str, help="name of the environment")

parser.add_argument("-f", "--file", type=str, help="list of programs")
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


def clever_parse(str_program: str) -> Program:
    constants = {}
    start = str_program.find("<<", 0)
    while start != -1:
        end = str_program.find(">>", start)
        str_cst = str_program[start + 2 : end]
        constants[str_cst] = (auto_type("float"), float(str_cst))
        start = end + 2
    return dsl.parse_program(str_program, type_request, constants)


with open(params.file) as fd:
    str_programs = fd.readlines()
    programs = [clever_parse(str_prog.replace("\n", "")) for str_prog in str_programs]
evaluate_programs_to_csv(programs, build_env, evaluator, MAX_BUDGET, output_file)
