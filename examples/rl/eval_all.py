import argparse
import json
from typing import Dict

import tqdm

from control_dsl import get_dsl
from rl.rl_utils import type_for_env
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
parser.add_argument("--warm", type=str, default="")
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


def clever_parse(str_program: str) -> Program:
    constants = {}
    while True:
        try:
            prog = dsl.parse_program(str_program, type_request, constants)
            return prog
        except AssertionError as e:
            msg: str = e.args[0]
            end = msg.rfind("'")
            start = msg.rfind("'", None, end - 1)
            str_float = msg[start + 1 : end]
            constants[str_float] = (auto_type("float"), float(str_float))


with open(params.file) as fd:
    str_programs = fd.readlines()
    programs = [
        clever_parse(str_prog.replace("\n", ""))
        for str_prog in tqdm.tqdm(str_programs, desc="parsing")
    ]
evaluate_programs_to_csv(
    programs, build_env, prog_evaluator, MAX_BUDGET, output_file, bootstrap=params.warm
)
