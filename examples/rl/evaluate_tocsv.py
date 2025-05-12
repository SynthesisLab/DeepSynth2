from typing import Callable, List
import gymnasium as gym
import tqdm
from examples.rl.control_dsl import get_dsl
from examples.rl.program_evaluator import ProgramEvaluator
from synth.syntax.program import Program
import pandas as pd
import os

from concurrent.futures import ProcessPoolExecutor, as_completed


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def eval_prog(
    programs: list[Program],
    env_factory,
    type_request,
    action_space,
    num_seeds: int,
    rews: list[dict],
):
    (
        dsl,
        prog_evaluator,
    ) = get_dsl(
        type_request,
        action_space,
    )
    program_evaluator = ProgramEvaluator(env_factory, prog_evaluator)
    results = []
    for program, rew in zip(programs, rews):
        for seed in range(num_seeds):
            if seed not in rew:
                env = env_factory()
                env.reset(seed=seed)
                program_evaluator.cache[program.hash] = (env, [])
                program_evaluator.eval(program)
                rew[seed] = program_evaluator.returns(program)[-1]
        results.append((str(program), rew))
    return results


def evaluate_programs_to_csv(
    programs: List[Program],
    env_factory: Callable[[], gym.Env],
    type_request,
    action_space,
    num_seeds: int,
    csv_file: str = "program_evaluation.csv",
    save_every: int = 50,
    bootstrap: str = "",
    procs: int = 1,
):
    """
    Evaluates a list of programs on an environment for multiple seeds and saves the results to a CSV.

    Args:
        programs: List of programs to evaluate.
        env_factory: Function that returns a new environment instance.
        evaluator: Evaluator object for evaluating programs.
        num_seeds: Number of seeds to use for evaluation.
        csv_file: Path to the CSV file to save the results.
    """
    results = {}

    progs_as_str = {str(p): i for i, p in enumerate(programs)}

    def parse_prog(str_prog: str) -> Program:
        key = progs_as_str.get(str_prog)
        if key is None:
            return None
        else:
            return programs[key]

    try:
        with open(bootstrap) as fd:
            lines = fd.readlines()
            lines.pop(0)
            for line in tqdm.tqdm(lines, desc="loading bootstrap"):
                row = line.split(",")
                program_str = row[0]
                program = parse_prog(program_str)
                if program is None:
                    continue
                rewards = []
                for el in row[1:]:
                    if str(el) != "None" and len(str(el).strip()) > 0:
                        rewards.append(float(el))
                    else:
                        break
                if len(rewards) <= 0:
                    continue
                results[program] = {i: ri for i, ri in enumerate(rewards)}
    except FileNotFoundError:
        pass

    def save():
        data = []
        for program, rewards in results.items():
            if len(rewards) == 0:
                continue
            row = [str(program)] + [
                rewards[i] if i in rewards else None for i in range(num_seeds)
            ]
            data.append(row)
        columns = ["Program"] + [f"reward{i + 1}" for i in range(num_seeds)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(csv_file, index=False)

    if os.path.exists(csv_file):
        progs_loaded = 0
        seeds_loaded = 0
        with open(csv_file) as fd:
            lines = fd.readlines()
            lines.pop(0)
            for line in tqdm.tqdm(lines, desc="loading existing data"):
                row = line.split(",")
                program_str = row[0]
                rewards = []
                for el in row[1:]:
                    if str(el) != "None" and len(str(el).strip()) > 0:
                        rewards.append(float(el))
                    else:
                        break
                if len(rewards) <= 0:
                    print("no data for:", program_str)
                    continue
                program = parse_prog(program_str)
                progs_loaded += 1
                if program:
                    results[program] = {i: ri for i, ri in enumerate(rewards)}
                    seeds_loaded += len(results[program])
                else:
                    print("failed parsing:", program)
        print(
            f"Loaded {progs_loaded}/{len(programs)} ({progs_loaded / len(programs):.2%}) programs and {seeds_loaded}/{len(programs) * num_seeds} ({seeds_loaded / (len(programs) * num_seeds):.2%}) episodes"
        )
    for program in programs:
        if program not in results:
            results[program] = {}

    relevant = [prog for prog in programs if len(results[prog]) < num_seeds]
    to_apply = [
        (
            progs,
            env_factory,
            type_request,
            action_space,
            num_seeds,
            [results[prog].copy() for prog in progs],
        )
        for progs in batch(relevant, save_every // 2)
    ]

    with ProcessPoolExecutor(procs) as p:
        futures = [p.submit(eval_prog, *el) for el in to_apply]
        remaining = len(futures)
        pbar = tqdm.tqdm(total=remaining)
        last_saved = False
        for f in as_completed(futures):
            if f.done():
                for prog_as_str, rew in f.result():
                    results[parse_prog(prog_as_str)] = rew
                    pbar.update(1)
                if last_saved:
                    last_saved = False
                else:
                    save()
                    last_saved = True
        pbar.close()

    save()
