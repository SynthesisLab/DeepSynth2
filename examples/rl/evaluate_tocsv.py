from typing import Callable, List
import gymnasium as gym
import tqdm
from examples.rl.program_evaluator import ProgramEvaluator
from synth.semantic.evaluator import Evaluator
from synth.syntax.program import Program
import pandas as pd
import os
import atexit

from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_programs_to_csv(
    programs: List[Program],
    env_factory: Callable[[], gym.Env],
    evaluator: Evaluator,
    num_seeds: int,
    csv_file: str = "program_evaluation.csv",
    save_every: int = 10,
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

    def eval_prog(program: Program, rew: dict):
        program_evaluator = ProgramEvaluator(env_factory, evaluator)
        if len(rew) == num_seeds:
            return program, rew
        for seed in range(num_seeds):
            if seed not in rew:
                env = env_factory()
                env.reset(seed=seed)
                program_evaluator.cache[program.hash] = (env, [])
                program_evaluator.eval(program)
                rew[seed] = program_evaluator.returns(program)[-1]
        return program, rew

    atexit.register(save)
    to_apply = [
        (prog, results[prog].copy())
        for prog in programs
        if len(results[prog]) < num_seeds
    ]

    with ProcessPoolExecutor(procs) as p:
        futures = [p.submit(eval_prog, el) for el in to_apply]
        remaining = len(futures)
        pbar = tqdm.tqdm(remaining)
        for f in as_completed(futures):
            if f.done():
                prog, rew = f.result()
                results[prog] = rew
                remaining -= 1
            if remaining % save_every == 0:
                save()
            pbar.update(1)
        pbar.close()

    save()
    atexit.unregister(save)
