from typing import Callable, List
import gymnasium as gym
import tqdm
from examples.rl.program_evaluator import ProgramEvaluator
from synth.semantic.evaluator import Evaluator
from synth.syntax.program import Program
import pandas as pd
import os
import atexit


def evaluate_programs_to_csv(
    programs: List[Program],
    env_factory: Callable[[], gym.Env],
    evaluator: Evaluator,
    num_seeds: int,
    csv_file: str = "program_evaluation.csv",
    save_every: int = 10
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
    program_evaluator = ProgramEvaluator(env_factory, evaluator)
    results = {}

    def save():
        data = []
        for program, rewards in results.items():
            if len(rewards) == 0:
                continue
            row = [str(program)] + [rewards[i] if i in rewards else None for i in range(num_seeds)]
            data.append(row)
        columns = ["Program"] + [f"reward{i + 1}" for i in range(num_seeds)]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(csv_file, index=False)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for index, row in tqdm.tqdm(df.iterrows(), desc="loading existing data"):
            program_str = row["Program"]
            rewards = [el for el in row[1:].tolist() if str(el) != "None" and len(str(el).strip()) > 0]
            if len(rewards) <= 0:
                continue
            program = next((p for p in programs if str(p) == program_str), None)
            if program:
                results[program] = {i: ri for i, ri in enumerate(rewards)}
            else:
                print("failed parsing:", program)

    for program in programs:
        if program not in results:
            results[program] = {}

    atexit.register(save)
    i = 0
    for program in tqdm.tqdm(programs):
        for seed in range(num_seeds):
            if seed not in results[program]:
                env = env_factory()
                env.reset(seed=seed)
                program_evaluator.cache[program.hash] = (env, [])
                program_evaluator.eval(program)
                results[program][seed] = program_evaluator.returns(program)[-1]
        i += 1
        if i >= save_every:
            save()
            i = 0
    if i != 0:
        save()
    atexit.unregister(save)
