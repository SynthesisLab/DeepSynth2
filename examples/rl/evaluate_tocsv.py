from typing import Callable, List, Tuple
import gymnasium as gym
from synth.semantic.evaluator import Evaluator
from synth.syntax.program import Program
import numpy as np
import pandas as pd
import os

def evaluate_programs_to_csv(
    programs: List[Program], 
    env_factory: Callable[[], gym.Env], 
    evaluator: Evaluator, 
    num_seeds: int, 
    csv_file: str = "program_evaluation.csv"
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

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for index, row in df.iterrows():
            program_str = row['Program']
            rewards = row[1:].tolist()
            program = next((p for p in programs if str(p) == program_str), None)
            if program:
                results[program] = rewards

    for program in programs:
        if program not in results:
            results[program] = [0.0] * num_seeds

    for program in programs:
        for seed in range(num_seeds):
            env = env_factory()
            env.reset(seed=seed)
            program_evaluator.cache[program.hash] = (env, [])
            program_evaluator.eval(program, 1)
            results[program][seed] = program_evaluator.returns(program)[-1]
        
    data = []
    for program, rewards in results.items():
        row = [str(program)] + rewards
        data.append(row)
    
    columns = ["Program"] + [f"reward{i+1}" for i in range(num_seeds)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_file, index=False)