from typing import Callable, List, Tuple
import gymnasium as gym
from synth.semantic.evaluator import Evaluator
from synth.syntax.program import Program
import numpy as np


def __state2env__(state: np.ndarray) -> Tuple:
    return tuple(state.tolist())


def __adapt_action2env__(env: gym.Env, action) -> List:
    if isinstance(env.action_space, gym.spaces.Box):
        if len(env.action_space.shape) == 1 and env.action_space.shape[0] == 1:
            return [min(max(action, env.action_space.low[0]), env.action_space.high[0])]
    return action


class ProgramEvaluator:
    def __init__(self, env_factory: Callable[[], gym.Env], evaluator: Evaluator):
        self.cache = {}
        self.env_factory = env_factory
        self.dsl_eval = evaluator
        self.recording = True
        self.tmp_keys = []

    def record(self, record: bool):
        if not self.recording and record:
            for key in self.tmp_keys:
                del self.cache[key]
            self.tmp_keys.clear()
        self.recording = record

    def delete_data(self, program: Program):
        del self.cache[program.hash]

    def returns(self, program: Program) -> List[float]:
        return self.cache.get(program.hash, (0, []))[1]

    def mean_return(self, program: Program) -> float:
        r = self.returns(program)
        if len(r) == 0:
            return 0
        return sum(r) / len(r)

    def samples(self, program: Program) -> int:
        return len(self.cache.get(program.hash, (0, []))[1])

    def add_returns(self, program: Program, returns: List[float]):
        if program.hash not in self.cache:
            self.cache[program.hash] = (self.env_factory(), [])
            if not self.recording:
                self.tmp_keys.append(program.hash)
        li = self.returns(program)
        for el in returns:
            li.append(el)

    def eval(self, program: Program, n_episodes: int = 1) -> bool:
        if program.hash not in self.cache:
            self.cache[program.hash] = (self.env_factory(), [])
            if not self.recording:
                self.tmp_keys.append(program.hash)
        env, returns = self.cache[program.hash]
        try:
            state = None
            for _ in range(n_episodes):
                episode = []
                state = env.reset()[0]
                done = False
                while not done:
                    input = __state2env__(state)
                    action = self.dsl_eval.eval(program, input)
                    adapted_action = __adapt_action2env__(env, action)
                    if adapted_action not in env.action_space:
                        return False
                    next_state, reward, done, truncated, _ = env.step(adapted_action)
                    done |= truncated
                    episode.append(reward)
                    state = next_state
                returns.append(sum(episode))
        except OverflowError:
            return False
        return True
