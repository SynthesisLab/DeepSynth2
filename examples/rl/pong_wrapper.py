from typing import Any
import gymnasium as gym

import numpy as np
# (
#     dict(
#         player_y=51,
#         player_x=46,
#         enemy_y=50,
#         enemy_x=45,
#         ball_x=49,
#         ball_y=54,
#         enemy_score=13,
#         player_score=14,
#     ),
# )


class AtariPreprocessing(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Implements the common preprocessing techniques for Atari environments (excluding frame stacking).

    This class follows the guidelines in Machado et al. (2018),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents".

    Specifically, the following preprocess stages applies to the atari environment:

    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        terminal_on_life_loss: bool = False,
    ):
        """Wrapper for Atari 2600 preprocessing.

        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
            terminal_on_life_loss (bool): `if True`, then :meth:`step()` returns `terminated=True` whenever a
                life is lost.

        Raises:
            DependencyNotInstalled: opencv-python package not installed
            ValueError: Disable frame-skipping in the original env
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            noop_max=noop_max,
            terminal_on_life_loss=terminal_on_life_loss,
        )
        gym.Wrapper.__init__(self, env)

        assert noop_max >= 0
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.terminal_on_life_loss = terminal_on_life_loss

        self.lives = 0
        self.game_over = False

    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale

    def step(self, action) -> tuple:
        """Applies the preprocessing for an :meth:`env.step`."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        obs, reward, terminated, truncated, info = self.env.step(action)
        total_reward += reward
        self.game_over = terminated
        if self.terminal_on_life_loss:
            new_lives = obs[13]
            terminated = terminated or new_lives != self.lives
            self.game_over = terminated
            self.lives = new_lives
        return obs, total_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple:
        """Resets the environment using preprocessing."""
        # NoopReset
        obs, reset_info = self.env.reset(seed=seed, options=options)

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                _, reset_info = self.env.reset(seed=seed, options=options)

        self.lives = obs[13]
        return obs, reset_info


class PongRAMWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = gym.spaces.Box(0, 255, shape=(8,))
        self.observation_space = gym.spaces.Box(0, 255, shape=(6,))

    def reset(self, *, seed=None, options=None):
        start_obs, info = super().reset(seed=seed, options=options)
        self.last = start_obs
        return start_obs, info

    def step(self, action):
        obs, a, b, c, d = super().step(action)
        return self.to_local(obs), a, b, c, d

    def to_local(self, observation: np.ndarray) -> np.ndarray:
        out = np.array(
            [
                self.last[51],
                # self.last[50],
                self.last[49],
                self.last[54],
                observation[51],
                # observation[50],
                observation[49],
                observation[54],
            ]
        )
        self.last = observation
        return out


import ale_py


gym.register_envs(ale_py)


def make_pong():
    env = gym.make("ALE/Pong-ram-v5", frameskip=1)

    env = AtariPreprocessing(
        env,
        noop_max=2,
        terminal_on_life_loss=True,
    )
    env = PongRAMWrapper(env)
    env = gym.wrappers.TimeLimit(env, 2000)
    return env
