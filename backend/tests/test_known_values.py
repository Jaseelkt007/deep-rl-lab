"""Correctness check with exactly known answers (deterministic, cannot flake).

The full pipeline (trainer -> n-step window -> update -> critic) must learn values we can
compute by hand. If these pass, the TD machinery is right, and a learning failure on
MountainCar is the problem's fault, not a bug.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl.config import RunConfig
from rl.trainer import Trainer

GAMMA = 0.9


class ChainEnv(gym.Env):
    """s0 -> s1 -> … -> s4 -> terminal, reward +1 per step, whatever the action."""

    observation_space = gym.spaces.Box(low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07]), dtype=np.float64)
    action_space = gym.spaces.Discrete(3)
    N = 5

    @staticmethod
    def obs(k: int) -> np.ndarray:
        return np.array([-1.0 + 0.35 * k, 0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        return self.obs(0), {}

    def step(self, action):
        self.k += 1
        return self.obs(self.k), 1.0, self.k >= self.N, False, {}


class LoopEnv(gym.Env):
    """One state that never ends by itself; the time limit cuts it after T steps.

    The true value (the task has no end) is 1 / (1 − γ) = 10 for γ = 0.9.
    """

    observation_space = ChainEnv.observation_space
    action_space = gym.spaces.Discrete(3)

    def __init__(self, T: int = 20):
        self.T = T

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([-0.5, 0.0]), {}

    def step(self, action):
        self.t += 1
        return np.array([-0.5, 0.0]), 1.0, False, self.t >= self.T, {}


def learned_values(tr: Trainer, states: list[np.ndarray]) -> np.ndarray:
    x = torch.as_tensor(tr.encoder(np.array(states), 0))
    return tr.agent.evaluate(x)[2].numpy()


@pytest.mark.parametrize("n_step", [1, 3])
def test_critic_learns_the_exact_chain_values(n_step):
    """SGD, because its step is lr·δ·∇V: at the fixed point δ = 0 and it stops. Adam's step
    stays ~lr even when δ ≈ 0, so with Adam V hovers around the answer instead of settling
    (it does reach it: lr 3e-3 gives the exact values after ~3000 episodes)."""
    exact = np.array([sum(GAMMA**j for j in range(ChainEnv.N - k)) for k in range(ChainEnv.N)])
    cfg = RunConfig(gamma=GAMMA, n_step=n_step, num_episodes=600, optimizer="sgd", critic_lr=0.05, actor_lr=0.01,
                    snapshot_every=10**6, state_probe_every=10**6, checkpoint_every=10**6)
    tr = Trainer(cfg, env=ChainEnv())
    tr.run()
    v = learned_values(tr, [ChainEnv.obs(k) for k in range(ChainEnv.N)])
    np.testing.assert_allclose(v, exact, rtol=0.05)


def test_bootstrap_at_truncation_learns_the_true_value_and_the_bug_does_not():
    common = dict(gamma=GAMMA, num_episodes=300, critic_lr=3e-3,
                  snapshot_every=10**6, state_probe_every=10**6, checkpoint_every=10**6)
    good = Trainer(RunConfig(**common), env=LoopEnv())
    good.run()
    bug = Trainer(RunConfig(bootstrap_on_truncation=False, **common), env=LoopEnv())
    bug.run()
    v_good = learned_values(good, [np.array([-0.5, 0.0])])[0]
    v_bug = learned_values(bug, [np.array([-0.5, 0.0])])[0]
    assert v_good == pytest.approx(1 / (1 - GAMMA), rel=0.05)
    assert v_bug < 0.9 * v_good  # treating the time limit as terminal underestimates V
