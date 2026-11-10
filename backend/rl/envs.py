"""Environment factory, observation scaling and potential-based reward shaping.

The environment always returns the RAW state (what the UI draws). The agent sees
features made by `ObsEncoder` (scaled to roughly [-1, 1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from . import double_cartpole  # noqa: F401  (registers DoubleCartPole-v0 with gymnasium)
from .config import RunConfig

# MountainCar-v0 physics constants (gymnasium/envs/classic_control/mountain_car.py)
MC_FORCE = 0.001
MC_GRAVITY = 0.0025
MC_MAX_SPEED = 0.07
CP_THETA_LIMIT = 12 * 2 * math.pi / 360  # pole falls beyond ±12°
DCP_LIMIT = math.radians(15)  # double cart-pole: each stick fails beyond ±15° from vertical


@dataclass(frozen=True)
class EnvInfo:
    obs_low: np.ndarray  # scaling range: low maps to -1, high maps to +1
    obs_high: np.ndarray
    obs_names: tuple[str, ...]
    action_names: tuple[str, ...]
    # Phase-plane maps: which two state dims form the axes, their ranges, and the
    # values used for the other dims (CartPole is 4-D, so its map is a slice).
    phase_dims: tuple[int, int]
    phase_ranges: tuple[tuple[float, float], tuple[float, float]]
    phase_base: tuple[float, ...]
    official_max_steps: int


ENV_INFO: dict[str, EnvInfo] = {
    "MountainCar-v0": EnvInfo(
        obs_low=np.array([-1.2, -MC_MAX_SPEED]),
        obs_high=np.array([0.6, MC_MAX_SPEED]),
        obs_names=("position", "velocity"),
        action_names=("left", "none", "right"),
        phase_dims=(0, 1),
        phase_ranges=((-1.2, 0.6), (-MC_MAX_SPEED, MC_MAX_SPEED)),
        phase_base=(0.0, 0.0),
        official_max_steps=200,
    ),
    "CartPole-v1": EnvInfo(
        # Cart position and pole angle have hard limits; the two velocities do not,
        # these ranges cover what a balancing policy visits.
        obs_low=np.array([-2.4, -3.0, -CP_THETA_LIMIT, -3.5]),
        obs_high=np.array([2.4, 3.0, CP_THETA_LIMIT, 3.5]),
        obs_names=("cart position", "cart velocity", "pole angle", "pole angular velocity"),
        action_names=("left", "right"),
        phase_dims=(2, 3),
        phase_ranges=((-CP_THETA_LIMIT, CP_THETA_LIMIT), (-2.0, 2.0)),
        phase_base=(0.0, 0.0, 0.0, 0.0),
        official_max_steps=500,
    ),
    "DoubleCartPole-v0": EnvInfo(
        # Angles: their failure limits (θ1 ±15°; φ = θ2 − θ1 can reach ±30° before a stick
        # fails). Velocities: what random pushes reach (99th percentile ≈ 1.1, 3.2, 4.9).
        obs_low=np.array([-2.4, -3.0, -DCP_LIMIT, -3.5, -2 * DCP_LIMIT, -5.0]),
        obs_high=np.array([2.4, 3.0, DCP_LIMIT, 3.5, 2 * DCP_LIMIT, 5.0]),
        obs_names=("cart position", "cart velocity", "lower stick angle", "lower stick angular velocity",
                   "angle between sticks", "angle between sticks, rate"),
        action_names=("left", "right"),
        phase_dims=(2, 4),
        phase_ranges=((-DCP_LIMIT, DCP_LIMIT), (-2 * DCP_LIMIT, 2 * DCP_LIMIT)),
        phase_base=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        official_max_steps=500,
    ),
}


# ----------------------------------------------------------------------------
# Observation scaling
# ----------------------------------------------------------------------------
def scale_obs(obs: np.ndarray, env_id: str) -> np.ndarray:
    """Affine map of the raw state to roughly [-1, 1]: low -> -1, high -> +1.

    MountainCar: position [-1.2, 0.6], velocity [-0.07, 0.07]. Without this the
    velocity input is ~100x smaller than position and the network barely sees it.
    """
    info = ENV_INFO[env_id]
    return 2.0 * (np.asarray(obs, dtype=np.float64) - info.obs_low) / (info.obs_high - info.obs_low) - 1.0


def rbf_features(x_scaled: np.ndarray, per_dim: int) -> np.ndarray:
    """Gaussian radial basis functions on a regular grid over [-1, 1]^d.

    Each feature responds to one small region of the state space, so the network
    gets a local, already "spread out" input (classic linear-RL features).
    Width = grid spacing. Output size per_dim^d.
    """
    d = x_scaled.shape[-1]
    axis = np.linspace(-1.0, 1.0, per_dim)
    centers = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)
    sigma = 2.0 / (per_dim - 1)
    diff = x_scaled[..., None, :] - centers  # [..., k^d, d]
    return np.exp(-0.5 * np.sum(diff**2, axis=-1) / sigma**2)


class ObsEncoder:
    """Turns raw states into network inputs. Works on a single state or a batch [..., d].

    obs_mode:  raw    -> the state as the env returns it (MountainCar velocity is ~0.01!)
               scaled -> affine map to [-1, 1]   (the plain setting)
               rbf    -> Gaussian RBF grid on the scaled state
    time_input: append the fraction of the episode left, (T - t) / T. Without it the
               time limit makes the problem non-Markov: the same state has a different
               future at t = 5 and at t = 195 (Pardo et al. 2018, "Time limits in RL").
    """

    def __init__(self, cfg: RunConfig):
        self.env_id = cfg.env_id
        self.mode = cfg.obs_mode
        self.time_input = cfg.time_input
        self.max_steps = cfg.max_episode_steps or ENV_INFO[cfg.env_id].official_max_steps
        state_dim = len(ENV_INFO[cfg.env_id].obs_low)
        self.rbf_k = cfg.rbf_per_dim or (10 if state_dim <= 2 else 5)
        base = self.rbf_k**state_dim if self.mode == "rbf" else state_dim
        self.dim = base + (1 if self.time_input else 0)

    def __call__(self, obs: np.ndarray, t: np.ndarray | int = 0) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if self.mode == "raw":
            x = obs
        elif self.mode == "scaled":
            x = scale_obs(obs, self.env_id)
        else:
            x = rbf_features(scale_obs(obs, self.env_id), self.rbf_k)
        if self.time_input:
            t_left = (self.max_steps - np.asarray(t, dtype=np.float64)) / self.max_steps
            t_left = np.broadcast_to(t_left, x.shape[:-1])[..., None]
            x = np.concatenate([x, t_left], axis=-1)
        return x.astype(np.float32)


def action_map(cfg: RunConfig) -> list[int]:
    """Agent action index -> env action. MountainCar 'no_noop' keeps only left/right."""
    n = len(ENV_INFO[cfg.env_id].action_names)
    if cfg.env_id == "MountainCar-v0" and cfg.actions == "no_noop":
        return [0, 2]
    return list(range(n))


# ----------------------------------------------------------------------------
# Potential-based reward shaping (Ng, Harada & Russell 1999)
# ----------------------------------------------------------------------------
def potential(obs: np.ndarray, env_id: str, kind: str) -> float:
    """Phi(s), normalised to [0, 1] so shaping_scale sets its size in reward units."""
    if env_id == "MountainCar-v0":
        x, v = float(obs[0]), float(obs[1])
        if kind == "height":
            # Height of the hill y = 0.45 sin(3x) + 0.55 (the drawn track), as a fraction.
            return (math.sin(3 * x) + 1.0) / 2.0
        if kind == "energy":
            # The env's own physics: dv/dt = -g cos(3x), so the potential energy is
            # U(x) = g sin(3x) / 3 and the kinetic energy is v²/2.
            pot = MC_GRAVITY / 3.0 * (math.sin(3 * x) + 1.0)
            kin = 0.5 * v * v
            e_max = MC_GRAVITY / 3.0 * 2.0 + 0.5 * MC_MAX_SPEED**2
            return (pot + kin) / e_max
    if env_id == "CartPole-v1":
        theta = float(obs[2])
        if kind in ("height", "energy"):
            # Height of the pole tip: 1 when upright, 0 at the failure angle.
            c_min = math.cos(CP_THETA_LIMIT)
            return max(0.0, (math.cos(theta) - c_min) / (1.0 - c_min))
    raise ValueError(f"no potential '{kind}' for {env_id}")


def shaping_bonus(phi_s: float, phi_next: float, terminated: bool, gamma: float) -> float:
    """F = γ·Φ(s') − Φ(s), with Φ(terminal) = 0.

    Using exactly this form, the shaped problem has the same optimal policy as the
    original one (Ng et al. 1999): over an episode the bonuses telescope to
    −Φ(s_0) + γ^T Φ(s_T), which no action can change. Φ(terminal) must be 0, else
    the agent is paid (or charged) for ending the episode.
    """
    return gamma * (0.0 if terminated else phi_next) - phi_s


class PotentialShaping(gym.Wrapper):
    """Adds F to the reward. info['r_env'] keeps the original reward: scores use it."""

    def __init__(self, env: gym.Env, env_id: str, kind: str, gamma: float, scale: float):
        super().__init__(env)
        self.env_id, self.kind, self.gamma, self.scale = env_id, kind, gamma, scale
        self._phi = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi = self.scale * potential(obs, self.env_id, self.kind)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        phi_next = self.scale * potential(obs, self.env_id, self.kind)
        f = shaping_bonus(self._phi, phi_next, terminated, self.gamma)
        self._phi = phi_next
        info = dict(info, r_env=float(r), shaping=f)
        return obs, float(r) + f, terminated, truncated, info


def make_env(cfg: RunConfig) -> gym.Env:
    """gym.make with the configured time limit (TimeLimit sets truncated=True at the limit)."""
    max_steps = cfg.max_episode_steps or ENV_INFO[cfg.env_id].official_max_steps
    env = gym.make(cfg.env_id, max_episode_steps=max_steps)
    if cfg.shaping != "none":
        env = PotentialShaping(env, cfg.env_id, cfg.shaping, cfg.gamma, cfg.shaping_scale)
    return env
