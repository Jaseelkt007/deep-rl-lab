"""CartPole-v1 as pure JAX functions (a line-by-line port of gymnasium's cartpole.py).

What changes compared with a gymnasium env, and why:

  gymnasium                              here
  ---------                              ----
  an object that keeps self.state        the state is a value you pass in and get back (EnvState)
  np.random inside the env               a JAX random key is an argument
  `if terminated:` in Python             jnp.where: both branches are computed, one is kept
  you call env.reset() when done         step() resets by itself (auto-reset), so a batch of
                                         512 envs never has to stop for one finished episode

These rules are what let jax.jit compile step() and jax.vmap run 512 copies of it in one call.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

GRAVITY, MASSCART, MASSPOLE, LENGTH, FORCE_MAG, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
TOTAL_MASS = MASSPOLE + MASSCART
POLEMASS_LENGTH = MASSPOLE * LENGTH
THETA_LIMIT = 12 * 2 * jnp.pi / 360
X_LIMIT = 2.4
MAX_STEPS = 500  # gymnasium's TimeLimit for CartPole-v1

OBS_DIM, N_ACTIONS = 4, 2


class EnvState(NamedTuple):
    s: jax.Array  # (4,) x, x_dot, theta, theta_dot
    t: jax.Array  # steps taken in this episode


def reset(key: jax.Array) -> tuple[jax.Array, EnvState]:
    s = jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05)
    return s, EnvState(s=s, t=jnp.array(0, jnp.int32))


def physics(s: jax.Array, action: jax.Array) -> jax.Array:
    """One Euler step of the cart-pole equations, exactly as gymnasium does it."""
    x, x_dot, theta, theta_dot = s
    force = jnp.where(action == 1, FORCE_MAG, -FORCE_MAG)
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin) / TOTAL_MASS
    thetaacc = (GRAVITY * sin - cos * temp) / (LENGTH * (4.0 / 3.0 - MASSPOLE * cos**2 / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * cos / TOTAL_MASS
    return jnp.stack([x + TAU * x_dot, x_dot + TAU * xacc, theta + TAU * theta_dot, theta_dot + TAU * thetaacc])


def step(key: jax.Array, state: EnvState, action: jax.Array):
    """Returns (obs, state, reward, terminated, truncated, final_obs).

    On the step that ends an episode, `obs` and `state` already belong to the NEXT episode (auto-reset), and
    `final_obs` is the real last observation of the finished one. PPO needs final_obs to bootstrap V after a
    time-limit truncation, like rl/ppo.py does with V(next_obs).
    """
    s2 = physics(state.s, action)
    t2 = state.t + 1
    terminated = (jnp.abs(s2[0]) > X_LIMIT) | (jnp.abs(s2[2]) > THETA_LIMIT)
    truncated = (t2 >= MAX_STEPS) & ~terminated
    reward = jnp.float32(1.0)  # CartPole-v1 pays 1 on every step, the failing one included

    obs_reset, state_reset = reset(key)
    done = terminated | truncated
    state_next = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_reset, EnvState(s=s2, t=t2))
    obs_next = jnp.where(done, obs_reset, s2)
    return obs_next, state_next, reward, terminated, truncated, s2


# The playground's ObsEncoder ("scaled"): an affine map of the state to roughly [-1, 1].
OBS_LOW = jnp.array([-2.4, -3.0, -THETA_LIMIT, -3.5])
OBS_HIGH = jnp.array([2.4, 3.0, THETA_LIMIT, 3.5])


def encode(obs: jax.Array) -> jax.Array:
    return 2.0 * (obs - OBS_LOW) / (OBS_HIGH - OBS_LOW) - 1.0
