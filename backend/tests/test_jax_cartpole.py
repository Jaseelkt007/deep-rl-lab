"""Known answer: the JAX CartPole follows gymnasium's CartPole exactly for the same start and actions."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from jaxrl import cartpole


def test_same_trajectory_as_gymnasium():
    with jax.enable_x64(True):  # gymnasium computes in float64
        _same_trajectory()


def _same_trajectory():
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=3)
    state = cartpole.EnvState(s=jnp.asarray(env.unwrapped.state, jnp.float64), t=jnp.array(0))
    rng = np.random.default_rng(0)
    for i in range(400):
        a = int(rng.integers(2)) if i % 3 else 1 - int(state.s[2] < 0)  # mostly balance, some noise
        g_obs, _, g_term, g_trunc, _ = env.step(a)
        _, state, _, term, trunc, final_obs = cartpole.step(jax.random.PRNGKey(i), state, jnp.array(a))
        np.testing.assert_allclose(np.asarray(final_obs), g_obs, rtol=1e-5, atol=1e-5)
        assert bool(term) == g_term and bool(trunc) == g_trunc
        if g_term or g_trunc:
            return
    raise AssertionError("the episode never ended; pick another seed")


def test_auto_reset():
    state = cartpole.EnvState(s=jnp.array([2.39, 1.0, 0.0, 0.0]), t=jnp.array(7))
    obs, state2, _, term, _, final_obs = cartpole.step(jax.random.PRNGKey(0), state, jnp.array(1))
    assert bool(term) and float(final_obs[0]) > 2.4  # the real last state is kept for V(s')
    assert int(state2.t) == 0 and float(jnp.abs(obs).max()) <= 0.05  # the env already restarted
