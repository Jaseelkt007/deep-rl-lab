"""Known answer for the JAX LunarLander (jaxrl/lunar.py): gymnasium's hand-written controller lands it.

In Box2D the controller scores mean 245, median 267, 90% of episodes >= 200 (200 episodes, seeds 0-199).
The JAX lander is close but not identical (see the module docstring): measured mean 218, median 264, 82% >= 200.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrl import lunar


def _first_episode_returns(n: int, seed: int = 1) -> np.ndarray:
    def run(key):
        k0, key = jax.random.split(key)
        obs, st = lunar.reset(k0)

        def body(c, k):
            obs, st, ret, done = c
            obs2, st2, r, term, trunc, _ = lunar.step(k, st, lunar.heuristic(obs))
            return (obs2, st2, ret + r * ~done, done | term | trunc), None

        (_, _, ret, _), _ = jax.lax.scan(body, (obs, st, 0.0, False), jax.random.split(key, lunar.MAX_STEPS))
        return ret

    return np.asarray(jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(seed), n)))


def test_reset_and_step_shapes():
    obs, st = lunar.reset(jax.random.PRNGKey(0))
    assert obs.shape == (8,) and float(obs[1]) > 1.0  # starts high above the pad
    obs2, _, r, term, trunc, final = lunar.step(jax.random.PRNGKey(1), st, jnp.array(0))
    assert obs2.shape == final.shape == (8,) and not bool(term) and not bool(trunc) and jnp.isfinite(r)


@pytest.mark.slow
def test_heuristic_lands():
    r = _first_episode_returns(100)
    assert np.median(r) > 230 and (r >= 200).mean() > 0.7, (np.median(r), (r >= 200).mean())
