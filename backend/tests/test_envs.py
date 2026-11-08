import numpy as np
import pytest

from rl.config import RunConfig
from rl.envs import ObsEncoder, action_map, make_env, potential, scale_obs, shaping_bonus


def test_scaling_maps_mountaincar_range_to_minus_one_one():
    low, high = np.array([-1.2, -0.07]), np.array([0.6, 0.07])
    np.testing.assert_allclose(scale_obs(low, "MountainCar-v0"), [-1, -1], atol=1e-12)
    np.testing.assert_allclose(scale_obs(high, "MountainCar-v0"), [1, 1], atol=1e-12)
    np.testing.assert_allclose(scale_obs((low + high) / 2, "MountainCar-v0"), [0, 0], atol=1e-12)


def test_scaled_observations_stay_in_range_during_random_play():
    cfg = RunConfig()
    env, enc = make_env(cfg), ObsEncoder(cfg)
    obs, _ = env.reset(seed=0)
    xs = [enc(obs)]
    for _ in range(300):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        xs.append(enc(obs))
        if term or trunc:
            obs, _ = env.reset()
    xs = np.array(xs)
    assert xs.min() >= -1.0 - 1e-6 and xs.max() <= 1.0 + 1e-6
    assert xs.shape[1] == 2


def test_encoder_modes_and_time_input_shapes():
    obs = np.array([[-0.5, 0.0], [0.3, 0.05]])
    np.testing.assert_allclose(ObsEncoder(RunConfig(obs_mode="raw"))(obs), obs, rtol=1e-6)
    assert ObsEncoder(RunConfig(obs_mode="rbf"))(obs).shape == (2, 100)
    enc = ObsEncoder(RunConfig(time_input=True))
    x = enc(obs, t=np.array([0, 100]))
    assert x.shape == (2, 3)
    assert x[:, -1].tolist() == pytest.approx([1.0, 0.5])  # fraction of 200 steps left


def test_mountaincar_time_limit_is_truncation_not_termination():
    env = make_env(RunConfig())
    env.reset(seed=0)
    for t in range(200):
        _, _, term, trunc, _ = env.step(1)  # "do nothing" never reaches the goal
        assert not term
        assert trunc == (t == 199)


def test_configurable_time_limit():
    env = make_env(RunConfig(max_episode_steps=50))
    env.reset(seed=0)
    for t in range(50):
        _, _, _, trunc, _ = env.step(1)
    assert trunc


def test_shaping_bonus_form_and_terminal_potential_is_zero():
    assert shaping_bonus(0.2, 0.5, terminated=False, gamma=0.9) == pytest.approx(0.9 * 0.5 - 0.2)
    assert shaping_bonus(0.2, 0.5, terminated=True, gamma=0.9) == pytest.approx(-0.2)


def test_shaping_telescopes_over_an_episode():
    """Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0): the total bonus is set by the end points only."""
    gamma, scale = 0.99, 10.0
    cfg = RunConfig(shaping="energy", shaping_scale=scale, gamma=gamma, max_episode_steps=60)
    env = make_env(cfg)
    obs0, _ = env.reset(seed=1)
    total, disc, obs = 0.0, 1.0, obs0
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, r, term, trunc, info = env.step(int(rng.integers(3)))
        assert r == pytest.approx(info["r_env"] + info["shaping"])
        total += disc * info["shaping"]
        disc *= gamma
        if term or trunc:
            break
    expected = disc * scale * potential(obs, cfg.env_id, "energy") - scale * potential(obs0, cfg.env_id, "energy")
    assert total == pytest.approx(expected, abs=1e-9)


def test_potentials_are_normalised():
    for kind in ("height", "energy"):
        for x in np.linspace(-1.2, 0.6, 19):
            for v in (-0.07, 0.0, 0.07):
                assert 0.0 <= potential(np.array([x, v]), "MountainCar-v0", kind) <= 1.0 + 1e-9


def test_action_map():
    assert action_map(RunConfig()) == [0, 1, 2]
    assert action_map(RunConfig(actions="no_noop")) == [0, 2]
    assert action_map(RunConfig(env_id="CartPole-v1", actions="no_noop")) == [0, 1]
