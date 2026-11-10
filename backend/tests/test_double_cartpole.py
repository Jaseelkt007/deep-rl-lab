import math

import numpy as np

from rl.double_cartpole import (DoubleCartPoleEnv, LQRController, Params, discretize, energy, from_obs,
                                linear_model, rk4_step, to_obs)


def test_energy_is_conserved_without_force():
    """No push, no friction: kinetic + potential energy must stay constant (checks the equations).
    10 s of free, violent swinging; the drift is integration error, so it must shrink fast with
    smaller time steps (RK4: ~16x per halving)."""
    drift = []
    for sub in (4, 8):
        p = Params(substeps=sub)
        q, qd = np.array([0.0, 0.4, -0.3]), np.array([0.2, 0.0, 0.5])
        e0 = energy(q, qd, p)
        for _ in range(500):
            q, qd = rk4_step(q, qd, 0.0, p)
        drift.append(abs(energy(q, qd, p) - e0) / e0)
    assert drift[0] < 1e-3 and drift[1] < drift[0] / 10


def test_small_angles_match_the_linear_model():
    """The difference must come only from the non-linear terms: 10x smaller state -> ~1000x smaller."""
    p = Params()
    Ad, Bd = discretize(*linear_model(p), p.dt)
    err = []
    for k in (1.0, 0.1):
        s0 = np.array([0.0, 0.01, -0.005, 0.0, 0.02, 0.0]) * k  # x, θ1, θ2, ẋ, θ̇1, θ̇2
        q, qd = rk4_step(s0[:3], s0[3:], k, p)
        err.append(np.max(np.abs(np.concatenate([q, qd]) - (Ad @ s0 + Bd[:, 0] * k))))
    assert err[0] < 1e-4 and err[1] < err[0] / 300


def test_straight_up_is_an_equilibrium_and_mirror_starts_mirror():
    p = Params()
    q, qd = rk4_step(np.zeros(3), np.zeros(3), 0.0, p)
    assert np.allclose(q, 0) and np.allclose(qd, 0)
    a = (np.array([0.0, 0.05, 0.02]), np.array([0.1, 0.0, -0.1]))
    b = (-a[0], -a[1])
    for _ in range(25):
        a = rk4_step(*a, 2.0, p)
        b = rk4_step(*b, -2.0, p)
    assert np.allclose(a[0], -b[0]) and np.allclose(a[1], -b[1])
    # no push: the lower stick, leaning to +x, falls further that way
    q, qd = np.array([0.0, 0.05, 0.05]), np.zeros(3)
    for _ in range(10):
        q, qd = rk4_step(q, qd, 0.0, p)
    assert q[1] > 0.05
    # pushing the cart to +x rotates a stick that leans to +x back toward vertical
    q, qd = np.array([0.0, 0.05, 0.05]), np.zeros(3)
    q, qd = rk4_step(q, qd, 10.0, p)
    assert qd[1] < 0


def test_observation_uses_the_angle_between_the_sticks():
    q, qd = np.array([0.1, 0.2, 0.05]), np.array([1.0, 2.0, 3.0])
    obs = to_obs(q, qd)
    assert np.allclose(obs, [0.1, 1.0, 0.2, 2.0, -0.15, 1.0])  # φ = θ2 − θ1, φ̇ = θ̇2 − θ̇1
    q2, qd2 = from_obs(obs)
    assert np.allclose(q2, q) and np.allclose(qd2, qd)


def test_failure_rules_and_start():
    env = DoubleCartPoleEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6,) and np.all(np.abs(obs) <= 0.05)
    env.q, env.qd = np.array([2.5, 0.0, 0.0]), np.zeros(3)  # cart off the track
    assert env.step(0)[2]
    env.q, env.qd = np.array([0.0, 0.0, math.radians(16)]), np.zeros(3)  # top stick past 15°
    assert env.step(1)[2]
    env.q, env.qd = np.array([0.0, math.radians(16), 0.0]), np.zeros(3)  # lower stick past 15°
    assert env.step(1)[2]
    env.reset(seed=0)
    assert env.step(1)[1] == 1.0  # reward "alive"


def test_upright_reward_is_one_when_straight_and_zero_at_the_limit():
    env = DoubleCartPoleEnv(reward="upright")
    env.reset(seed=0)
    env.q, env.qd = np.zeros(3), np.zeros(3)
    assert env.step(0)[1] > 0.999
    env.q, env.qd = np.array([0.0, env.angle_limit, env.angle_limit]) * 0.999, np.zeros(3)
    assert env.step(0)[1] < 0.01


def test_lqr_stabilises_the_linear_model():
    p = Params()
    ctrl = LQRController(p, (-10.0, 10.0))
    Ad, Bd = discretize(*linear_model(p), p.dt)
    eig = np.linalg.eigvals(Ad - Bd @ ctrl.K[None, :])
    assert np.max(np.abs(eig)) < 1.0
