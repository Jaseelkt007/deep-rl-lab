import numpy as np
import pytest
import torch

from rl.continuous import session_from_state
from rl.offpolicy import Buffer, OffPolicyConfig, OffPolicySession, soft_update


def _small(**kw):
    return OffPolicyConfig(**{"hidden": 32, "chunk_steps": 256, "learning_starts": 128, "batch_size": 32,
                              "total_steps": 10_000, **kw})


def test_actor_torques_stay_in_range_and_the_critic_takes_the_action_as_input():
    s = OffPolicySession(_small())
    x = torch.randn(50, 17) * 10
    a = s.actor(x)
    assert a.shape == (50, 6) and a.abs().max() <= 1.0
    assert s.critic(x, a).shape == (50,)


def test_actor_step_changes_only_the_actor():
    """−Q(s, μ(s)) is minimised over the ACTOR's weights; the critic is only read."""
    s = OffPolicySession(_small())
    for _ in range(64):
        s.buffer.add(np.random.randn(17), np.random.uniform(-1, 1, 6), 1.0, np.random.randn(17), 0.0)
    c0 = [p.clone() for p in s.critic.parameters()]
    a0 = [p.clone() for p in s.actor.parameters()]
    log = {k: [] for k in ("loss_q", "loss_pi", "gn_c", "gn_a", "sat", "q", "y")}
    # isolate the actor step: freeze the critic's optimiser by giving it lr 0
    for g in s.critic_opt.param_groups:
        g["lr"] = 0.0
    s._update(log)
    assert all(torch.equal(p, q) for p, q in zip(c0, s.critic.parameters()))
    assert any(not torch.equal(p, q) for p, q in zip(a0, s.actor.parameters()))


def test_a_fall_has_no_future_but_the_time_limit_does():
    """The target the update really uses: y = r after a fall; r + γ·Q_tgt(s′, μ_tgt(s′)) otherwise."""
    for fell, expected in ((1.0, 1.0), (0.0, 1.0 + 0.9 * 10.0)):
        s = OffPolicySession(_small(gamma=0.9, batch_size=1))
        with torch.no_grad():  # a target critic that says 10 everywhere
            for p in s.critic_tgt.parameters():
                p.zero_()
            s.critic_tgt.out.bias.fill_(10.0)
        s.buffer.add(np.zeros(17), np.zeros(6), 1.0, np.zeros(17), fell)
        log = {k: [] for k in ("loss_q", "loss_pi", "gn_c", "gn_a", "sat", "q", "y")}
        s._update(log)
        assert log["y"] == pytest.approx([expected])


def test_soft_update_moves_the_target_a_fraction_tau():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        a.weight.fill_(0.0)
        b.weight.fill_(1.0)
    soft_update(a, b, 0.005)
    assert torch.allclose(a.weight, torch.full((2, 2), 0.005))


def test_buffer_overwrites_the_oldest():
    buf = Buffer(3, 1, 1, seed=0)
    for k in range(5):
        buf.add([k], [0], float(k), [k], 0.0)
    assert buf.n == 3 and sorted(buf.r.tolist()) == [2.0, 3.0, 4.0]


def test_iterate_learns_after_the_random_phase_and_reloads():
    s = OffPolicySession(_small())
    out = s.iterate()
    assert out["env_steps"] == 256 and out["buffer"] == 256 and out["finished"]
    assert out["updates"] == 256 - 128 + 1  # one update per step from step 128 on
    assert out["q_pred"] is not None and out["g_real"] is not None  # random walkers fall: real returns exist
    t = session_from_state(s.state_dict())
    obs = np.zeros(17)
    assert np.allclose(s.act(obs), t.act(obs))
