import numpy as np
import pytest
import torch

from rl.dqn import DQN_VARIANTS, DQNConfig, DQNSession, Replay


def make(**over):
    return DQNSession(DQNConfig(**{"env_id": "CartPole-v1", "hidden": 16, "chunk_steps": 200, "total_steps": 10_000, **over}))


def test_variants_are_valid_and_one_switch_apart():
    for over in DQN_VARIANTS.values():
        DQNConfig(**over)
    assert DQN_VARIANTS["dqn"] == {**DQN_VARIANTS["q_replay"], "target_net": True}


def test_epsilon_schedule():
    s = make(total_steps=1000, eps_fraction=0.1)
    assert s.epsilon() == pytest.approx(1.0)
    s.steps = 50
    assert s.epsilon() == pytest.approx(1.0 - 0.95 * 0.5)
    s.steps = 100
    assert s.epsilon() == pytest.approx(0.05)
    s.steps = 900
    assert s.epsilon() == pytest.approx(0.05)


def test_policy_is_epsilon_greedy():
    s = make()
    s.qnet.eps = 0.2
    x = torch.randn(5, s.encoder.dim)
    p = s.qnet(x).exp()
    greedy = s.qnet.qvalues(x).argmax(-1)
    torch.testing.assert_close(p.sum(-1), torch.ones(5))
    torch.testing.assert_close(p[torch.arange(5), greedy], torch.full((5,), 0.8 + 0.2 / 2))
    assert torch.equal(p.argmax(-1), greedy)


def test_replay_ring_overwrites_the_oldest():
    b = Replay(3, 2, seed=0)
    for i in range(5):
        b.add(np.full(2, i), i, float(i), np.full(2, i), 0.0)
    assert b.n == 3 and sorted(b.a.tolist()) == [2, 3, 4]
    s, a, r, s2, d = b.sample(10)
    assert s.shape == (10, 2) and set(a.tolist()) <= {2, 3, 4}


def test_target_uses_terminated_only_and_the_frozen_copy():
    s = make(replay=True, target_net=True, learning_starts=10_000)  # no learning: weights stay
    with torch.no_grad():
        for p in s.qnet.parameters():
            p.add_(0.5)  # the online network moves; the target copy does not
    x = torch.randn(1, s.encoder.dim)
    qt = float(s.target.qvalues(x).max())
    assert qt != pytest.approx(float(s.qnet.qvalues(x).max()))
    s.opt = torch.optim.SGD(s.qnet.parameters(), lr=0.0)  # look at the target without changing anything
    _, _, y = s._learn(x, torch.tensor([0]), torch.tensor([1.0]), x, torch.tensor([0.0]))
    assert y[0] == pytest.approx(1.0 + 0.99 * qt, rel=1e-5)
    _, _, y = s._learn(x, torch.tensor([0]), torch.tensor([1.0]), x, torch.tensor([1.0]))
    assert y[0] == pytest.approx(1.0)


def test_a_chunk_has_the_interface_of_a_ppo_batch():
    for name, over in DQN_VARIANTS.items():
        s = make(**{**over, "learning_starts": min(over.get("learning_starts", 0), 50)})
        tr = s.iterate()
        sm = tr["summary"]
        assert sm["steps"] == 200 and len(tr["advantages"]) == 200 == len(tr["actor"]["probs_before"])
        assert sum(len(seg["states"]) for seg in tr["episodes"]) == 200
        assert sm["grad_steps"] > 0, name
        assert 0 < sm["eps"] <= 1 and sm["per_epoch"] == []


def test_q_online_learns_cartpole_values():
    """After a few thousand steps Q(s, a) should rise from ~0 towards the real returns (CartPole: +1 per step)."""
    s = make(total_steps=6_000, lr=1e-3)
    for _ in range(25):
        tr = s.iterate()
    assert tr["summary"]["actor_loss"] > 3  # mean Q(s, a) clearly above its start
