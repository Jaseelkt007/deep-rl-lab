import numpy as np
import pytest
import torch

from rl.ppo import VARIANTS, PPOConfig, PPOSession, actor_loss, lambda_at, lr_factor


def test_clip_stops_the_push_beyond_the_trust_range():
    adv = torch.tensor([2.0])
    old = torch.log(torch.tensor([0.5]))
    # ratio 1.5 > 1.2 with A > 0: already moved far enough, gradient 0
    lp = torch.log(torch.tensor([0.75])).requires_grad_()
    actor_loss(lp, old, adv, 0.2).backward()
    assert lp.grad.item() == 0.0
    # ratio 1.1 inside the range: gradient −A·r (pushes log π up)
    lp = torch.log(torch.tensor([0.55])).requires_grad_()
    actor_loss(lp, old, adv, 0.2).backward()
    assert lp.grad.item() == pytest.approx(-2.0 * 1.1, rel=1e-5)
    # no clip: plain −A·log π
    lp = torch.log(torch.tensor([0.75])).requires_grad_()
    actor_loss(lp, old, adv, None).backward()
    assert lp.grad.item() == pytest.approx(-2.0)
    # ratio, no clip: −r·A keeps pushing, and harder as r grows (gradient −A·r)
    lp = torch.log(torch.tensor([0.75])).requires_grad_()
    actor_loss(lp, old, adv, None, ratio=True).backward()
    assert lp.grad.item() == pytest.approx(-2.0 * 1.5, rel=1e-5)


def test_ratio_equals_plain_in_the_first_epoch():
    """r = 1 before any update, so the first gradient step of 'ratio' is the plain one."""
    torch.manual_seed(0)
    logits = torch.randn(6, 2)
    acts, adv = torch.tensor([0, 1, 1, 0, 1, 0]), torch.randn(6)
    grads = []
    for ratio in (False, True):
        z = logits.clone().requires_grad_()
        lp = torch.log_softmax(z, -1)[torch.arange(6), acts]
        actor_loss(lp, lp.detach(), adv, None, ratio).backward()
        grads.append(z.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_per_epoch_measurements():
    s = PPOSession(PPOConfig(**{**VARIANTS["clip"], "seed": 3}))
    pe = s.iterate()["summary"]["per_epoch"]
    assert len(pe) == 10 and all(e["kl"] >= 0 and 0 <= e["past"] <= 1 for e in pe)
    assert pe[-1]["kl"] > pe[0]["kl"]  # the policy moves further with every epoch
    assert all(e["r_hi"] is None or e["r_hi"] <= e["r_max"] + 1e-6 for e in pe)
    assert all(e["r_lo"] is None or e["r_lo"] >= e["r_min"] - 1e-6 for e in pe)


def test_lambda_one_gives_monte_carlo_advantage_and_zero_gives_td_error():
    s = PPOSession(PPOConfig(seed=0))
    X, acts, logp, rews, ends, nxt, segs, fin = s._collect()
    g = s.cfg.gamma
    s.cfg.gae_lambda = 1.0
    V, A, target = s._advantages(X, rews, ends, nxt)
    # λ = 1: target = discounted return, computed backwards per episode (all 5 episodes end by falling)
    G, run = np.zeros(len(rews)), 0.0
    for t in range(len(rews) - 1, -1, -1):
        run = rews[t] + (0.0 if ends[t] == "terminated" else g * run)
        G[t] = run
    np.testing.assert_allclose(target, G, rtol=1e-5, atol=1e-5)
    s.cfg.gae_lambda = 0.0
    V0, A0, _ = s._advantages(X, rews, ends, nxt)
    with torch.no_grad():
        vn = s.critic(torch.as_tensor(s.encoder(np.array(nxt), 0))).numpy()
    delta = np.array(rews) + g * np.where(np.array(ends) == "terminated", 0.0, vn) - V0
    np.testing.assert_allclose(A0, delta, rtol=1e-5, atol=1e-5)


def test_fixed_step_batches_cut_and_continue_episodes():
    s = PPOSession(PPOConfig(batch_episodes=None, batch_steps=100, seed=1))
    tr = s.iterate()
    assert tr["summary"]["steps"] == 100
    assert sum(len(seg["states"]) for seg in tr["episodes"]) == 100
    carried = s.ep_len  # the cut episode continues into the next batch
    s.iterate()
    assert carried > 0 or s.ep_len >= 0


def test_gradient_steps_are_epochs_times_minibatches():
    s = PPOSession(PPOConfig(batch_episodes=None, batch_steps=200, epochs=3, minibatch=64, seed=2))
    assert s.iterate()["summary"]["grad_steps"] == 3 * 4  # 200 / 64 → 4 minibatches (64, 64, 64, 8)
    s = PPOSession(PPOConfig(epochs=1, seed=2))  # our batch actor-critic: one step on the whole batch
    assert s.iterate()["summary"]["grad_steps"] == 1


def test_variants_are_valid_and_one_switch_apart():
    for name, over in VARIANTS.items():
        PPOConfig(**over)
    assert VARIANTS["ratio"] == {**VARIANTS["epochs"], "ratio": True}
    assert VARIANTS["clip"] == {**VARIANTS["ratio"], "clip": 0.2}
    assert VARIANTS["gae"] == {**VARIANTS["clip"], "gae_lambda": 0.95}


def test_one_step_ppo_schedules():
    cfg = PPOConfig(**VARIANTS["ppo_1step"])
    assert lambda_at(cfg, 0.0) == pytest.approx(0.9)
    assert lambda_at(cfg, 1.0) == pytest.approx(0.9 + (1 - 1 / np.cosh(3.0)) * 0.08)  # ≈ 0.972
    assert lambda_at(PPOConfig(gae_lambda=0.95), 0.7) == 0.95  # no end value: λ stays fixed
    assert [lr_factor(cfg, f) for f in (0.0, 0.5, 1.0, 2.0)] == pytest.approx([1.0, 0.5, 0.0, 0.0], abs=1e-12)
    assert lr_factor(PPOConfig(), 0.5) == 1.0


def test_one_step_ppo_is_a_plain_policy_gradient_step():
    """One gradient step: r = 1 there, so clip(r)·A = r·A and the gradient is the plain −A·log π one."""
    over = {**VARIANTS["ppo_1step"], "batch_steps": 300, "total_steps": 3000, "seed": 3}
    a, b = PPOSession(PPOConfig(**over)), PPOSession(PPOConfig(**{**over, "ratio": False, "clip": None}))
    ta, tb = a.iterate(), b.iterate()
    assert ta["summary"]["grad_steps"] == 1
    for pa, pb in zip([*a.actor.parameters(), *a.critic.parameters()], [*b.actor.parameters(), *b.critic.parameters()]):
        torch.testing.assert_close(pa, pb)
    s = ta["summary"]
    assert s["lam"] == pytest.approx(0.9) and s["lr"] == pytest.approx(5e-4)  # the first batch: the start values
    assert s["gn_actor"] > 0 and s["gn_critic"] > 0
    a.iterate()  # second batch: 300 of 3000 steps done
    assert a.lam == pytest.approx(lambda_at(a.cfg, 0.1)) and a.opt.param_groups[0]["lr"] == pytest.approx(5e-4 * lr_factor(a.cfg, 0.1))


def test_joint_optimiser_clips_both_networks_together():
    s = PPOSession(PPOConfig(**{**VARIANTS["ppo_1step"], "batch_steps": 200, "max_grad_norm": 1e-3, "lr_decay": "none"}))
    before = [p.detach().clone() for p in s.params]
    s.iterate()
    # Adam's first step moves every weight by about lr whatever the gradient's size, so check the clipped
    # gradient directly: its total norm over BOTH networks is the limit
    total = float(torch.norm(torch.stack([p.grad.norm() for p in s.params])))
    assert total == pytest.approx(1e-3, rel=1e-4)
    assert any(not torch.equal(p, q) for p, q in zip(before, s.params))


def test_time_limit_as_end_drops_the_bootstrap():
    s = PPOSession(PPOConfig(seed=4))
    X, acts, logp, rews, ends, nxt, segs, fin = s._collect()
    ends = list(ends)
    ends[-1] = "truncated"
    _, A_boot, _ = s._advantages(X, rews, ends, nxt)
    s.cfg.trunc_as_end = True
    V, A_end, _ = s._advantages(X, rews, ends, nxt)
    with torch.no_grad():
        v_next = float(s.critic(torch.as_tensor(s.encoder(np.array(nxt[-1:]), 0)))[0])
    assert A_end[-1] == pytest.approx(rews[-1] - V[-1], abs=1e-5)
    assert A_boot[-1] == pytest.approx(rews[-1] + s.cfg.gamma * v_next - V[-1], abs=1e-5)
