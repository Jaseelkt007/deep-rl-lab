import math

import pytest
import torch
import torch.nn.functional as F

from rl.actor_critic import ActorCritic, RunningNorm, lr_factor, should_bootstrap, td_target
from rl.config import RunConfig

GAMMA = 0.9


def make_agent(**kw) -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(obs_dim=2, n_actions=3, cfg=RunConfig(gamma=GAMMA, **kw))


def constant_critic(agent: ActorCritic, value: float) -> None:
    """Make V(s) = value for every s (zero output weights, bias = value)."""
    with torch.no_grad():
        agent.critic.out.weight.zero_()
        agent.critic.out.bias.fill_(value)


XS = torch.tensor([[0.1, -0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])


# ---------------------------------------------------------------- TD target
def test_td_target_terminated_does_not_bootstrap():
    assert td_target(r=-1.0, v_next=5.0, terminated=True, gamma=GAMMA) == -1.0


def test_td_target_truncated_bootstraps():
    assert td_target(r=-1.0, v_next=5.0, terminated=False, gamma=GAMMA) == pytest.approx(-1.0 + GAMMA * 5.0)


def test_should_bootstrap_truth_table():
    assert should_bootstrap(terminated=False, truncated=False)
    assert not should_bootstrap(terminated=True, truncated=False)
    assert should_bootstrap(terminated=False, truncated=True)  # time limit: the future still exists
    assert not should_bootstrap(terminated=True, truncated=True)
    # the deliberate bug switch
    assert not should_bootstrap(terminated=False, truncated=True, bootstrap_on_truncation=False)


@pytest.mark.parametrize("bootstrap", [True, False])
def test_update_target_uses_v_next_only_when_bootstrapping(bootstrap):
    agent = make_agent()
    constant_critic(agent, 5.0)
    out = agent.update(XS[:2], action=0, rewards=[-1.0], bootstrap=bootstrap)["stats"]
    expected = -1.0 + GAMMA * 5.0 if bootstrap else -1.0
    assert out["target"] == pytest.approx(expected)
    assert out["delta"] == pytest.approx(expected - 5.0)
    assert out["critic_loss"] == pytest.approx(0.5 * out["delta"] ** 2)


def test_critic_moves_towards_target_and_v_next_gets_no_gradient():
    agent = make_agent(critic_lr=1e-3)
    out = agent.update(XS[:2], action=1, rewards=[1.0], bootstrap=True)["stats"]
    assert abs(out["target"] - out["v_after"]) < abs(out["target"] - out["v"])
    # semi-gradient check: the critic gradient equals −δ ∇V(s_t) (no term from V(s_{t+1}))
    agent2 = make_agent()
    v = agent2.critic(XS[0:1]).squeeze(0)
    v.backward()
    grad_v = [p.grad.clone() for p in agent2.critic.parameters()]
    agent3 = make_agent()
    with torch.no_grad():
        target = 1.0 + GAMMA * float(agent3.critic(XS[1:2]))
    loss = 0.5 * (target - agent3.critic(XS[0:1]).squeeze(0)).pow(2)
    delta = target - agent3.critic(XS[0:1]).item()
    loss.backward()
    for g_loss, g_v in zip((p.grad for p in agent3.critic.parameters()), grad_v):
        torch.testing.assert_close(g_loss, -delta * g_v)


# --------------------------------------------------------------- actor sign
@pytest.mark.parametrize("reward,sign", [(+10.0, +1), (-10.0, -1)])
def test_actor_update_raises_log_prob_when_delta_positive(reward, sign):
    agent = make_agent()
    constant_critic(agent, 0.0)
    out = agent.update(XS[:2], action=2, rewards=[reward], bootstrap=False)["stats"]
    assert math.copysign(1, out["delta"]) == sign
    assert (out["logp_after"] - out["logp"]) * sign > 0
    assert (out["p_a_after"] - out["p_a"]) * sign > 0


def test_actor_loss_gradient_on_logits_is_minus_delta_times_score():
    """∂L_π/∂logits = −δ (onehot(a) − π): descent pushes logit a up when δ > 0."""
    logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    delta, a = 2.0, 1
    loss = -delta * F.log_softmax(logits, -1)[a]
    loss.backward()
    expected = -delta * (F.one_hot(torch.tensor(a), 3).float() - F.softmax(logits.detach(), -1))
    torch.testing.assert_close(logits.grad, expected)
    assert logits.grad[a] < 0  # so a gradient-descent step increases logit a


# ------------------------------------------------------------ n-step / GAE
def test_nstep_target_is_discounted_sum_plus_bootstrap():
    agent = make_agent(n_step=3)
    constant_critic(agent, 2.0)
    out = agent.update(XS, action=0, rewards=[1.0, 2.0, 3.0], bootstrap=True)["stats"]
    assert out["target"] == pytest.approx(1 + GAMMA * 2 + GAMMA**2 * 3 + GAMMA**3 * 2.0)
    out = make_agent(n_step=3)
    constant_critic(out, 2.0)
    s = out.update(XS, action=0, rewards=[1.0, 2.0, 3.0], bootstrap=False)["stats"]
    assert s["target"] == pytest.approx(1 + GAMMA * 2 + GAMMA**2 * 3)


def test_gae_lambda_limits():
    """λ = 1 gives the n-step advantage; λ = 0 gives the one-step δ_t."""
    rewards = [1.0, -2.0, 0.5]
    a1 = make_agent(n_step=3, gae_lambda=1.0)
    a_n = make_agent(n_step=3)
    a0 = make_agent(n_step=3, gae_lambda=0.0)
    s1 = a1.update(XS, 0, rewards, True)["stats"]
    sn = a_n.update(XS, 0, rewards, True)["stats"]
    s0 = a0.update(XS, 0, rewards, True)["stats"]
    assert s1["adv_raw"] == pytest.approx(sn["adv_raw"])
    assert s0["adv_raw"] == pytest.approx(s0["delta"])


def test_entropy_bonus_lowers_actor_loss():
    plain, ent = make_agent(), make_agent(entropy_beta=0.1)
    lp = plain.update(XS[:2], 0, [1.0], True)["stats"]
    le = ent.update(XS[:2], 0, [1.0], True)["stats"]
    assert le["actor_loss"] == pytest.approx(lp["actor_loss"] - 0.1 * lp["entropy"])


# ---------------------------------------------------------------- clipping
def test_joint_clip_scales_both_networks_by_the_same_factor():
    agent = make_agent(grad_clip="joint", grad_clip_norm=0.01)
    constant_critic(agent, 0.0)
    s = agent.update(XS[:2], 0, [100.0], False)["stats"]
    assert s["clip_actor"] == pytest.approx(s["clip_critic"]) and s["clip_actor"] < 1.0
    assert s["critic_share"] == pytest.approx(s["gn_critic"] ** 2 / (s["gn_actor"] ** 2 + s["gn_critic"] ** 2))


def test_separate_clip_and_no_clip():
    s = make_agent(grad_clip="separate", grad_clip_norm=0.01).update(XS[:2], 0, [100.0], False)["stats"]
    assert s["clip_actor"] < 1.0 and s["clip_critic"] < 1.0 and s["clip_actor"] != s["clip_critic"]
    s = make_agent().update(XS[:2], 0, [100.0], False)["stats"]
    assert s["clip_actor"] == 1.0 and s["clip_critic"] == 1.0


# ---------------------------------------------------------------- schedules
def test_lr_factor():
    assert lr_factor("constant", 0.7) == 1.0
    assert lr_factor("linear", 0.25) == pytest.approx(0.75)
    assert lr_factor("cosine", 0.0) == pytest.approx(1.0)
    assert lr_factor("cosine", 0.5) == pytest.approx(0.5)
    assert lr_factor("cosine", 1.0) == pytest.approx(0.0, abs=1e-12)


def test_running_norm_standardises():
    norm = RunningNorm(beta=0.9)
    vals = [norm(x) for x in [10.0, 12.0, 8.0, 11.0, 9.0] * 40]
    assert abs(sum(vals[-50:]) / 50) < 0.5
    assert all(abs(v) < 5 for v in vals[-50:])


def test_actor_weight_scales_policy_gradient_term_only():
    """S&B's I = γ^t factor: the actor term is scaled, the critic is untouched."""
    full, half = make_agent(), make_agent()
    sf = full.update(XS[:2], 1, [3.0], True)["stats"]
    sh = half.update(XS[:2], 1, [3.0], True, actor_weight=0.5)["stats"]
    assert sh["actor_loss"] == pytest.approx(0.5 * sf["actor_loss"])
    assert sh["gn_actor"] == pytest.approx(0.5 * sf["gn_actor"], rel=1e-5)
    assert sh["critic_loss"] == pytest.approx(sf["critic_loss"])


def test_clipped_norms_reported():
    s = make_agent(grad_clip="joint", grad_clip_norm=0.01).update(XS[:2], 0, [100.0], False)["stats"]
    total_after = (s["gn_actor_clipped"] ** 2 + s["gn_critic_clipped"] ** 2) ** 0.5
    assert total_after == pytest.approx(0.01, rel=1e-3)


def test_sgd_step_is_lr_times_gradient():
    agent = make_agent(optimizer="sgd", actor_lr=0.1, critic_lr=0.01)
    s = agent.update(XS[:2], 0, [1.0], True)["stats"]
    assert s["dw_actor"] == pytest.approx(0.1 * s["gn_actor"], rel=1e-4)
    assert s["dw_critic"] == pytest.approx(0.01 * s["gn_critic"], rel=1e-4)


def test_adam_step_size_does_not_shrink_with_the_gradient():
    """The core of the CartPole collapse: Adam's first step is ~lr per weight even when δ is tiny."""
    small, big = make_agent(), make_agent()
    constant_critic(small, 0.0)
    constant_critic(big, 0.0)
    s_small = small.update(XS[:2], 0, [1e-6], False)["stats"]
    s_big = big.update(XS[:2], 0, [1.0], False)["stats"]
    assert s_small["gn_actor"] < 1e-5 * s_big["gn_actor"]  # gradient: a million times smaller
    assert s_small["dw_actor"] > 0.3 * s_big["dw_actor"]  # step: same order of magnitude (ε damps a little)


def test_value_norm_reports_values_in_raw_units():
    agent = make_agent(value_norm=True)
    for _ in range(300):
        s = agent.update(XS[:2], 0, [100.0], False)["stats"]  # target is always 100
    assert s["v_scale"] == pytest.approx(100.0, rel=0.05)  # RMS of the targets
    _, _, v = agent.evaluate(XS[0:1])
    assert float(v[0]) == pytest.approx(s["v_after"], rel=1e-5)
    assert abs(s["v_after"] - 100.0) < abs(s["v"] - 100.0) + 1.0


def test_actor_only_sgd_and_amplification_factor():
    agent = make_agent(actor_optimizer="sgd", actor_lr=0.1)
    assert isinstance(agent.actor_opt, torch.optim.SGD) and isinstance(agent.critic_opt, torch.optim.Adam)
    s = agent.update(XS[:2], 0, [1.0], True)["stats"]
    assert s["amp_actor"] == pytest.approx(1.0, rel=1e-4)  # SGD: ‖Δw‖ = lr·‖∇‖
    assert s["amp_critic"] > 1.0  # Adam's first step: ~lr per weight, whatever the gradient size


def test_larger_adam_eps_damps_tiny_gradient_steps():
    small = {eps: make_agent(adam_eps=eps) for eps in (1e-8, 1e-3)}
    dw = {}
    for eps, agent in small.items():
        constant_critic(agent, 0.0)
        dw[eps] = agent.update(XS[:2], 0, [1e-6], False)["stats"]["dw_actor"]
    assert dw[1e-3] < 0.01 * dw[1e-8]


def test_value_norm_preserve_keeps_v_unchanged_when_sigma_moves():
    agent = make_agent(value_norm=True)
    for _ in range(20):
        agent.update(XS[:2], 0, [1.0], False)  # σ settles near 1
    before = agent.evaluate(XS)[2].clone()
    sigma_old = agent.v_scale
    agent.update_value_scale(500.0)  # a big target moves σ
    assert agent.v_scale > 1.5 * sigma_old
    torch.testing.assert_close(agent.evaluate(XS)[2], before, rtol=1e-5, atol=1e-5)


def test_value_norm_without_preserve_lets_v_jump():
    agent = make_agent(value_norm=True, value_norm_preserve=False)
    for _ in range(20):
        agent.update(XS[:2], 0, [1.0], False)
    before = agent.evaluate(XS)[2].clone()
    agent.update_value_scale(500.0)
    assert not torch.allclose(agent.evaluate(XS)[2], before, rtol=1e-2)
