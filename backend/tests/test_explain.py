import numpy as np
import pytest

from rl.explain import ExplainSession, LearnConfig


def flat(ep_list, key):
    return [x for ep in ep_list for x in ep[key]]


def test_policy_is_fixed_while_collecting_and_probs_match():
    tr = ExplainSession(LearnConfig(seed=1)).iterate()
    # probabilities recorded while playing == the old actor's output on the same states
    np.testing.assert_allclose(flat(tr["episodes"], "probs"), tr["actor"]["probs_before"], rtol=1e-5, atol=1e-6)


def test_returns_tail_terminated_vs_truncated():
    gamma = 0.99
    cp = ExplainSession(LearnConfig(env_id="CartPole-v1", seed=0)).iterate()
    ep = cp["episodes"][0]
    assert ep["terminated"] and ep["tail"] == 0.0
    n = ep["length"]
    assert cp["returns"][n - 1] == pytest.approx(ep["rewards"][-1])  # G_T-1 = r: nothing after the fall
    mc = ExplainSession(LearnConfig(env_id="MountainCar-v0", seed=0)).iterate()
    ep = mc["episodes"][0]
    assert ep["truncated"] and ep["length"] == 200
    assert mc["returns"][199] == pytest.approx(-1.0 + gamma * ep["final_value"], rel=1e-5)  # bootstrap


def test_advantage_is_return_minus_value():
    tr = ExplainSession(LearnConfig(seed=2)).iterate()
    np.testing.assert_allclose(tr["advantages"], np.array(tr["returns"]) - np.array(tr["values"]), rtol=1e-5, atol=1e-5)


def test_sgd_step_is_minus_lr_times_gradient_and_each_step_helps():
    cfg = LearnConfig(optimizer="sgd", actor_lr=0.01, critic_lr=1e-3, seed=3)
    tr = ExplainSession(cfg).iterate()
    for net, lr in (("critic", cfg.critic_lr), ("actor", cfg.actor_lr)):
        for k, d in tr[net]["delta"].items():
            np.testing.assert_allclose(d, -lr * np.array(tr[net]["grads"][k]), rtol=1e-4, atol=1e-7)
            np.testing.assert_allclose(np.array(tr["weights_before"][net][k]) + np.array(d),
                                       tr[net]["weights_after"][k], rtol=1e-5, atol=1e-6)
    assert tr["critic"]["loss_after"] < tr["critic"]["loss"]  # V moved towards G on this batch
    adv = np.array(tr["advantages"])
    gain = np.array(tr["actor"]["logp_after"]) - np.array(tr["actor"]["logp_before"])
    assert float(np.mean(adv * gain)) > 0  # good actions became more likely, bad ones less


def test_deterministic_and_forward():
    a, b = ExplainSession(LearnConfig(seed=5)), ExplainSession(LearnConfig(seed=5))
    assert a.iterate()["returns"] == b.iterate()["returns"]
    last = a.forward(3)
    assert last["k"] == 4 and len(a.summaries) == 4 and "episodes" in last


def test_describe():
    d = ExplainSession(LearnConfig(env_id="MountainCar-v0")).describe()
    assert d["n_episodes"] == 2 and len(d["maps"]["value"]) == 25 * 25
    assert set(d["weights"]["actor"]) == {"W1", "b1", "W2", "b2"}
    assert np.array(d["weights"]["actor"]["W1"]).shape == (8, 2)
