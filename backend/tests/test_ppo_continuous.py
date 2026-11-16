import math

import numpy as np
import pytest
import torch

from rl.networks import GaussianActor
from rl.ppo_continuous import PPOContinuousConfig, PPOContinuousSession, RunningMeanStd, gaussian_kl


def test_log_prob_is_the_sum_of_one_gaussian_per_joint():
    torch.manual_seed(0)
    actor = GaussianActor(5, 3, (8,), log_std_init=-0.5)
    x, a = torch.randn(4, 5), torch.randn(4, 3)
    mu, sigma = actor(x), math.exp(-0.5)
    by_hand = (-0.5 * ((a - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(-1)
    assert torch.allclose(actor.dist(x).log_prob(a).sum(-1), by_hand, atol=1e-5)
    # entropy Σ ½ log(2πe σ²) does not depend on the state
    assert actor.dist(x).entropy().sum(-1)[0].item() == pytest.approx(3 * 0.5 * math.log(2 * math.pi * math.e * sigma**2), rel=1e-5)


def test_closed_form_kl_matches_torch():
    torch.manual_seed(1)
    mu0, mu1, ls0, ls1 = torch.randn(7, 6), torch.randn(7, 6), torch.randn(7, 6) * 0.3, torch.randn(7, 6) * 0.3
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu0, ls0.exp()),
                                            torch.distributions.Normal(mu1, ls1.exp())).sum(-1)
    assert torch.allclose(gaussian_kl(mu0, ls0, mu1, ls1), ref, atol=1e-5)
    assert gaussian_kl(mu0, ls0, mu0, ls0).abs().max() < 1e-6


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, (500, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(0), atol=1e-3)
    assert np.allclose(rms.var, data.var(0), atol=1e-2)


def _small(**kw):
    return PPOContinuousConfig(batch_steps=256, minibatch=64, epochs=2, total_steps=1024, **kw)


def test_stored_log_probs_match_the_stored_inputs():
    """The ratio must be exactly 1 before the first update: X is stored already normalised (with the stats of
    that moment) and log π uses the UNCLIPPED sampled action."""
    s = PPOContinuousSession(_small())
    X, A, logp_old, rews, ends, X_next, _ = s._collect()
    with torch.no_grad():
        lp = s.actor.dist(X).log_prob(A).sum(-1)
    assert torch.allclose(lp, logp_old, atol=1e-4)
    assert (A.abs() > 1).any()  # σ = 1 at the start: some samples fall outside [−1, 1] and are only clipped for the env


def test_iterate_and_save_load_give_the_same_policy():
    s = PPOContinuousSession(_small())
    out = s.iterate()
    assert out["steps"] == 256 and out["env_steps"] == 256
    assert out["finished"] and all(e["len"] >= 1 for e in out["finished"])  # random walkers fall fast
    assert out["kl"] >= 0 and 0 <= out["clip_frac"] <= 1
    t = PPOContinuousSession(_small(seed=5))
    t.load_state_dict(s.state_dict())
    obs = np.zeros(17)
    assert np.allclose(s.act(obs), t.act(obs))


def test_learning_rate_falls_linearly_to_zero():
    s = PPOContinuousSession(_small())
    lrs = [s.iterate()["lr"] for _ in range(4)]
    assert lrs == pytest.approx([3e-4, 2.25e-4, 1.5e-4, 0.75e-4])
