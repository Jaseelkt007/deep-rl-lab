import gymnasium as gym
import numpy as np
import pytest

from rl.config import RunConfig
from rl.trainer import Trainer


class FakeEnv(gym.Env):
    """MountainCar-shaped env whose episodes always last `length` steps."""

    observation_space = gym.spaces.Box(low=np.array([-1.2, -0.07]), high=np.array([0.6, 0.07]), dtype=np.float64)
    action_space = gym.spaces.Discrete(3)

    def __init__(self, length: int, end: str):
        self.length, self.end, self.k = length, end, 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.k = 0
        return np.array([-0.5, 0.0]), {}

    def step(self, action):
        self.k += 1
        done = self.k >= self.length
        obs = np.array([-0.5 + 0.01 * self.k, 0.001 * self.k])
        return obs, -1.0, done and self.end == "terminated", done and self.end == "truncated", {}


def spy_updates(trainer: Trainer) -> list[tuple[int, bool]]:
    calls: list[tuple[int, bool]] = []
    original = trainer.agent.update

    def spy(xs, action, rewards, bootstrap, actor_weight=1.0):
        calls.append((len(rewards), bootstrap))
        assert xs.shape[0] == len(rewards) + 1
        return original(xs, action, rewards, bootstrap, actor_weight)

    trainer.agent.update = spy
    return calls


@pytest.mark.parametrize("end,boot", [("terminated", False), ("truncated", True)])
def test_nstep_flush_uses_shorter_windows_at_episode_end(end, boot):
    tr = Trainer(RunConfig(n_step=5, num_episodes=3), env=FakeEnv(length=3, end=end))
    calls = spy_updates(tr)
    for _ in range(3):
        tr.step()
    assert calls == [(3, boot), (2, boot), (1, boot)]
    assert tr.episode == 1
    # the next episode starts from scratch: nothing crosses the boundary
    for _ in range(3):
        tr.step()
    assert calls[3:] == [(3, boot), (2, boot), (1, boot)]


def test_nstep_window_inside_an_episode():
    tr = Trainer(RunConfig(n_step=2, num_episodes=2), env=FakeEnv(length=5, end="terminated"))
    calls = spy_updates(tr)
    for _ in range(5):
        tr.step()
    assert calls == [(2, True), (2, True), (2, True), (2, False), (1, False)]


def test_bug_switch_drops_bootstrap_at_truncation():
    tr = Trainer(RunConfig(bootstrap_on_truncation=False, num_episodes=2), env=FakeEnv(length=3, end="truncated"))
    calls = spy_updates(tr)
    for _ in range(3):
        tr.step()
    assert calls == [(1, True), (1, True), (1, False)]


def collect(cfg: RunConfig, env=None) -> list[dict]:
    recs: list[dict] = []
    Trainer(cfg, emit=recs.append, env=env).run()
    return recs


def test_td_target_at_the_real_time_limit_and_at_termination():
    gamma = 0.99
    mc = [r for r in collect(RunConfig(num_episodes=1, gamma=gamma)) if r["type"] == "step"]
    last = mc[-1]
    assert last["trunc"] and not last["term"] and len(mc) == 200
    assert last["target"] == pytest.approx(last["r_train"] + gamma * last["v_next"], rel=1e-6)

    cp = [r for r in collect(RunConfig(env_id="CartPole-v1", num_episodes=1)) if r["type"] == "step"]
    last = cp[-1]
    assert last["term"] and not last["trunc"]
    assert last["target"] == pytest.approx(last["r_train"])


def _comparable(recs):
    return [{k: v for k, v in r.items() if k != "elapsed"} for r in recs if r["type"] != "checkpoint"]


@pytest.mark.parametrize("env_id", ["MountainCar-v0", "CartPole-v1"])
def test_same_seed_gives_identical_records(env_id):
    cfg = RunConfig(env_id=env_id, num_episodes=3, snapshot_every=1, state_probe_every=1)
    a, b = _comparable(collect(cfg)), _comparable(collect(cfg))
    assert a == b
    c = _comparable(collect(cfg.model_copy(update={"seed": 1})))
    assert a != c


def test_scores_use_original_reward_when_shaping():
    recs = collect(RunConfig(num_episodes=1, shaping="energy"))
    steps = [r for r in recs if r["type"] == "step"]
    ep = next(r for r in recs if r["type"] == "episode")
    assert ep["return"] == pytest.approx(sum(s["r_env"] for s in steps)) == -200
    assert any(s["r_shaped"] != s["r_env"] for s in steps)
    assert ep["return_shaped"] != ep["return"]


def test_record_types_and_snapshot_shapes():
    cfg = RunConfig(num_episodes=2, snapshot_every=1, state_probe_every=1, grid_size=10)
    recs = collect(cfg)
    types = [r["type"] for r in recs]
    assert types[0] == "run_start" and types[-1] == "run_end"
    snaps = [r for r in recs if r["type"] == "snapshot"]
    assert [s["ep"] for s in snaps] == [0, 1, 2]
    assert len(snaps[0]["value"]) == 100 and len(snaps[0]["p_right"]) == 100
    probe = next(r for r in recs if r["type"] == "probe")
    assert set(probe["returns"]) == {"policy", "greedy", "fixed", "fixed_greedy"}
    assert 0.0 <= probe["state_dependence"] <= 1.0


def test_reward_scale_changes_training_reward_not_score():
    recs = collect(RunConfig(num_episodes=1, reward_scale=0.01))
    steps = [r for r in recs if r["type"] == "step"]
    ep = next(r for r in recs if r["type"] == "episode")
    assert all(s["r_train"] == pytest.approx(0.01 * s["r_env"]) for s in steps)
    assert ep["return"] == -200 and ep["return_train"] == pytest.approx(-2.0)
    assert steps[-1]["target"] == pytest.approx(steps[-1]["r_train"] + 0.99 * steps[-1]["v_next"], rel=1e-6)
