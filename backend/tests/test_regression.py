"""The default run must stay bit-identical when switches are added (new fields aside)."""

import hashlib
import json

import pytest

from rl.config import RunConfig
from rl.trainer import Trainer

# sha256 of the records of these runs, computed before adam_eps / actor_optimizer / amp_* existed
GOLDEN = {
    "MountainCar-v0": "dad708c2f43e75058bb42126ac4a7318d7e06743605de0d988d1b7a6b185b69b",
    "CartPole-v1": "4bc71ee84317018a8e38ac0feda08cf4a843784b4b242d4f07699fa8b5dab5f3",
}
NEW_CONFIG_KEYS = {"adam_eps", "actor_optimizer", "value_norm_preserve", "actor_adam_eps"}
NEW_RECORD_KEYS = {"amp_actor", "amp_critic", "mean_amp_actor", "mean_amp_critic"}


def digest(cfg: RunConfig) -> str:
    recs: list[dict] = []
    Trainer(cfg, emit=recs.append).run()
    out = []
    for r in recs:
        if r["type"] == "checkpoint":
            continue
        r = {k: v for k, v in r.items() if k != "elapsed" and k not in NEW_RECORD_KEYS}
        if r["type"] == "run_start":
            r["config"] = {k: v for k, v in r["config"].items() if k not in NEW_CONFIG_KEYS}
        out.append(r)
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("env_id,episodes", [("MountainCar-v0", 2), ("CartPole-v1", 3)])
def test_default_run_is_bit_identical(env_id, episodes):
    cfg = RunConfig(env_id=env_id, num_episodes=episodes, snapshot_every=1, state_probe_every=1, grid_size=10)
    assert digest(cfg) == GOLDEN[env_id]


def test_explicit_defaults_match_too():
    """Setting the new switches to their defaults explicitly changes nothing."""
    base = dict(num_episodes=2, snapshot_every=1, state_probe_every=1, grid_size=10)
    assert digest(RunConfig(adam_eps=1e-8, actor_optimizer="adam", **base)) == GOLDEN["MountainCar-v0"]
