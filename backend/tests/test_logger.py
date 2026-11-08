import json
import math

from rl.config import RunConfig
from rl.logger import RunLogger, RunReader, clean, list_runs
from rl.trainer import Trainer


def test_clean_makes_json_safe():
    out = clean({"a": float("nan"), "b": [1.23456789, math.inf], "c": 3})
    assert out == {"a": None, "b": [1.23457, None], "c": 3}
    json.dumps(out, allow_nan=False)


def test_run_roundtrip(tmp_path):
    cfg = RunConfig(num_episodes=3, snapshot_every=1, state_probe_every=2, log_steps_every=2, grid_size=8)
    emitted = []
    logger = RunLogger(cfg, root=tmp_path)

    def emit(rec):
        emitted.append(rec)
        logger(rec)

    Trainer(cfg, emit=emit).run()
    reader = RunReader(tmp_path / logger.run_id)
    ov = reader.overview()
    assert ov["finished"] and len(ov["episodes"]) == 3 and len(ov["snapshots"]) == 4 and len(ov["probes"]) == 1
    assert ov["episodes_with_steps"] == [1, 3]  # log_steps_every = 2
    steps = reader.steps(3)
    assert len(steps) == 200 and steps[0]["t"] == 0 and steps[-1]["trunc"]
    expected = [r for r in emitted if r["type"] == "step" and r["ep"] == 3]
    assert steps[10]["a"] == expected[10]["a"]
    assert reader.steps(2) == []
    assert (tmp_path / logger.run_id / "checkpoints" / "ep00003.pt").exists()
    runs = list_runs(tmp_path)
    assert runs[0]["run_id"] == logger.run_id and runs[0]["summary"]["status"] == "finished"
    assert runs[0]["label"] == "plain"


def test_reader_picks_up_appended_records(tmp_path):
    cfg = RunConfig(num_episodes=2, snapshot_every=100, state_probe_every=100)
    logger = RunLogger(cfg, root=tmp_path)
    tr = Trainer(cfg, emit=logger)
    for _ in range(200):
        tr.step()
    reader = RunReader(tmp_path / logger.run_id)
    assert len(reader.episodes) == 1
    tr.run()
    reader.refresh()
    assert len(reader.episodes) == 2 and reader.finished
