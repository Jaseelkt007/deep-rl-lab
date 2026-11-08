import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_RUNS_DIR", str(tmp_path))
    import server

    with TestClient(server.app) as c:
        yield c
        c.post("/api/control", json={"action": "reset"})


def wait_for(client, state, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get("/api/status").json()
        if st["state"] == state:
            return st
        time.sleep(0.05)
    raise AssertionError(f"state never became {state}: {st}")


def test_live_run_finishes_and_can_be_reopened(client):
    cfg = {"env_id": "CartPole-v1", "num_episodes": 4, "snapshot_every": 2, "state_probe_every": 2, "grid_size": 8}
    run_id = client.post("/api/runs", json={"config": cfg}).json()["run_id"]
    st = wait_for(client, "finished")
    assert st["episode"] == 4 and st["run_id"] == run_id

    runs = client.get("/api/runs").json()
    assert runs[0]["run_id"] == run_id and runs[0]["summary"]["status"] == "finished"
    detail = client.get(f"/api/runs/{run_id}").json()
    assert len(detail["episodes"]) == 4 and len(detail["snapshots"]) == 3 and detail["finished"]
    steps = client.get(f"/api/runs/{run_id}/episodes/2/steps").json()
    assert len(steps) == detail["episodes"][1]["length"]
    lite = client.get(f"/api/runs/{run_id}?lite=true").json()
    assert "layers" not in lite["episodes"][0] and lite["snapshots"] == []
    assert 0.0 <= lite["episodes"][0]["critic_sat"] <= 1.0
    assert client.get("/api/runs/../etc").status_code == 404


def test_pause_step_resume(client):
    client.post("/api/runs", json={"config": {"num_episodes": 50}, "paused": True})
    st = wait_for(client, "paused")
    before = st["total_steps"]
    client.post("/api/control", json={"action": "step"})
    time.sleep(0.3)
    assert client.get("/api/status").json()["total_steps"] == before + 1
    client.post("/api/control", json={"action": "resume"})
    time.sleep(0.3)
    assert client.get("/api/status").json()["total_steps"] > before + 1
    client.post("/api/control", json={"action": "stop"})
    assert client.get("/api/status").json()["state"] == "stopped"


def test_websocket_streams_reset_status_and_steps(client):
    client.post("/api/runs", json={"config": {"num_episodes": 50}, "speed": 200})
    seen = set()
    with client.websocket_connect("/ws") as ws:
        t0 = time.time()
        while time.time() - t0 < 5 and not {"reset", "status", "steps"} <= seen:
            seen.add(ws.receive_json()["type"])
    assert {"reset", "status", "steps"} <= seen


def test_batch_runs_seeds_in_background(client):
    cfg = {"env_id": "CartPole-v1", "num_episodes": 2, "snapshot_every": 100, "state_probe_every": 100}
    r = client.post("/api/batch", json={"config": cfg, "seeds": [3, 4]}).json()
    assert len(r["run_ids"]) == 2 and r["group"].startswith("batch-")
    t0 = time.time()
    while time.time() - t0 < 120:
        runs = {m["run_id"]: m for m in client.get("/api/runs").json()}
        if all(rid in runs and runs[rid]["summary"]["status"] == "finished" for rid in r["run_ids"]):
            break
        time.sleep(0.5)
    assert all(runs[rid]["summary"]["status"] == "finished" for rid in r["run_ids"])
    assert sorted(runs[rid]["seed"] for rid in r["run_ids"]) == [3, 4]
    assert len({runs[rid]["config_key"] for rid in r["run_ids"]}) == 1  # same setup, different seeds
