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


def test_learn_endpoints(client):
    d = client.post("/api/learn/new", json={"env_id": "CartPole-v1", "seed": 1}).json()
    sid = d["session"]
    assert d["n_episodes"] == 5 and "W1" in d["weights"]["critic"]
    t1 = client.post(f"/api/learn/{sid}/step").json()
    assert t1["trace"]["k"] == 1 and len(t1["summaries"]) == 1
    assert len(t1["trace"]["values"]) == len(t1["trace"]["returns"]) == t1["trace"]["summary"]["steps"]
    t = client.post(f"/api/learn/{sid}/forward", json={"n": 3}).json()
    assert t["trace"]["k"] == 4 and len(t["summaries"]) == 4
    assert client.get(f"/api/learn/{sid}/trace/1").json()["trace"]["k"] == 1
    assert client.get(f"/api/learn/{sid}/trace/2").status_code == 404  # fast-forwarded, not recorded
    assert client.post("/api/learn/nope/step").status_code == 404


def test_cli_endpoints(client, tmp_path):
    import json as _json

    d = tmp_path / "cli" / "ppo" / "episodes"
    d.mkdir(parents=True)
    (d.parent / "settings.json").write_text(_json.dumps({"algo": "ppo", "config": {"epochs": 10}}))
    (d.parent / "status.json").write_text(_json.dumps({"state": "finished", "iter": 2, "env_steps": 4096}))
    (d.parent / "log.csv").write_text("iter,ret,env_steps,eval\n1,15.4,2048,9.0\n2,12.6,4096,\n")
    (d.parent / "per_epoch.csv").write_text("iter,epoch,kl,past,r_pos,r_neg,r_max,r_min\n"
                                            "1,1,0.01,0.0,1.1,0.9,1.3,0.7\n1,2,0.03,0.2,1.3,0.8,1.6,0.5\n"
                                            "2,1,0.02,0.1,1.2,0.9,1.4,0.6\n2,2,0.05,0.4,1.5,0.7,2.0,0.4\n")
    (d / "iter0001.json").write_text(_json.dumps({"name": "iter0001", "iter": 1, "steps": 3, "terminated": True,
                                                  "states": [[0, 0, 0, 0]] * 3, "actions": [0, 1, 0], "probs": [[0.5, 0.5]] * 3}))
    algos = client.get("/api/cli/algos").json()
    assert [a["algo"] for a in algos] == ["ppo"] and algos[0]["speed"][1] == {"iter": 2.0, "env_steps": 4096.0, "ret": 12.6, "eval": None,
                                                                         "KL": None, "clip_frac": None}
    inner = algos[0]["inner"]  # averaged over iterations 1 and 2 (both still learning)
    assert inner["epoch"] == [1.0, 2.0] and inner["kl"] == pytest.approx([0.015, 0.04]) and inner["n_iters"] == 2
    r = client.get("/api/cli/run/ppo").json()
    assert r["rows"][0]["env_steps"] == 2048.0 and r["episodes"][0]["steps"] == 3
    assert len(r["per_epoch"]) == 4 and r["per_epoch"][3]["r_max"] == 2.0
    assert client.get("/api/cli/run/ppo/episode/iter0001").json()["actions"] == [0, 1, 0]
    assert client.get("/api/cli/run/ppo/episodes").json() == [{"name": "iter0001", "iter": 1, "steps": 3, "terminated": True,
                                                                 "kind": "greedy"}]
    assert client.get("/api/cli/run/ppo/episode/..%2Fsettings").status_code == 404
    assert client.get("/api/cli/run/nope").status_code == 404


def test_cli_endpoints_per_environment(client, tmp_path):
    import json as _json

    d = tmp_path / "double_cartpole" / "clip"
    (d / "episodes").mkdir(parents=True)
    (d / "settings.json").write_text(_json.dumps({"algo": "clip", "env": "double_cartpole", "config": {}}))
    (d / "log.csv").write_text("iter,ret,env_steps,eval\n1,12.0,60,8.0\n")
    assert [a["algo"] for a in client.get("/api/cli/algos?env=double_cartpole").json()] == ["clip"]
    assert client.get("/api/cli/algos").json() == []  # CartPole runs live in runs/cli/
    assert client.get("/api/cli/run/clip?env=double_cartpole").json()["rows"][0]["ret"] == 12.0
    assert client.get("/api/cli/run/clip").status_code == 404
    assert client.get("/api/cli/algos?env=nope").status_code == 404
    assert client.get("/api/cli/algos?env=double_cartpole_joint").json() == []
    assert client.get("/api/cli/algos?env=mountaincar").json() == []
    assert client.get("/api/cli/algos?env=lunarlander").json() == []
