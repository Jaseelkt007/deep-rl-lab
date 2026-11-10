"""FastAPI server: REST control + one WebSocket stream.

The live run trains in a background thread; its records go to disk (RunLogger) and to a
Hub, from which every connected browser is fed at ~20 Hz. Seed batches for ablations run
headless in a process pool and only write to disk.

    uv run uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal

import csv as _csv

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rl.config import RunConfig
from rl.explain import ExplainSession, LearnConfig
from rl.logger import RunLogger, RunReader, clean, list_runs, make_run_id, run_to_disk, runs_dir
from rl.trainer import Trainer

TICK = 0.05  # WebSocket push interval (s)
STREAM_LIMIT = 400  # more new steps than this per tick -> send the last full episode instead
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# ----------------------------------------------------------------------------
# Hub: thread-safe hand-over from the training thread to WebSocket clients
# ----------------------------------------------------------------------------
class Hub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.generation = 0
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.generation += 1  # clients see a new generation and resync from scratch
            self.run_start: dict | None = None
            self.history: list[dict] = []  # episode, snapshot, probe, run_end records
            self.cur_steps: list[dict] = []  # steps of the episode in progress
            self.last_steps: list[dict] = []  # steps of the last finished episode
            self.step_seq = 0
            self.recent: deque[tuple[int, dict]] = deque(maxlen=STREAM_LIMIT)

    def emit(self, rec: dict) -> None:
        t = rec["type"]
        if t == "checkpoint":
            return
        rec = clean(rec)
        with self.lock:
            if t == "step":
                self.cur_steps.append(rec)
                self.step_seq += 1
                self.recent.append((self.step_seq, rec))
            elif t == "episode":
                self.last_steps, self.cur_steps = self.cur_steps, []
                self.history.append(rec)
            elif t == "run_start":
                self.run_start = rec
            else:
                self.history.append(rec)


# ----------------------------------------------------------------------------
# The live run
# ----------------------------------------------------------------------------
class LiveRun:
    def __init__(self, cfg: RunConfig, hub: Hub, speed: float | None):
        self.cfg, self.hub, self.speed = cfg, hub, speed
        self.logger = RunLogger(cfg)
        self.run_id = self.logger.run_id
        hub.reset()

        def emit(rec: dict) -> None:
            self.logger(rec)
            hub.emit(rec)

        self.trainer = Trainer(cfg, emit=emit)
        self.state: str = "running"
        self.error: str | None = None
        self._stop = False
        self._step_requests = 0
        self._wake = threading.Event()
        self._rate: deque[float] = deque(maxlen=200)
        self.thread = threading.Thread(target=self._loop, name="trainer", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        next_t = time.perf_counter()
        try:
            while not self._stop:
                if self.state == "paused":
                    if self._step_requests > 0:
                        self._step_requests -= 1
                        self._one_step()
                    else:
                        self._wake.wait(0.1)
                        self._wake.clear()
                    next_t = time.perf_counter()
                    continue
                self._one_step()
                if self.speed:
                    next_t = max(next_t + 1.0 / self.speed, time.perf_counter() - 0.2)
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        except Exception:  # surface training errors in the UI instead of dying silently
            self.state, self.error = "error", traceback.format_exc(limit=5)
            self.logger.close(status="error")

    def _one_step(self) -> None:
        alive = self.trainer.step()
        self._rate.append(time.perf_counter())
        if not alive:
            self.state, self._stop = "finished", True
            if self.trainer.diverged:
                self.error = f"Diverged in episode {self.trainer.episode + 1}: values became inf/NaN. Try smaller learning rates."

    def steps_per_sec(self) -> float:
        if len(self._rate) < 2 or time.perf_counter() - self._rate[-1] > 1.0:
            return 0.0
        return (len(self._rate) - 1) / (self._rate[-1] - self._rate[0] + 1e-9)

    def control(self, action: str) -> None:
        if self.state in ("finished", "error", "stopped"):
            return
        if action == "pause":
            self.state = "paused"
        elif action == "resume":
            self.state = "running"
        elif action == "step":
            self.state = "paused"
            self._step_requests += 1
        self._wake.set()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self.thread.join(timeout=5)
        if self.state not in ("finished", "error"):
            self.state = "stopped"
        self.logger.close(status="stopped")


# ----------------------------------------------------------------------------
# App state
# ----------------------------------------------------------------------------
hub = Hub()
live: LiveRun | None = None
pool: ProcessPoolExecutor | None = None
jobs: dict[str, dict] = {}  # run_id -> {group, future}
readers: dict[str, RunReader] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    runs_dir().mkdir(exist_ok=True)
    yield
    if live:
        live.stop()
    if pool:
        pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Deep RL playground", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def status() -> dict:
    running_jobs = [rid for rid, j in jobs.items() if not j["future"].done()]
    base = {"type": "status", "jobs_running": len(running_jobs), "jobs_total": len(jobs)}
    if live is None:
        return {**base, "state": "idle"}
    return {
        **base, "state": live.state, "run_id": live.run_id, "label": live.cfg.label(),
        "env_id": live.cfg.env_id, "episode": live.trainer.episode, "num_episodes": live.cfg.num_episodes,
        "total_steps": live.trainer.total_steps, "speed": live.speed,
        "steps_per_sec": round(live.steps_per_sec(), 1), "error": live.error,
    }


# ----------------------------------------------------------------------------
# REST
# ----------------------------------------------------------------------------
class StartRequest(BaseModel):
    config: RunConfig = Field(default_factory=RunConfig)
    speed: float | None = Field(None, gt=0)  # steps per second, None = as fast as possible
    paused: bool = False


class ControlRequest(BaseModel):
    action: Literal["pause", "resume", "step", "stop", "reset"]


class SpeedRequest(BaseModel):
    steps_per_sec: float | None = Field(None, gt=0)


class BatchRequest(BaseModel):
    config: RunConfig
    seeds: list[int] = Field(default=[0, 1, 2], min_length=1, max_length=32)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/config/default")
def default_config() -> dict:
    return {"config": RunConfig().model_dump(mode="json"), "schema": RunConfig.model_json_schema()}


@app.get("/api/status")
def get_status() -> dict:
    return status()


@app.post("/api/runs")
def start_run(req: StartRequest) -> dict:
    global live
    if live:
        live.stop()
    live = LiveRun(req.config, hub, req.speed)
    if req.paused:
        live.control("pause")
    return {"run_id": live.run_id, "label": req.config.label()}


@app.post("/api/control")
def control(req: ControlRequest) -> dict:
    global live
    if req.action in ("stop", "reset"):
        if live:
            live.stop()
        if req.action == "reset":
            live = None
            hub.reset()
    elif live:
        live.control(req.action)
    return status()


@app.post("/api/speed")
def set_speed(req: SpeedRequest) -> dict:
    if live:
        live.speed = req.steps_per_sec
    return status()


@app.get("/api/runs")
def runs() -> list[dict]:
    out = list_runs()
    for meta in out:
        job = jobs.get(meta["run_id"])
        if job and not job["future"].done():
            meta["summary"]["status"] = "running (batch)"
    return out


def _reader(run_id: str) -> RunReader:
    root = runs_dir()
    if not RUN_ID_RE.match(run_id) or not (root / run_id / "config.json").exists():
        raise HTTPException(404, f"no run {run_id}")
    r = readers.get(run_id)
    if r is None:
        r = readers[run_id] = RunReader(root / run_id)
    else:
        r.refresh()
    return r


LITE_DROP = ("layers", "hidden")


def _lite(e: dict) -> dict:
    out = {k: v for k, v in e.items() if k not in LITE_DROP}
    hidden = e.get("hidden")
    if hidden:
        out["critic_sat"] = hidden["critic"][-1]["saturated"]
        out["actor_sat"] = hidden["actor"][-1]["saturated"]
    return out


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str, lite: bool = False) -> dict:
    ov = _reader(run_id).overview()
    if lite:  # for the compare page: scalar curves only (+ last-layer saturation for the collapse chain)
        ov["episodes"] = [_lite(e) for e in ov["episodes"]]
        ov["snapshots"] = []
    return ov


@app.get("/api/runs/{run_id}/episodes/{ep}/steps")
def run_steps(run_id: str, ep: int) -> list[dict]:
    return _reader(run_id).steps(ep)


@app.post("/api/batch")
def start_batch(req: BatchRequest) -> dict:
    """Run one setup with several seeds, headless, in parallel processes."""
    global pool
    if pool is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    group = f"batch-{time.strftime('%Y%m%d-%H%M%S')}"
    run_ids = []
    for seed in req.seeds:
        cfg = RunConfig.model_validate(req.config.model_dump() | {"seed": seed})
        run_id = make_run_id(cfg)
        fut: Future = pool.submit(run_to_disk, cfg.model_dump_json(), run_id, group)
        jobs[run_id] = {"group": group, "future": fut}
        run_ids.append(run_id)
    return {"group": group, "run_ids": run_ids}


# ----------------------------------------------------------------------------
# Learn page: batch actor-critic, one fully recorded iteration at a time
# ----------------------------------------------------------------------------
learn_sessions: OrderedDict[str, tuple[ExplainSession, threading.Lock]] = OrderedDict()
LEARN_MAX_SESSIONS = 8


class ForwardRequest(BaseModel):
    n: int = Field(10, ge=1, le=300)


def _learn(sid: str) -> tuple[ExplainSession, threading.Lock]:
    if sid not in learn_sessions:
        raise HTTPException(404, "unknown learn session (the server may have restarted): start a new one")
    learn_sessions.move_to_end(sid)
    return learn_sessions[sid]


@app.post("/api/learn/new")
def learn_new(cfg: LearnConfig) -> dict:
    sid = uuid.uuid4().hex[:10]
    session = ExplainSession(cfg)
    learn_sessions[sid] = (session, threading.Lock())
    while len(learn_sessions) > LEARN_MAX_SESSIONS:
        learn_sessions.popitem(last=False)
    return clean({"session": sid, **session.describe()})


@app.post("/api/learn/{sid}/step")
def learn_step(sid: str) -> dict:
    session, lock = _learn(sid)
    with lock:
        trace = session.iterate()
    return clean({"trace": trace, "summaries": session.summaries})


@app.post("/api/learn/{sid}/forward")
def learn_forward(sid: str, req: ForwardRequest) -> dict:
    session, lock = _learn(sid)
    with lock:
        trace = session.forward(req.n)
    return clean({"trace": trace, "summaries": session.summaries})


@app.get("/api/learn/{sid}/trace/{k}")
def learn_trace(sid: str, k: int) -> dict:
    session, _ = _learn(sid)
    if k not in session.traces:
        raise HTTPException(404, f"iteration {k} was fast-forwarded and not recorded")
    return clean({"trace": session.traces[k], "summaries": session.summaries})


# ----------------------------------------------------------------------------
# Terminal runs: the latest run of each algorithm, saved by scripts/train_cartpole.py in
# runs/cli/<algo>/ (batch_ac, epochs, clip, gae, ppo)
# ----------------------------------------------------------------------------
CLI_NAME_RE = re.compile(r"^(iter\d{4}|good)$")
CLI_ALGO_RE = re.compile(r"^[a-z_]+$")


def _num(v: str):
    if v in ("", "None", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def _cli_dir(algo: str):
    d = runs_dir() / "cli" / algo
    if not CLI_ALGO_RE.match(algo) or not (d / "settings.json").exists():
        raise HTTPException(404, f"no run for '{algo}' yet: uv run python scripts/train_cartpole.py --algo {algo}")
    return d


def _cli_rows(d) -> list[dict]:
    if not (d / "log.csv").exists():
        return []
    with open(d / "log.csv", newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in _csv.DictReader(f)]


def _cli_per_epoch(d) -> list[dict]:
    if not (d / "per_epoch.csv").exists():
        return []
    with open(d / "per_epoch.csv", newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in _csv.DictReader(f)]


def _inner_summary(rows: list[dict], per_epoch: list[dict]) -> dict | None:
    """Inside one batch, averaged: after each epoch, how far the policy is from π_old. Only the
    iterations where the batch was still learning (mean length < 475); after that A ≈ 0 and nothing moves."""
    learning = {r["iter"] for r in rows if isinstance(r.get("ret"), float) and r["ret"] < 475}
    sel = [e for e in per_epoch if e["iter"] in learning]
    if not sel:
        return None
    keys = ("kl", "past", "r_pos", "r_neg", "r_hi", "r_lo")
    out: dict = {"n_iters": len({e["iter"] for e in sel}), "epoch": [], **{k: [] for k in keys}}
    for ep in sorted({e["epoch"] for e in sel}):
        es = [e for e in sel if e["epoch"] == ep]
        out["epoch"].append(ep)
        for k in keys:
            vals = [e[k] for e in es if isinstance(e.get(k), float)]
            out[k].append(sum(vals) / len(vals) if vals else None)
    return out


def _cli_json(path):
    return json.loads(path.read_text()) if path.exists() else None


@app.get("/api/cli/algos")
def cli_algos() -> list[dict]:
    """Every algorithm that has a run: status, settings, scorecard and its speed curve (for comparing)."""
    out = []
    for settings in sorted((runs_dir() / "cli").glob("*/settings.json")):
        d = settings.parent
        rows = _cli_rows(d)
        out.append(clean({
            "algo": d.name, "settings": json.loads(settings.read_text()), "status": _cli_json(d / "status.json") or {},
            "scorecard": _cli_json(d / "scorecard.json"),
            "speed": [{k: r.get(k) for k in ("iter", "env_steps", "ret", "eval", "KL", "clip_frac")} for r in rows],
            "inner": _inner_summary(rows, _cli_per_epoch(d)),
        }))
    return out


def _cli_episodes(d) -> list[dict]:
    out = []
    for p in sorted((d / "episodes").glob("*.json")):
        ep = json.loads(p.read_text())
        out.append({k: ep[k] for k in ("name", "iter", "steps", "terminated")})
    return out


@app.get("/api/cli/run/{algo}")
def cli_run(algo: str) -> dict:
    d = _cli_dir(algo)
    return clean({"algo": algo, "settings": json.loads((d / "settings.json").read_text()),
                  "status": _cli_json(d / "status.json") or {}, "rows": _cli_rows(d), "episodes": _cli_episodes(d),
                  "per_epoch": _cli_per_epoch(d), "scorecard": _cli_json(d / "scorecard.json")})


@app.get("/api/cli/run/{algo}/episodes")
def cli_episode_list(algo: str) -> list[dict]:
    """The recorded episodes of one algorithm (no trajectories), for the side-by-side replay."""
    return _cli_episodes(_cli_dir(algo))


@app.get("/api/cli/run/{algo}/episode/{name}")
def cli_episode(algo: str, name: str) -> dict:
    p = _cli_dir(algo) / "episodes" / f"{name}.json"
    if not CLI_NAME_RE.match(name) or not p.exists():
        raise HTTPException(404, f"no episode {name}")
    return json.loads(p.read_text())


@app.get("/api/cli/run/{algo}/gif/{name}")
def cli_gif(algo: str, name: str):
    p = _cli_dir(algo) / "episodes" / f"{name}.gif"
    if not CLI_NAME_RE.match(name) or not p.exists():
        raise HTTPException(404, f"no gif {name}")
    return FileResponse(p, media_type="image/gif")


# ----------------------------------------------------------------------------
# WebSocket stream
# ----------------------------------------------------------------------------
@app.websocket("/ws")
async def stream(ws: WebSocket) -> None:
    """Server -> client messages:
    reset   {run_start, history, steps}   full state of the live run (on connect / new run)
    records {records}                     new episode / snapshot / probe / run_end records
    steps   {steps, replace}              new step records (replace = a whole episode trace)
    status  {...}                         run state, every 0.5 s
    """
    await ws.accept()
    gen, main_cursor, step_cursor, last_status = -1, 0, 0, 0.0
    try:
        while True:
            msgs: list[dict] = []
            with hub.lock:
                if hub.generation != gen:
                    gen = hub.generation
                    msgs.append({"type": "reset", "run_start": hub.run_start, "history": list(hub.history),
                                 "steps": list(hub.cur_steps) or list(hub.last_steps)})
                    main_cursor, step_cursor = len(hub.history), hub.step_seq
                else:
                    if len(hub.history) > main_cursor:
                        msgs.append({"type": "records", "records": hub.history[main_cursor:]})
                        main_cursor = len(hub.history)
                    n_new = hub.step_seq - step_cursor
                    if 0 < n_new < STREAM_LIMIT:
                        msgs.append({"type": "steps", "replace": False,
                                     "steps": [r for s, r in hub.recent if s > step_cursor]})
                    elif n_new >= STREAM_LIMIT:
                        msgs.append({"type": "steps", "replace": True, "steps": list(hub.last_steps)})
                    step_cursor = hub.step_seq
            now = time.time()
            if now - last_status > 0.5:
                msgs.append(status())
                last_status = now
            for m in msgs:
                await ws.send_text(json.dumps(m, allow_nan=False))
            await asyncio.sleep(TICK)
    except (WebSocketDisconnect, RuntimeError):
        return
