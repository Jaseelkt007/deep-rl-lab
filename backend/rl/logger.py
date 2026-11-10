"""Run storage: runs/<run_id>/{config.json, metrics.jsonl, summary.json, checkpoints/}.

metrics.jsonl has one JSON record per line, in the order the trainer emitted them, so a
finished run can be reopened and replayed without training again. Step lines come before
their episode's summary line; the reader indexes byte offsets so it only parses the steps
of the episode you ask for.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from .config import RunConfig
from .trainer import Trainer

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def runs_dir() -> Path:
    """Where runs live: $RL_RUNS_DIR if set (tests use a temp dir), else <project>/runs."""
    return Path(os.environ.get("RL_RUNS_DIR", DEFAULT_RUNS_DIR))

ENV_SHORT = {"MountainCar-v0": "mc", "CartPole-v1": "cp", "DoubleCartPole-v0": "dcp", "DoubleCartPoleJoint-v0": "dcpj"}
_STEP_PREFIX = re.compile(rb'^\{"type":"step","ep":(\d+),')


def clean(obj, sig: int = 6):
    """JSON-safe copy: floats rounded to `sig` significant digits, NaN/inf -> null."""
    if isinstance(obj, float | np.floating):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return float(f"{x:.{sig}g}")
    if isinstance(obj, dict):
        return {k: clean(v, sig) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v, sig) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def dumps(rec: dict) -> str:
    return json.dumps(clean(rec), separators=(",", ":"), allow_nan=False)


# Switches added after runs were saved, with the value that reproduces the OLD behaviour when
# the key is missing from a saved config.json (their current defaults may differ).
LEGACY_DEFAULTS = {"value_norm_preserve": False}


def migrate_meta(meta: dict) -> dict:
    """Read-time migration of a saved config.json: fill in missing switches with the value the
    run actually used, and recompute label/helpers from that. The file itself is not changed."""
    cfg = meta.get("config")
    if not cfg:
        return meta
    missing = {k: v for k, v in LEGACY_DEFAULTS.items() if k not in cfg}
    if missing:
        cfg = {**cfg, **missing}
        run_cfg = RunConfig.model_validate(cfg)
        meta = {**meta, "config": run_cfg.model_dump(mode="json"), "label": run_cfg.label(), "helpers": run_cfg.helpers()}
    return meta


def config_key(cfg: RunConfig) -> str:
    """Hash of everything except the seed: runs with the same key are seeds of one setup."""
    d = cfg.model_dump(mode="json")
    d.pop("seed")
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def make_run_id(cfg: RunConfig) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", cfg.label().lower()).strip("-")[:40] or "plain"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{ENV_SHORT[cfg.env_id]}-{slug}-s{cfg.seed}-{config_key(cfg)[:4]}"


class RunLogger:
    """Trainer callback that writes every record to disk."""

    def __init__(self, cfg: RunConfig, run_id: str | None = None, root: Path | None = None, group: str | None = None):
        self.cfg = cfg
        self.run_id = run_id or make_run_id(cfg)
        self.dir = (root or runs_dir()) / self.run_id
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.meta = {
            "run_id": self.run_id, "label": cfg.label(), "helpers": cfg.helpers(), "env_id": cfg.env_id,
            "seed": cfg.seed, "config_key": config_key(cfg), "group": group,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg.model_dump(mode="json"),
        }
        (self.dir / "config.json").write_text(json.dumps(self.meta, indent=2))
        self.f = open(self.dir / "metrics.jsonl", "w", encoding="utf-8")
        self.summary = {"status": "running", "episodes": 0, "total_steps": 0, "recent_return": None,
                        "recent_success": None, "best_return": None}
        self._recent: list[tuple[float, bool]] = []

    def __call__(self, rec: dict) -> None:
        t = rec["type"]
        if t == "checkpoint":
            torch.save(rec["state"], self.dir / "checkpoints" / f"ep{rec['ep']:05d}.pt")
            return
        if t == "step" and (rec["ep"] - 1) % self.cfg.log_steps_every != 0:
            return
        self.f.write(dumps(rec) + "\n")
        if t == "episode":
            self._recent = (self._recent + [(rec["return"], rec["success"])])[-100:]
            best = self.summary["best_return"]
            self.summary.update(
                episodes=rec["ep"], total_steps=rec["total_steps"],
                recent_return=float(np.mean([r for r, _ in self._recent])),
                recent_success=float(np.mean([s for _, s in self._recent])),
                best_return=rec["return"] if best is None else max(best, rec["return"]),
            )
            self.f.flush()
            if rec["ep"] % 10 == 0:
                self._write_summary()
        elif t == "diverged":
            self.summary["diverged_at"] = rec["ep"]
        elif t == "run_end":
            self.summary["status"] = "diverged" if rec.get("diverged") else "finished"
            self.close()

    def _write_summary(self) -> None:
        (self.dir / "summary.json").write_text(json.dumps(clean(self.summary)))

    def close(self, status: str | None = None) -> None:
        if status and self.summary["status"] == "running":
            self.summary["status"] = status
        if not self.f.closed:
            self.f.flush()
            self.f.close()
        self._write_summary()


# ----------------------------------------------------------------------------
# Reading runs back
# ----------------------------------------------------------------------------
def list_runs(root: Path | None = None) -> list[dict]:
    out = []
    for d in sorted((root or runs_dir()).glob("*/config.json"), reverse=True):
        try:
            meta = migrate_meta(json.loads(d.read_text()))
            s = d.parent / "summary.json"
            meta["summary"] = json.loads(s.read_text()) if s.exists() else {"status": "running", "episodes": 0}
            meta.pop("config", None)
            out.append(meta)
        except (OSError, json.JSONDecodeError):
            continue
    return out


class RunReader:
    """Indexes metrics.jsonl once; re-scans only the part appended since (live runs)."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.meta = migrate_meta(json.loads((run_dir / "config.json").read_text()))
        self.path = run_dir / "metrics.jsonl"
        self.offset = 0
        self.run_start: dict | None = None
        self.episodes: list[dict] = []
        self.snapshots: list[dict] = []
        self.probes: list[dict] = []
        self.step_ranges: dict[int, list[int]] = {}  # ep -> [start byte, end byte]
        self.finished = False
        self.diverged: dict | None = None
        self.refresh()

    def refresh(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            pos = self.offset
            for line in f:
                if not line.endswith(b"\n"):
                    break  # partially written line: read it next time
                m = _STEP_PREFIX.match(line)
                if m:
                    ep = int(m.group(1))
                    rng = self.step_ranges.setdefault(ep, [pos, pos])
                    rng[1] = pos + len(line)
                else:
                    rec = json.loads(line)
                    t = rec["type"]
                    if t == "episode":
                        self.episodes.append(rec)
                    elif t == "snapshot":
                        self.snapshots.append(rec)
                    elif t == "probe":
                        self.probes.append(rec)
                    elif t == "run_start":
                        self.run_start = rec
                    elif t == "run_end":
                        self.finished = True
                    elif t == "diverged":
                        self.diverged = rec
                pos += len(line)
            self.offset = pos

    def steps(self, ep: int) -> list[dict]:
        rng = self.step_ranges.get(ep)
        if not rng:
            return []
        with open(self.path, "rb") as f:
            f.seek(rng[0])
            chunk = f.read(rng[1] - rng[0])
        return [json.loads(line) for line in chunk.splitlines() if line]

    def overview(self) -> dict:
        s = self.dir / "summary.json"
        return {
            "meta": self.meta, "run_start": self.run_start, "episodes": self.episodes,
            "snapshots": self.snapshots, "probes": self.probes, "finished": self.finished, "diverged": self.diverged,
            "episodes_with_steps": sorted(self.step_ranges),
            "summary": json.loads(s.read_text()) if s.exists() else None,
        }


def run_to_disk(cfg_json: str, run_id: str | None = None, group: str | None = None, root: str | None = None) -> str:
    """Train one run headless and write it to runs/. Picklable entry point for process pools."""
    cfg = RunConfig.model_validate_json(cfg_json)
    logger = RunLogger(cfg, run_id=run_id, root=Path(root) if root else None, group=group)
    try:
        Trainer(cfg, emit=logger).run()
    finally:
        logger.close(status="stopped")
    return logger.run_id
