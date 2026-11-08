"""Pre-registered experiments: run them, then analyse them with metrics fixed in advance.

    uv run python scripts/experiments.py run 01-cartpole-collapse
    uv run python scripts/experiments.py analyse 01-cartpole-collapse

Hypotheses, predictions and results live in docs/experiments/<name>.md. The predictions
were committed before the runs started.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RunConfig  # noqa: E402
from rl.logger import RunReader, make_run_id, run_to_disk, runs_dir  # noqa: E402

CHECK_LRS = {"actor_lr": 5e-4, "critic_lr": 2.5e-3}  # from scripts/check_cartpole.py

EXPERIMENTS: dict[str, dict] = {
    "01-cartpole-collapse": {
        "base": {"env_id": "CartPole-v1", "num_episodes": 1000, "log_steps_every": 50, **CHECK_LRS},
        "seeds": [10, 11, 12, 13, 14],  # held out: not used to choose the LRs
        "setups": {
            "control": {},
            "value_norm": {"value_norm": True},
            "reward x0.01": {"reward_scale": 0.01},
            # SGD cannot use Adam's LRs (its step is lr·gradient): 20x the check LRs
            "sgd (lr x20)": {"optimizer": "sgd", "actor_lr": 1e-2, "critic_lr": 5e-2},
            "entropy 0.01": {"entropy_beta": 0.01},
        },
    },
    "02-mountaincar-formulation": {
        "base": {"env_id": "MountainCar-v0", "num_episodes": 500, "log_steps_every": 25},
        "seeds": [10, 11, 12, 13, 14],
        "setups": {
            "plain": {},
            "bootstrap off (bug)": {"bootstrap_on_truncation": False},
            "time input + bootstrap off": {"time_input": True, "bootstrap_on_truncation": False},
            "shaping energy": {"shaping": "energy"},
            "n-step 10, GAE 0.95": {"n_step": 10, "gae_lambda": 0.95},
            "entropy 0.01": {"entropy_beta": 0.01},
        },
    },
}


def group_name(exp: str, setup: str) -> str:
    return f"{exp}: {setup}"


def run(exp: str, workers: int) -> None:
    spec = EXPERIMENTS[exp]
    jobs = []
    for setup, over in spec["setups"].items():
        for seed in spec["seeds"]:
            cfg = RunConfig.model_validate({**spec["base"], **over, "seed": seed})
            jobs.append((cfg.model_dump_json(), make_run_id(cfg), group_name(exp, setup)))
    print(f"{exp}: {len(jobs)} runs, {workers} at a time", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rid in pool.map(run_to_disk, *zip(*jobs)):
            print("  saved", rid, flush=True)


# ----------------------------------------------------------------------------- analysis
def load_episodes(run_dir: Path) -> list[dict]:
    with open(run_dir / "metrics.jsonl") as f:
        return [json.loads(line) for line in f if line.startswith('{"type":"episode"')]


def ma(x: np.ndarray, w: int) -> np.ndarray:
    """ma[i] = mean of x[i-w+1 .. i] (NaN for the first w-1)."""
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def first(mask: np.ndarray) -> int | None:
    idx = np.nonzero(mask)[0]
    return int(idx[0]) + 1 if len(idx) else None  # 1-based episode


def cartpole_metrics(eps: list[dict]) -> dict:
    """Metrics fixed before running (docs/experiments/01-cartpole-collapse.md)."""
    r = np.array([e["return"] for e in eps], dtype=float)
    m = ma(r, 100)
    learned_ep = first(m > 400)
    collapsed_ep = None
    if learned_ep is not None:
        after = np.nonzero(m[learned_ep:] < 50)[0]
        collapsed_ep = int(after[0]) + learned_ep + 1 if len(after) else None
    ent = ma(np.array([e["mean_entropy"] for e in eps]), 10)
    sat = ma(np.array([e["hidden"]["critic"][-1]["saturated"] for e in eps]), 10)
    sat_ep, ent_ep = first(sat > 0.9), first(ent < 0.1)
    return {
        "learned": learned_ep is not None, "learned_ep": learned_ep,
        "collapsed": collapsed_ep is not None, "collapsed_ep": collapsed_ep,
        "stays_solved": float(np.mean(r[-200:] > 400)), "best_ma": float(np.nanmax(m)), "final_ma": float(m[-1]),
        "sat90_ep": sat_ep, "entropy01_ep": ent_ep,
        "gap": (ent_ep - sat_ep) if sat_ep is not None and ent_ep is not None else None,
    }


def mountaincar_metrics(eps: list[dict], run_dir: Path) -> dict:
    succ = np.array([e["success"] for e in eps], dtype=float)
    v = np.array([e["mean_v"] for e in eps])
    # Known answers (docs/experiments/02, amendment): checked on the last logged episode
    reader = RunReader(run_dir)
    last_ep = max(reader.step_ranges) if reader.step_ranges else None
    steps = reader.steps(last_ep) if last_ep else []
    full = len(steps) == 200 and not steps[-1]["term"]
    known = {}
    if full:
        known = {
            "V_t0": steps[0]["v"], "V_t199": steps[-1]["v"],
            "delta_t<199": float(np.mean([s["delta"] for s in steps[:-1]])), "delta_t199": steps[-1]["delta"],
        }
    return {
        "first_success": first(succ > 0), "successes": int(succ.sum()),
        "success_last100": float(succ[-100:].mean()),
        "max_pos_last100": float(np.mean([e["max_pos"] for e in eps[-100:]])),
        "mean_abs_delta_last100": float(np.mean([e["mean_abs_delta"] for e in eps[-100:]])),
        "mean_v_last100": float(v[-100:].mean()),
        "entropy_last100": float(np.mean([e["mean_entropy"] for e in eps[-100:]])),
        **known,
    }


def verdict_01(rows: list[dict], control_rows: list[dict]) -> str:
    """Decision rule from docs/experiments/01 (amendment): collapse counted among learned seeds."""
    learned = [r for r in rows if r["learned"]]
    if len(learned) < 3:
        return f"INCONCLUSIVE for collapse (learned on {len(learned)} seeds)"
    collapsed = sum(r["collapsed"] for r in learned)
    stays = float(np.median([r["stays_solved"] for r in learned]))
    c_learned = [r for r in control_rows if r["learned"]]
    c_collapsed = sum(r["collapsed"] for r in c_learned)
    if collapsed <= 1 and stays >= 0.8 and c_collapsed >= 4:
        return f"FIXES (collapsed {collapsed}/{len(learned)}, median stays_solved {stays:.2f}; control {c_collapsed}/{len(c_learned)})"
    return f"does not fix (collapsed {collapsed}/{len(learned)}, median stays_solved {stays:.2f}; control {c_collapsed}/{len(c_learned)})"


def analyse(exp: str) -> dict:
    spec = EXPERIMENTS[exp]
    cartpole = spec["base"]["env_id"] == "CartPole-v1"
    out: dict[str, list[dict]] = {}
    for cfg_path in sorted(runs_dir().glob("*/config.json")):
        meta = json.loads(cfg_path.read_text())
        group = meta.get("group") or ""
        if not group.startswith(exp + ": "):
            continue
        eps = load_episodes(cfg_path.parent)
        if len(eps) < spec["base"]["num_episodes"]:
            continue  # unfinished
        m = cartpole_metrics(eps) if cartpole else mountaincar_metrics(eps, cfg_path.parent)
        out.setdefault(group.split(": ", 1)[1], []).append({"seed": meta["seed"], **m})
    for setup in spec["setups"]:
        rows = sorted(out.get(setup, []), key=lambda d: d["seed"])
        print(f"\n== {setup}  ({len(rows)} seeds)")
        for row in rows:
            print("  " + "  ".join(f"{k}={_fmt(v)}" for k, v in row.items()))
        if rows and cartpole:
            learned = [r for r in rows if r["learned"]]
            print(f"  -> learned {len(learned)}/{len(rows)}, collapsed {sum(r['collapsed'] for r in learned)}/{len(learned)}")
            if setup != "control":
                print("  -> verdict: " + verdict_01(rows, out.get("control", [])))
        elif rows:
            print(f"  -> any success {sum(r['successes'] > 0 for r in rows)}/{len(rows)}, "
                  f"success last100 mean {np.mean([r['success_last100'] for r in rows]):.2f}")
    return out


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "analyse"])
    ap.add_argument("name", choices=list(EXPERIMENTS))
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    run(a.name, a.workers) if a.action == "run" else analyse(a.name)
