"""Run one setup headless with several seeds and save the runs to runs/ (for ablations).

    uv run python scripts/run_batch.py --env CartPole-v1 --seeds 0 1 2 --set actor_lr=5e-4 critic_lr=2.5e-3
    uv run python scripts/run_batch.py --set entropy_beta=0.01 n_step=5 --episodes 500

Each --set key=value is a RunConfig field (values are parsed as JSON when possible).
Open the results in the web app (saved runs, compare page).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RunConfig  # noqa: E402
from rl.logger import make_run_id, run_to_disk  # noqa: E402


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MountainCar-v0", choices=["MountainCar-v0", "CartPole-v1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--group", default=None, help="name shown on the compare page")
    args = ap.parse_args()

    overrides = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    base = RunConfig(env_id=args.env, num_episodes=args.episodes, **overrides)
    group = args.group or f"batch-{time.strftime('%Y%m%d-%H%M%S')}"
    cfgs = [base.model_copy(update={"seed": s}) for s in args.seeds]
    ids = [make_run_id(c) for c in cfgs]
    print(f"{base.env_id} [{base.label()}] seeds {args.seeds} -> group {group}")
    with ProcessPoolExecutor(max_workers=len(cfgs)) as pool:
        for rid in pool.map(run_to_disk, [c.model_dump_json() for c in cfgs], ids, [group] * len(cfgs)):
            print("  saved runs/" + rid)


if __name__ == "__main__":
    main()
