"""Known-answer check: plain one-step actor-critic must solve CartPole-v1.

If this passes, the algorithm code is right, and a MountainCar failure is the problem's
fault (sparse reward), not a bug.

    uv run python scripts/check_cartpole.py                 # seeds 0 1 2, 1000 episodes
    uv run python scripts/check_cartpole.py --seeds 0 --episodes 500

Pass criterion: the 100-episode moving average of the return exceeds 400 within
the episode budget, for the majority of seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.config import RunConfig  # noqa: E402
from rl.trainer import Trainer  # noqa: E402


def run_seed(seed: int, episodes: int, window: int, threshold: float) -> dict:
    cfg = RunConfig(
        env_id="CartPole-v1", seed=seed, num_episodes=episodes,
        snapshot_every=10**6, state_probe_every=10**6, checkpoint_every=10**6,
    )
    returns: list[float] = []

    def emit(rec: dict) -> None:
        if rec["type"] == "episode":
            returns.append(rec["return"])

    t0 = time.time()
    Trainer(cfg, emit=emit).run()
    r = np.array(returns)
    ma = np.convolve(r, np.ones(window) / window, mode="valid")  # ma[i] = mean of episodes i..i+window-1
    hit = np.nonzero(ma > threshold)[0]
    return {
        "seed": seed, "label": cfg.label(), "episodes": len(r),
        "first_episode_above": int(hit[0] + window) if len(hit) else None,
        "best_ma": float(ma.max()), "final_ma": float(ma[-1]),
        "seconds": round(time.time() - t0, 1), "returns": r.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=400.0)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON file for the full curves")
    args = ap.parse_args()

    with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
        results = list(pool.map(run_seed, args.seeds, *zip(*[(args.episodes, args.window, args.threshold)] * len(args.seeds))))

    print(f"CartPole-v1, plain one-step actor-critic, {args.episodes} episodes, "
          f"{args.window}-episode moving average, threshold {args.threshold:g}")
    for r in results:
        when = f"first above at episode {r['first_episode_above']}" if r["first_episode_above"] else "never above"
        print(f"  seed {r['seed']}: best MA {r['best_ma']:6.1f}, final MA {r['final_ma']:6.1f}, {when}  ({r['seconds']}s)")
    passed = sum(r["first_episode_above"] is not None for r in results)
    ok = passed * 2 > len(results)
    print(f"{'PASS' if ok else 'FAIL'}: {passed}/{len(results)} seeds reached the threshold")
    if args.out:
        args.out.write_text(json.dumps(results))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
