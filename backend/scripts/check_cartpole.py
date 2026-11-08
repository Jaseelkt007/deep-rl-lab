"""Known-answer check: plain one-step actor-critic must solve CartPole-v1.

If this passes, the algorithm code is right, and a MountainCar failure is the problem's
fault (sparse reward), not a bug.

    uv run python scripts/check_cartpole.py                 # seeds 0 1 2, 1000 episodes
    uv run python scripts/check_cartpole.py --seeds 0 --episodes 500

Pass criterion: the 100-episode moving average of the return exceeds 400 within
the episode budget, for the majority of seeds. By default it stops a seed as soon as it
passes; --full trains all episodes to show whether the policy stays good or collapses.

The exact-value correctness check (no learning noise) is tests/test_known_values.py.
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

# Learning rates for the check. A sweep (2 seeds each, 1000 episodes) gave:
#   actor 1e-3 / critic 5e-3 (the defaults)  learn to MA ~300-360, then collapse to ~9
#   actor 5e-4 / critic 2.5e-3               pass at episodes 240 and 198   <- used here
#   actor 3e-4 / critic 1e-3                 pass at ~300;  1e-3 / 1e-3: 322 and 682
#   actor 1e-4 / critic 1e-3                 one seed passes, one never learns
# The defaults fail through critic saturation + Adam (see README), not through a bug.
CHECK_ACTOR_LR = 5e-4
CHECK_CRITIC_LR = 2.5e-3


def run_seed(seed: int, episodes: int, window: int, threshold: float, overrides: dict, full: bool) -> dict:
    cfg = RunConfig(
        env_id="CartPole-v1", seed=seed, num_episodes=episodes,
        snapshot_every=10**6, state_probe_every=10**6, checkpoint_every=10**6, **overrides,
    )
    returns: list[float] = []
    first_above: list[int] = []

    def emit(rec: dict) -> None:
        if rec["type"] == "episode":
            returns.append(rec["return"])
            if len(returns) >= window and not first_above and np.mean(returns[-window:]) > threshold:
                first_above.append(len(returns))

    t0 = time.time()
    trainer = Trainer(cfg, emit=emit)
    while trainer.step():
        if first_above and not full:
            break  # passed; --full keeps training to see whether it stays solved
    r = np.array(returns)
    ma = np.convolve(r, np.ones(window) / window, mode="valid") if len(r) >= window else np.array([r.mean()])
    return {
        "seed": seed, "config": overrides, "episodes": len(r),
        "first_episode_above": first_above[0] if first_above else None,
        "best_ma": float(ma.max()), "final_ma": float(ma[-1]),
        "seconds": round(time.time() - t0, 1), "returns": r.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=400.0)
    ap.add_argument("--actor-lr", type=float, default=CHECK_ACTOR_LR)
    ap.add_argument("--critic-lr", type=float, default=CHECK_CRITIC_LR)
    ap.add_argument("--full", action="store_true", help="train all episodes instead of stopping at the pass")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON file for the full curves")
    args = ap.parse_args()

    overrides = {"actor_lr": args.actor_lr, "critic_lr": args.critic_lr}
    n = len(args.seeds)
    with ProcessPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(run_seed, args.seeds, [args.episodes] * n, [args.window] * n,
                                [args.threshold] * n, [overrides] * n, [args.full] * n))

    print(f"CartPole-v1, plain one-step actor-critic (actor lr {args.actor_lr:g}, critic lr {args.critic_lr:g}), "
          f"{args.episodes} episodes, {args.window}-episode moving average, threshold {args.threshold:g}")
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
