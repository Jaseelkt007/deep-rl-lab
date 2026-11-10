"""Watch the double cart-pole in a pygame window (no learning, just the setup).

    uv run python scripts/play_double_cartpole.py                   # the hand-written controller (LQR, ±10 N)
    uv run python scripts/play_double_cartpole.py --policy random   # random pushes: falls after ~15 steps
    uv run python scripts/play_double_cartpole.py --policy none     # no push at all: watch the two sticks fall
    uv run python scripts/play_double_cartpole.py --fps 10 --print-every 1   # slow motion, all 6 numbers per step

The window shows the cart (black), the lower stick (brown) and the top stick (violet) on a free hinge.
The terminal prints how each episode ended: time limit, cart left the track, lower or top stick fell.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.double_cartpole import DoubleCartPoleEnv, reference_controller  # noqa: E402


def why_it_ended(env: DoubleCartPoleEnv, terminated: bool) -> str:
    if not terminated:
        return "reached the step limit"
    x, t1, t2 = env.q
    if abs(x) > env.X_LIMIT:
        return "cart left the track"
    return "lower stick fell" if abs(t1) > env.angle_limit else "top stick fell"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["lqr", "random", "none"], default="lqr")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500, help="step limit per episode (500 = training)")
    ap.add_argument("--fps", type=int, default=50, help="50 = real time (one step = 0.02 s); lower = slow motion")
    ap.add_argument("--print-every", type=int, default=0, help="print the 6 state numbers every N steps (0 = never)")
    ap.add_argument("--start-noise", type=float, default=0.05, help="start: all 6 numbers uniform in ± this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    levels = (0.0,) if args.policy == "none" else (-10.0, 10.0)  # "none": a single action that pushes 0 N
    env = DoubleCartPoleEnv(render_mode="human", force_levels=levels, start_noise=args.start_noise)
    env.metadata = {**env.metadata, "render_fps": args.fps}
    ctrl = reference_controller()
    rng = np.random.default_rng(args.seed)
    names = ("x", "ẋ", "θ1", "θ̇1", "φ", "φ̇")
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        print(f"episode {ep + 1}: start " + "  ".join(f"{n} {v:+.3f}" for n, v in zip(names, obs)))
        terminated, steps = False, 0
        while steps < args.steps:
            a = ctrl(obs) if args.policy == "lqr" else int(rng.integers(2)) if args.policy == "random" else 0
            obs, _, terminated, _, info = env.step(a)
            steps += 1
            if args.print_every and steps % args.print_every == 0:
                top = math.degrees(obs[2] + obs[4])
                print(f"  step {steps:4d}  push {info['force']:+5.1f} N   " + "  ".join(f"{n} {v:+.3f}" for n, v in zip(names, obs))
                      + f"   (lower {math.degrees(obs[2]):+5.1f}°, top {top:+5.1f}°)")
            if terminated:
                break
        print(f"episode {ep + 1}: {steps} steps ({why_it_ended(env, terminated)})\n")
    env.close()


if __name__ == "__main__":
    main()
