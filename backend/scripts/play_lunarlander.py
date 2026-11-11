"""Watch LunarLander in a window (no learning, just the setup).

    uv run python scripts/play_lunarlander.py                        # random actions, 3 episodes
    uv run python scripts/play_lunarlander.py --policy heuristic     # Gymnasium's hand-written controller
    uv run python scripts/play_lunarlander.py --policy nothing       # no engine at all: it just falls
    uv run python scripts/play_lunarlander.py --fps 15 --print-every 10   # slow motion, the 8 numbers every 10 steps
    uv run python scripts/play_lunarlander.py --policy heuristic --wind  # with wind and gusts (strength 15, gusts 1.5)
    uv run python scripts/play_lunarlander.py --wind --wind-power 20 --turbulence 2 --gravity -5   # other settings

Prints each episode's score (200 = solved) and how it ended: landed, crashed, flew away or time limit.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

ACTIONS = ("nothing", "left engine", "main engine", "right engine")


def how_it_ended(env, obs, terminated: bool) -> str:
    u = env.unwrapped
    if not terminated:
        return "time limit (still flying after 1,000 steps)"
    if u.game_over:
        return "crashed"
    if abs(float(obs[0])) >= 1.0:
        return "flew away"
    return "landed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["random", "heuristic", "nothing"], default="random")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=50, help="frames per second (50 = real time)")
    ap.add_argument("--print-every", type=int, default=0, help="print the 8 state numbers every N steps (0 = never)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wind", action="store_true", help="turn on wind and random gusts (Gymnasium's own)")
    ap.add_argument("--wind-power", type=float, default=15.0, help="steady wind strength (with --wind)")
    ap.add_argument("--turbulence", type=float, default=1.5, help="random gusts (with --wind)")
    ap.add_argument("--gravity", type=float, default=-10.0, help="gravity (−10 default, −5 = moon-like)")
    args = ap.parse_args()

    kw = {"gravity": args.gravity}
    if args.wind:
        kw.update(enable_wind=True, wind_power=args.wind_power, turbulence_power=args.turbulence)
    env = gym.make("LunarLander-v3", render_mode="human", **kw)
    print(f"settings: {kw}")
    env.unwrapped.metadata["render_fps"] = args.fps
    rng = np.random.default_rng(args.seed)
    names = ("x", "y", "vx", "vy", "angle", "ang.vel", "leg L", "leg R")
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        score, steps = 0.0, 0
        while True:
            if args.policy == "heuristic":
                a = int(heuristic(env, obs))
            elif args.policy == "nothing":
                a = 0
            else:
                a = int(rng.integers(4))
            obs, r, term, trunc, _ = env.step(a)
            score += float(r)
            steps += 1
            if args.print_every and steps % args.print_every == 0:
                print(f"  step {steps:4d}  {ACTIONS[a]:12s}  score {score:7.1f}   "
                      + "  ".join(f"{n} {v:+.2f}" for n, v in zip(names, obs)))
            if term or trunc:
                break
        print(f"episode {ep + 1}: {steps} steps, score {score:.0f} ({how_it_ended(env, obs, term)})")
    env.close()


if __name__ == "__main__":
    main()
