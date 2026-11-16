"""Watch Walker2d (MuJoCo) in a window (no learning, just the setup).

    uv run python scripts/play_walker2d.py                     # random torques, 3 episodes
    uv run python scripts/play_walker2d.py --policy zero       # no torque at all: it just collapses
    uv run python scripts/play_walker2d.py --slow 4            # 4× slower than real time
    uv run python scripts/play_walker2d.py --print-every 5     # torso height, angle, speed and reward every 5 steps
    uv run python scripts/play_walker2d.py --gif walker.gif    # no window: save a GIF instead
    uv run python scripts/play_walker2d.py --policy trained    # the last trained policy (runs/walker2d/ppo)
    uv run python scripts/play_walker2d.py --policy trained --run ../runs/walker2d/ppo --episodes 5

The robot: a torso and two legs (thigh, leg, foot each), seen from the side, moving in 2D.
State: 17 numbers (torso height and angle, 6 joint angles, and the 9 velocities).
Action: 6 continuous torques in [−1, 1] (hip, knee, ankle of each leg).
Reward per step: forward speed + 1 for staying up − 0.001·(torques²).
The episode ends when it falls (torso height outside 0.8–2.0 or |angle| > 1 rad) or after 1,000 steps.
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["random", "zero", "trained"], default="random")
    ap.add_argument("--run", default=None, help="with --policy trained: the run folder (default runs/walker2d/ppo)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--slow", type=float, default=1.0, help="slow motion factor (1 = real time)")
    ap.add_argument("--print-every", type=int, default=0, help="print the state every N steps (0 = never)")
    ap.add_argument("--gif", default=None, help="save the episodes to this GIF instead of opening a window")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = gym.make("Walker2d-v5", render_mode="rgb_array" if args.gif else "human",
                   **({"width": 480, "height": 360} if args.gif else {}))
    dt = env.unwrapped.dt  # 0.008 s per step
    rng = np.random.default_rng(args.seed)
    frames = []
    session = None
    if args.policy == "trained":
        import sys
        from pathlib import Path

        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rl.logger import runs_dir
        from rl.ppo_continuous import PPOContinuousConfig, PPOContinuousSession

        path = Path(args.run) if args.run else runs_dir() / "walker2d" / "ppo"
        state = torch.load(path / "final_weights.pt", weights_only=False)
        session = PPOContinuousSession(PPOContinuousConfig(**state["config"]))
        session.load_state_dict(state)
        print(f"policy: {path} (the mean action, no sampling)")
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        score, steps = 0.0, 0
        while True:
            if session is not None:
                a = session.act(obs)
            elif args.policy == "random":
                a = rng.uniform(-1, 1, 6).astype(np.float32)
            else:
                a = np.zeros(6, np.float32)
            obs, r, term, trunc, info = env.step(a)
            score += float(r)
            steps += 1
            if args.gif:
                frames.append(env.render())
            elif args.slow > 1:
                time.sleep(dt * (args.slow - 1))
            if args.print_every and steps % args.print_every == 0:
                print(f"  step {steps:4d}  height {obs[0]:.2f}  angle {obs[1]:+.2f}  speed {info['x_velocity']:+.2f}  "
                      f"reward {r:+.2f}  score {score:6.1f}")
            if term or trunc:
                break
        how = "fell" if term else "time limit (still standing after 1,000 steps)"
        print(f"episode {ep + 1}: {steps} steps, score {score:.1f}, moved {info['x_position']:+.2f} m ({how})")
    env.close()
    if args.gif:
        import imageio

        imageio.mimsave(args.gif, frames[::2], duration=2 * dt * args.slow * 1000, loop=0)
        print(f"saved {len(frames[::2])} frames to {args.gif}")


if __name__ == "__main__":
    main()
