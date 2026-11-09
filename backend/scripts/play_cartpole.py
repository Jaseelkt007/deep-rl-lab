"""Watch a hand-written CartPole policy play in a pygame window.

    uv run python scripts/play_cartpole.py                  # the good policy, 3 episodes
    uv run python scripts/play_cartpole.py --policy random  # compare: random pushes
    uv run python scripts/play_cartpole.py --k 0            # only look at the angle (falls after ~40 steps)

The policy:  push RIGHT if  pole_angle + k * pole_angular_velocity > 0,  else push LEFT.
With k = 0.5 it keeps the pole up for all 500 steps (the maximum).
"""

import argparse

import gymnasium as gym
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["manual", "random"], default="manual")
    ap.add_argument("--k", type=float, default=0.5, help="weight of the angular velocity")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = gym.make("CartPole-v1", render_mode="human")  # opens the window
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        steps = 0
        while True:
            cart_pos, cart_vel, angle, ang_vel = state
            if args.policy == "manual":
                action = 1 if angle + args.k * ang_vel > 0 else 0  # 1 = push right, 0 = push left
            else:
                action = int(rng.integers(2))
            state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                why = "pole fell / cart left the track" if terminated else "reached the 500-step limit"
                print(f"episode {ep + 1}: {steps} steps ({why})")
                break
    env.close()


if __name__ == "__main__":
    main()
