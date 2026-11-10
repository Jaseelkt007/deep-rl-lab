"""Can the double cart-pole be balanced at all with these pushes? A check BEFORE any RL.

A deterministic hand-written controller (LQR, the textbook linear controller) plays from many
random starts. If even it cannot balance with two constant pushes (±10 N), RL failures would
mean nothing; then we use more force levels.

    uv run python scripts/check_double_cartpole.py

Controllers, from most to least capable:
    continuous   any force the controller asks for (no limit)     -> are physics + controller right?
    clipped      any force between −10 and +10 N
    5 levels     the closest of −10, −5, 0, +5, +10 N
    2 levels     ±10 N: push in the direction the controller asks (the RL setting we want)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.double_cartpole import DoubleCartPoleEnv, LQRController, Params, rk4_step  # noqa: E402

TWO = (-10.0, 10.0)
FIVE = (-10.0, -5.0, 0.0, 5.0, 10.0)


def run(force_fn, seed: int, max_steps: int, start_noise: float, limit_deg: float) -> tuple[int, str]:
    """One episode with any force function. Returns (steps survived, why it ended)."""
    env = DoubleCartPoleEnv(start_noise=start_noise, angle_limit_deg=limit_deg)
    obs, _ = env.reset(seed=seed)
    p, lim = env.p, env.angle_limit
    for t in range(1, max_steps + 1):
        env.q, env.qd = rk4_step(env.q, env.qd, force_fn(obs), p)
        x, t1, t2 = env.q
        obs = np.array([x, env.qd[0], t1, env.qd[1], t2 - t1, env.qd[2] - env.qd[1]])
        if abs(x) > env.X_LIMIT:
            return t, "cart left the track"
        if abs(t1) > lim:
            return t, "lower stick fell"
        if abs(t2) > lim:
            return t, "top stick fell"
    return max_steps, "survived"


def evaluate(name: str, force_fn, starts: int, max_steps: int, start_noise: float, limit_deg: float) -> dict:
    lengths, reasons = [], {}
    for s in range(starts):
        n, why = run(force_fn, 10_000 + s, max_steps, start_noise, limit_deg)
        lengths.append(n)
        reasons[why] = reasons.get(why, 0) + 1
    ok = reasons.get("survived", 0)
    fails = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()) if k != "survived") or "-"
    print(f"  {name:<34} survived {ok:3d}/{starts}  median {int(np.median(lengths)):6d} steps   failures: {fails}")
    return {"name": name, "survived": ok, "starts": starts, "median": float(np.median(lengths)), "reasons": reasons}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starts", type=int, default=100)
    ap.add_argument("--steps", type=int, default=500, help="500 = the training episode length")
    ap.add_argument("--long-steps", type=int, default=10_000)
    ap.add_argument("--start-noise", type=float, default=0.05)
    ap.add_argument("--limit", type=float, default=15.0, help="failure angle of each stick, degrees")
    args = ap.parse_args()
    p = Params()

    # LQR weights: how much the controller cares about (x, θ1, θ2, ẋ, θ̇1, θ̇2) vs the push size
    weightings = {
        "angles": (np.diag([1.0, 100.0, 100.0, 0.1, 1.0, 1.0]), 1.0),
        "angles, cheap push": (np.diag([1.0, 100.0, 100.0, 0.1, 1.0, 1.0]), 0.01),
        "angles + cart": (np.diag([10.0, 100.0, 100.0, 1.0, 1.0, 1.0]), 1.0),
        "gentle": (np.diag([1.0, 10.0, 10.0, 0.1, 0.1, 0.1]), 1.0),
    }
    print(f"start: all 6 numbers in ±{args.start_noise}; fail: |x| > 2.4 or a stick > {args.limit}°; "
          f"{args.starts} starts\n")
    results = {}
    for wname, (Q, R) in weightings.items():
        ctrl = LQRController(p, TWO, Q=Q, R=R)
        two = LQRController(p, TWO, Q=Q, R=R)
        five = LQRController(p, FIVE, Q=Q, R=R)
        print(f"LQR weights '{wname}': gains K = {np.round(ctrl.K, 1).tolist()}")
        for label, fn in [
            ("continuous (no force limit)", ctrl.force),
            ("clipped to ±10 N", lambda o, c=ctrl: float(np.clip(c.force(o), -10, 10))),
            ("5 levels (−10, −5, 0, +5, +10 N)", lambda o, c=five: FIVE[c(o)]),
            ("2 levels (±10 N)", lambda o, c=two: TWO[c(o)]),
        ]:
            results[(wname, label)] = evaluate(label, fn, args.starts, args.steps, args.start_noise, args.limit)
        print()

    # the best weighting for each action set, tested for much longer
    print(f"Long test ({args.long_steps} steps, 20 starts) with the best weighting per action set:")
    for label, levels in [("5 levels (−10, −5, 0, +5, +10 N)", FIVE), ("2 levels (±10 N)", TWO)]:
        best = max(weightings, key=lambda w: (results[(w, label)]["survived"], results[(w, label)]["median"]))
        c = LQRController(p, levels, *weightings[best])
        evaluate(f"{label} · '{best}'", lambda o, c=c, lv=levels: lv[c(o)], 20, args.long_steps,
                 args.start_noise, args.limit)
    print(f"\n(one step = 0.02 s; 500 steps = 10 s; {args.long_steps} steps = {args.long_steps * 0.02 / 60:.1f} min)")
    _ = math


if __name__ == "__main__":
    main()
