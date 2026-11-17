"""LunarLander (Box2D, unchanged): PyTorch PPO (rl/ppo.py) vs JAX PPO with the env outside JAX (jaxrl/ppo_hybrid.py).

Same "ppo" settings as the playground uses for LunarLander: hidden 64, 2,048-step batches, 10 epochs,
minibatch 64. This answers: how much faster does training get if we keep Box2D and only move PPO to JAX?

    uv run python scripts/bench_lunar_hybrid.py --steps 40960                 # speed only
    uv run python scripts/bench_lunar_hybrid.py --steps 1000000 --learn       # also the learning curve
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from jaxrl.ppo import Config
from jaxrl.ppo_hybrid import train_hybrid
from rl.envs import scale_obs

ENV = "LunarLander-v3"


def encode(obs):
    return scale_obs(obs, ENV)


def run_torch(steps: int, seed: int = 0, log: int = 0):
    from rl.ppo import VARIANTS, PPOConfig, PPOSession

    cfg = PPOConfig(**{**PPOConfig().model_dump(), **VARIANTS["ppo"], "env_id": ENV, "hidden": 64,
                       "seed": seed, "total_steps": steps})
    s = PPOSession(cfg)
    curve, t0 = [], time.perf_counter()
    while s.env_steps + 2048 <= steps:
        sm = s.iterate(record=False)["summary"]
        rets = sm["finished_returns"]
        curve.append(float(np.mean(rets)) if rets else float("nan"))
        if log and len(curve) % log == 0:
            print(f"    batch {len(curve)}: mean return {np.nanmean(curve[-log:]):7.1f}", flush=True)
    return time.perf_counter() - t0, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40_960)
    ap.add_argument("--num-envs", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--modes", nargs="+", default=["sync", "async"])
    ap.add_argument("--learn", action="store_true", help="print learning curves (use with --steps 1000000)")
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()
    log = 50 if args.learn else 0
    rows = []
    if not args.skip_torch:
        print("PyTorch rl/ppo.py, 1 env", flush=True)
        t, curve = run_torch(args.steps, log=log)
        rows.append(("PyTorch rl/ppo.py, 1 env", t, None, curve))
    for mode in args.modes:
        for n in args.num_envs:
            if mode == "async" and n == 1:
                continue
            name = f"JAX PPO + Box2D, {n} env{'s' if n > 1 else ''} ({mode})"
            print(name, flush=True)
            cfg = Config(num_envs=n, total_steps=args.steps, hidden=64)
            t0 = time.perf_counter()
            curve, secs = train_hybrid(cfg, ENV, encode, seed=0, vec_mode=mode, log=log)
            rows.append((name, time.perf_counter() - t0, secs, curve))

    base = rows[0][1] if not args.skip_torch else None
    print(f"\n{'setup':40s} {'time':>7s} {'steps/s':>9s} {'vs torch':>9s} {'last-20 return':>15s}   time split")
    for name, t, secs, curve in rows:
        sps = args.steps / t
        vs = f"{base / t:8.1f}x" if base else "        -"
        if secs:
            secs = {**secs, "Python glue + compile": t - sum(secs.values())}
        split = "  ".join(f"{k} {100 * v / t:.0f}%" for k, v in secs.items()) if secs else ""
        print(f"{name:40s} {t:6.1f}s {sps:9,.0f} {vs} {np.nanmean(curve[-20:]):15.1f}   {split}")


if __name__ == "__main__":
    main()
