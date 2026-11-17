"""PPO on LunarLander with EVERYTHING in JAX: env (jaxrl/lunar.py on Jax2D), policy and update, one compiled program.

Same PPO settings as the playground's LunarLander "ppo" run (hidden 64, 2,048-step batches, 10 epochs, minibatch 64).

    uv run python scripts/bench_jax_lunar.py --num-envs 1 16 64 256        # speed (steps/s, compile excluded)
    uv run python scripts/bench_jax_lunar.py --learn --steps 1000000      # learning curve, 16 envs x 128 steps
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from jaxrl import lunar
from jaxrl.ppo import Config, make_train


def run(num_envs: int, steps: int, seeds: int):
    cfg = Config(num_envs=num_envs, total_steps=steps, hidden=64)
    f = jax.jit(jax.vmap(make_train(cfg, lunar)))
    keys = jax.random.split(jax.random.PRNGKey(0), seeds)
    t0 = time.perf_counter()
    compiled = f.lower(keys).compile()
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(compiled(keys))
    return tc, time.perf_counter() - t0, np.asarray(out["metrics"]["mean_return"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--steps", type=int, default=20_480, help="env steps per seed")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--learn", action="store_true")
    args = ap.parse_args()
    print(f"JAX devices: {jax.devices()}")
    if args.learn:
        tc, tr, curve = run(16, args.steps, args.seeds)
        print(f"16 envs x {args.seeds} seed(s), {args.steps:,} steps each: compile {tc:.1f} s, run {tr:.1f} s, "
              f"{args.seeds * args.steps / tr:,.0f} steps/s")
        c = np.nanmean(curve, axis=0)
        for i in range(0, len(c), max(1, len(c) // 20)):
            print(f"  after {(i + 1) * 2048:>9,} steps: mean return {c[i]:7.1f}")
        print(f"  last 20 batches: {np.nanmean(c[-20:]):.1f} (per seed: {np.round(np.nanmean(curve[:, -20:], 1), 1)})")
        return
    print(f"{'num_envs':>9s} {'compile':>8s} {'run':>8s} {'steps/s':>10s}")
    for n in args.num_envs:
        tc, tr, _ = run(n, args.steps, args.seeds)
        print(f"{n:9d} {tc:7.1f}s {tr:7.2f}s {args.seeds * args.steps / tr:10,.0f}", flush=True)


if __name__ == "__main__":
    main()
