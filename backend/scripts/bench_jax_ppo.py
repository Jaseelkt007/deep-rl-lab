"""PyTorch PPO (rl/ppo.py) vs the same PPO compiled with JAX (jaxrl/ppo.py), on CartPole.

Both use the "ppo" variant settings (2,048-step batches, 10 epochs, minibatch 64, hidden 8, ...).
The JAX side runs with 1, 8, 64, ... parallel envs; the batch stays 2,048 steps, so every run does the
same number of PPO updates on the same amount of data. Only the speed should change.

    uv run python scripts/bench_jax_ppo.py                      # 200k steps, 5 seeds
    uv run python scripts/bench_jax_ppo.py --torch-seeds 0      # skip the slow PyTorch runs
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from jaxrl import cartpole
from jaxrl.ppo import Config, make_train


def run_torch(seed: int, steps: int) -> tuple[float, list[float]]:
    from rl.ppo import VARIANTS, PPOConfig, PPOSession

    cfg = PPOConfig(**{**PPOConfig().model_dump(), **VARIANTS["ppo"], "env_id": "CartPole-v1", "hidden": 8,
                       "seed": seed, "total_steps": steps})
    s = PPOSession(cfg)
    curve = []
    t0 = time.perf_counter()
    while s.env_steps + 2048 <= steps:
        curve.append(s.iterate(record=False)["summary"]["mean_return"])
    return time.perf_counter() - t0, curve


def run_jax(num_envs: int, seeds: int, steps: int):
    cfg = Config(num_envs=num_envs, total_steps=steps)
    train_all_seeds = jax.jit(jax.vmap(make_train(cfg, cartpole)))  # one program trains every seed at once
    keys = jax.random.split(jax.random.PRNGKey(0), seeds)
    t0 = time.perf_counter()
    compiled = train_all_seeds.lower(keys).compile()
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(compiled(keys))
    t_run = time.perf_counter() - t0
    return t_compile, t_run, np.asarray(out["metrics"]["mean_return"])  # (seeds, updates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000, help="env steps per seed")
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the JAX runs (trained together via vmap)")
    ap.add_argument("--torch-seeds", type=int, default=3)
    ap.add_argument("--num-envs", type=int, nargs="+", default=[1, 8, 64, 512])
    args = ap.parse_args()
    print(f"JAX devices: {jax.devices()}   steps per seed: {args.steps:,}\n")
    last = 10  # report the mean return over the last 10 batches

    rows = []
    if args.torch_seeds:
        times, finals = [], []
        for seed in range(args.torch_seeds):
            t, curve = run_torch(seed, args.steps)
            times.append(t)
            finals.append(np.mean(curve[-last:]))
            print(f"  torch seed {seed}: {t:6.1f} s, last-{last} mean return {finals[-1]:6.1f}")
        sps = args.steps / np.mean(times)
        rows.append(("PyTorch, rl/ppo.py, 1 env", 0.0, np.mean(times) * args.torch_seeds, args.torch_seeds,
                     sps, np.mean(finals), np.std(finals)))

    for n in args.num_envs:
        tc, tr, curves = run_jax(n, args.seeds, args.steps)
        finals = curves[:, -last:].mean(axis=1)
        rows.append((f"JAX, {n} env{'s' if n > 1 else ''} x {args.seeds} seeds", tc, tr, args.seeds,
                     args.seeds * args.steps / tr, finals.mean(), finals.std()))
        print(f"  jax num_envs={n}: compile {tc:.1f} s, run {tr:.2f} s")

    base = rows[0][4] if args.torch_seeds else None
    print(f"\n{'setup':34s} {'compile':>8s} {'run':>8s} {'steps/s':>11s} {'vs torch':>9s} {'return (last 10)':>17s}")
    for name, tc, tr, _, sps, m, sd in rows:
        vs = f"{sps / base:8.1f}x" if base else "        -"
        print(f"{name:34s} {tc:7.1f}s {tr:7.1f}s {sps:11,.0f} {vs} {m:10.1f} ± {sd:5.1f}")


if __name__ == "__main__":
    main()
