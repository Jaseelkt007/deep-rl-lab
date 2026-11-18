"""Where the time and the memory go when the JAX LunarLander runs on the GPU vs the CPU.

Three things are measured for each number of parallel landers N, each in its own process (clean memory counters):

    env     the physics only: N landers step with random actions inside one compiled scan (no network, no PPO)
    ppo     the same PPO as rl/ppo.py (batch 2,048, 10 epochs, minibatch 64): the batch stays 2,048 steps,
            so more landers means shorter rollouts, and the 320 small gradient steps per batch stay the same
    scaled  XLRON-style: each lander plays 128 steps, so the batch is N x 128, cut into 32 minibatches per epoch

Memory is read from JAX itself, not nvidia-smi: JAX normally grabs 75% of the GPU at startup, so we turn that
off (XLA_PYTHON_CLIENT_PREALLOCATE=false) and report XLA's plan for the compiled program plus the peak in use.

    uv run python scripts/bench_jax_gpu.py                      # GPU and CPU, all modes (about 20 minutes)
    uv run python scripts/bench_jax_gpu.py --devices cuda --modes env

Measured 2026-11-18 on the laptop (RTX 4050 6 GB, 12-core CPU, machine otherwise idle), steps/s:
    N landers            1      16      64     256    1024    2048    4096
    env, GPU           456   6,077  23,849  80,124 225,521 297,201 362,210
    env, CPU         3,370   9,855               6,409           7,414
    ppo, GPU           115   4,095          55,805         118,999
    ppo, CPU                10,531
    scaled, GPU              3,916          49,844 148,293 218,504
    scaled, CPU             10,489           6,200
References: gymnasium Box2D env.step alone 30,287 (1 core); PyTorch rl/ppo.py on Box2D 3,294.
XLA's planned memory: env 0.1 MB (N=1) -> 27 MB (N=4096), about 6.6 KB per lander; scaled PPO at N=2048: 93 MB.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PLAN = {
    "cuda": {"env": [1, 16, 64, 256, 1024, 2048, 4096], "ppo": [1, 16, 256, 2048], "scaled": [16, 256, 1024, 2048]},
    "cpu": {"env": [1, 16, 256, 2048], "ppo": [16], "scaled": [16, 256]},
}


def measure(mode: str, n: int) -> dict:
    import jax
    import jax.numpy as jnp

    from jaxrl import lunar
    from jaxrl.ppo import Config, make_train

    if mode == "env":
        steps_per_env = 1000

        def run(key):
            k0, key = jax.random.split(key)
            obs, st = jax.vmap(lunar.reset)(jax.random.split(k0, n))

            def body(st, k):
                ka, ks = jax.random.split(k)
                a = jax.random.randint(ka, (n,), 0, 4)
                _, st, r, *_ = jax.vmap(lunar.step)(jax.random.split(ks, n), st, a)
                return st, r.sum()

            return jax.lax.scan(body, st, jax.random.split(key, steps_per_env))[1].sum()

        total = n * steps_per_env
    else:
        if mode == "ppo":
            cfg = Config(num_envs=n, hidden=64, total_steps=16 * 2048)
        else:
            batch = n * 128
            cfg = Config(num_envs=n, hidden=64, batch_steps=batch, minibatch=batch // 32, total_steps=4 * batch)
        run = make_train(cfg, lunar)
        total = cfg.num_updates * cfg.batch_steps

    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    compiled = jax.jit(run).lower(key).compile()
    t_compile = time.perf_counter() - t0
    jax.block_until_ready(compiled(key))  # warm-up call (first-run allocations)
    t0 = time.perf_counter()
    jax.block_until_ready(compiled(key))
    t_run = time.perf_counter() - t0

    mem = compiled.memory_analysis()
    stats = jax.devices()[0].memory_stats() or {}
    return {"device": jax.devices()[0].platform, "mode": mode, "n": n, "steps": total,
            "compile_s": round(t_compile, 1), "run_s": round(t_run, 3), "steps_per_s": round(total / t_run),
            "plan_mb": round((mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes) / 2**20, 1)
            if mem else None,
            "peak_mb": round(stats["peak_bytes_in_use"] / 2**20, 1) if "peak_bytes_in_use" in stats else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    ap.add_argument("--modes", nargs="+", default=["env", "ppo", "scaled"])
    ap.add_argument("--one", nargs=2, metavar=("MODE", "N"), help=argparse.SUPPRESS)  # internal: one measurement
    args = ap.parse_args()
    if args.one:
        print("RESULT " + json.dumps(measure(args.one[0], int(args.one[1]))), flush=True)
        return

    rows = []
    for dev in args.devices:
        for mode in args.modes:
            for n in PLAN[dev][mode]:
                env = {**os.environ, "JAX_PLATFORMS": dev, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                       "TF_CPP_MIN_LOG_LEVEL": "3"}
                out = subprocess.run([sys.executable, "-u", __file__, "--one", mode, str(n)], env=env,
                                     capture_output=True, text=True)
                line = next((x for x in out.stdout.splitlines() if x.startswith("RESULT ")), None)
                if line is None:
                    print(f"{dev} {mode} N={n}: FAILED\n{out.stderr[-1500:]}", flush=True)
                    continue
                r = json.loads(line[7:])
                rows.append(r)
                print(f"{dev:4s} {mode:6s} N={n:5d}  {r['steps_per_s']:>11,} steps/s  compile {r['compile_s']:5.1f}s  "
                      f"XLA plan {r['plan_mb']} MB  peak {r['peak_mb']} MB", flush=True)
    with open("bench_jax_gpu.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
