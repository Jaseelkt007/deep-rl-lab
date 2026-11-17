"""PPO in JAX with the environment LEFT OUTSIDE JAX (gymnasium vector envs, e.g. Box2D LunarLander).

This is the "keep the physics engine, make only the training fast" option. The policy call and the whole PPO
update (advantages + epochs + minibatches) are compiled JAX. The env cannot be: Box2D is C++ that JAX cannot
trace. So collection stays a Python loop:

    for each of T steps:        Python
        act_jit(actor, obs)     compiled, but called from Python (and its result copied back to NumPy)
        envs.step(actions)      Box2D, one env after another (sync) or in worker processes (async)
    learn_jit(nets, batch)      compiled: one call per batch

Compare jaxrl/ppo.py, where the T-step loop itself is a lax.scan inside the compiled program.
"""

from __future__ import annotations

import time

import gymnasium as gym
import jax
import numpy as np
from gymnasium.vector import AutoresetMode

from .ppo import Config, act, make_learn


def train_hybrid(cfg: Config, env_id: str, encode, seed: int = 0, vec_mode: str = "sync", log=None):
    """Returns (curve of mean finished-episode returns per batch, seconds spent in each part)."""
    cfg.check()
    N, T = cfg.num_envs, cfg.rollout
    envs = gym.make_vec(env_id, num_envs=N, vectorization_mode=vec_mode,
                        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP})
    obs_dim, n_actions = envs.single_observation_space.shape[0], int(envs.single_action_space.n)
    init, learn = make_learn(cfg)
    act_jit, learn_jit = jax.jit(act), jax.jit(learn)
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    nets = init(k, obs_dim, n_actions)

    obs, _ = envs.reset(seed=seed)
    ep_ret = np.zeros(N)
    secs = {"env (Box2D + gymnasium)": 0.0, "policy calls": 0.0, "PPO update": 0.0}
    curve = []
    for u in range(cfg.num_updates):
        buf = {k_: [] for k_ in ("x", "a", "logp", "r", "term", "done", "x_next")}
        finished = []
        for _ in range(T):
            x = encode(obs).astype(np.float32)
            key, k = jax.random.split(key)
            t0 = time.perf_counter()
            a, logp = act_jit(nets[0], x, k)
            a = np.asarray(a)  # wait for the result and copy it to NumPy for gymnasium
            t1 = time.perf_counter()
            obs2, r, term, trunc, info = envs.step(a)
            secs["env (Box2D + gymnasium)"] += time.perf_counter() - t1
            secs["policy calls"] += t1 - t0
            final = obs2.copy()
            if "final_obs" in info:  # SAME_STEP autoreset: obs2 is already the next episode's start
                for i in np.flatnonzero(info["_final_obs"]):
                    final[i] = info["final_obs"][i]
            done = term | trunc
            ep_ret += r
            finished.extend(ep_ret[done].tolist())
            ep_ret[done] = 0.0
            for k_, v in (("x", x), ("a", a), ("logp", np.asarray(logp)), ("r", r.astype(np.float32)),
                          ("term", term), ("done", done), ("x_next", encode(final).astype(np.float32))):
                buf[k_].append(v)
            obs = obs2
        traj = {k_: np.stack(v) for k_, v in buf.items()}  # (T, N, ...)
        key, k = jax.random.split(key)
        t0 = time.perf_counter()
        nets = jax.block_until_ready(learn_jit(nets, traj, k))
        secs["PPO update"] += time.perf_counter() - t0
        curve.append(float(np.mean(finished)) if finished else float("nan"))
        if log and (u + 1) % log == 0:
            print(f"    batch {u + 1}/{cfg.num_updates}: mean return {np.nanmean(curve[-log:]):7.1f}", flush=True)
    envs.close()
    return curve, secs
