"""PPO the XLRON / PureJaxRL way: the whole training run is ONE compiled JAX function.

Same algorithm and settings as the "ppo" variant in rl/ppo.py (PyTorch):
    actor and critic: separate MLPs, 1 hidden layer of `hidden` tanh units, PyTorch's default init
    batch of 2,048 steps, 10 epochs, minibatches of 64, clip 0.2, GAE λ 0.95, γ 0.99,
    advantages normalised per minibatch, two Adams (actor 1e-3, critic 3e-3), no entropy bonus, no grad clip,
    V(next state) bootstraps after a time-limit truncation and at the end of the batch.

What is different is only HOW the loop runs. Side by side with rl/ppo.py:

    rl/ppo.py (PyTorch)                                  here (JAX)
    -------------------                                  ----------
    _collect: Python `while True`, one env,              env_step(): the actor sees all N envs at once,
      actor on ONE observation per call                    jax.vmap(env.step) advances all N envs in one call,
                                                           jax.lax.scan repeats it `rollout` times
    _advantages: Python `for t in reversed(...)`         gae(): jax.lax.scan(..., reverse=True)
    for epoch: / for minibatch: in Python                jax.lax.scan over epochs, and inside it over minibatches
    iterate() called from Python per batch               jax.lax.scan over all batches (num_updates)
    -                                                    jax.jit(train): everything above becomes ONE program;
                                                           Python calls it once for the whole run
    -                                                    jax.vmap(train) over random keys: several seeds in one call
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax


@dataclass(frozen=True)
class Config:
    num_envs: int = 1  # parallel environments (XLRON: --NUM_ENVS)
    batch_steps: int = 2048  # steps per PPO batch, summed over all envs; rollout = batch_steps / num_envs
    total_steps: int = 200_000
    epochs: int = 10
    minibatch: int = 64
    hidden: int = 8
    actor_lr: float = 1e-3
    critic_lr: float = 3e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2

    @property
    def rollout(self) -> int:
        return self.batch_steps // self.num_envs

    @property
    def num_updates(self) -> int:
        return self.total_steps // self.batch_steps

    def check(self) -> None:
        if self.batch_steps % self.num_envs or self.batch_steps % self.minibatch:
            raise ValueError("batch_steps must be a multiple of num_envs and of minibatch")


# ---------- networks: plain JAX, a list of (W, b) pairs ----------

def init_mlp(key, sizes):
    """PyTorch nn.Linear's default init: W and b ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))."""
    params = []
    for k, (n_in, n_out) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        kw, kb = jax.random.split(k)
        bound = 1.0 / math.sqrt(n_in)
        params.append((jax.random.uniform(kw, (n_in, n_out), minval=-bound, maxval=bound),
                       jax.random.uniform(kb, (n_out,), minval=-bound, maxval=bound)))
    return params


def mlp(params, x):
    """Works on one input (d,) or any batch (..., d): the matmul handles the extra axes."""
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


# ---------- the PPO update: advantages + epochs x minibatches (shared by both trainers) ----------

def make_learn(cfg: Config):
    """learn(nets, traj, key) -> nets. traj holds (T, N) arrays: x, a, logp, r, term, done, x_next.

    This is rl/ppo.py's _advantages + the epoch/minibatch loops of iterate(), written as scans so it compiles.
    """
    opt_a, opt_c = optax.adam(cfg.actor_lr), optax.adam(cfg.critic_lr)
    critic_v = lambda p, x: mlp(p, x)[..., 0]  # noqa: E731

    def gae(critic, traj):
        """rl/ppo.py _advantages, as a reverse scan over time."""
        v = critic_v(critic, traj["x"])
        v_next = critic_v(critic, traj["x_next"])  # V of the real next state (before any auto-reset)

        def back(carry, t):
            r, term, done, v_t, vn = t
            delta = r + cfg.gamma * vn * (1.0 - term) - v_t  # terminated: nothing comes after
            carry = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done) * carry  # episode ended: restart the sum
            return carry, carry

        n = traj["r"].shape[1]
        _, adv = jax.lax.scan(back, jnp.zeros(n), (traj["r"], traj["term"], traj["done"], v, v_next), reverse=True)
        return adv, adv + v

    def critic_loss(critic, mb):
        return 0.5 * jnp.mean((mb["target"] - critic_v(critic, mb["x"])) ** 2)

    def actor_loss(actor, mb):
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std(ddof=1) + 1e-8)  # torch .std() is ddof=1
        logp = jnp.take_along_axis(jax.nn.log_softmax(mlp(actor, mb["x"])), mb["a"][:, None], axis=1)[:, 0]
        ratio = jnp.exp(logp - mb["logp"])
        return -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv))

    def minibatch_step(nets, mb):
        actor, critic, oa, oc = nets
        upd, oc = opt_c.update(jax.grad(critic_loss)(critic, mb), oc)
        critic = optax.apply_updates(critic, upd)
        upd, oa = opt_a.update(jax.grad(actor_loss)(actor, mb), oa)
        actor = optax.apply_updates(actor, upd)
        return (actor, critic, oa, oc), None

    def epoch(carry, _):
        nets, batch, key = carry
        key, k = jax.random.split(key)
        perm = jax.random.permutation(k, cfg.batch_steps)
        mbs = jax.tree.map(lambda z: z[perm].reshape(-1, cfg.minibatch, *z.shape[1:]), batch)
        nets, _ = jax.lax.scan(minibatch_step, nets, mbs)
        return (nets, batch, key), None

    def learn(nets, traj, key):
        adv, target = gae(nets[1], traj)  # advantages with the critic before any update
        flat = lambda z: z.reshape(cfg.batch_steps, *z.shape[2:])  # noqa: E731  (T, N, ...) -> (T*N, ...)
        batch = {"x": flat(traj["x"]), "a": flat(traj["a"]), "logp": flat(traj["logp"]),
                 "adv": flat(adv), "target": flat(target)}
        (nets, _, _), _ = jax.lax.scan(epoch, (nets, batch, key), None, cfg.epochs)
        return nets

    def init(key, obs_dim, n_actions):
        k_a, k_c = jax.random.split(key)
        actor = init_mlp(k_a, [obs_dim, cfg.hidden, n_actions])
        critic = init_mlp(k_c, [obs_dim, cfg.hidden, 1])
        return actor, critic, opt_a.init(actor), opt_c.init(critic)

    return init, learn


def act(actor, x, key):
    """Sample one action per row of x. Returns (action, log pi(action))."""
    logits = mlp(actor, x)
    a = jax.random.categorical(key, logits)
    return a, jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], axis=1)[:, 0]


# ---------- the fully compiled trainer: env + policy + update in one program ----------

def make_train(cfg: Config, env):
    """env: a module with reset(key), step(key, state, action), encode(obs), OBS_DIM, N_ACTIONS."""
    cfg.check()
    N, T = cfg.num_envs, cfg.rollout
    init, learn = make_learn(cfg)
    env_step_all = jax.vmap(env.step)  # one call steps all N envs
    env_reset_all = jax.vmap(env.reset)

    def env_step(carry, _):
        nets, env_state, obs, ep_ret, key = carry
        key, k_act, k_env = jax.random.split(key, 3)
        x = env.encode(obs)  # (N, obs_dim)
        a, logp = act(nets[0], x, k_act)  # ONE actor call for all N envs
        obs2, env_state2, r, term, trunc, final_obs = env_step_all(jax.random.split(k_env, N), env_state, a)
        done = term | trunc
        ep_ret2 = ep_ret + r
        tr = {"x": x, "a": a, "logp": logp, "r": r, "term": term, "done": done,
              "x_next": env.encode(final_obs), "ep_return": jnp.where(done, ep_ret2, 0.0)}
        return (nets, env_state2, obs2, jnp.where(done, 0.0, ep_ret2), key), tr

    def update(carry, _):
        carry, traj = jax.lax.scan(env_step, carry, None, T)  # 1. collect T steps of all N envs
        nets, env_state, obs, ep_ret, key = carry
        key, k = jax.random.split(key)
        nets = learn(nets, traj, k)  # 2. advantages, 3. epochs x minibatches
        n_done = traj["done"].sum()
        metrics = {"mean_return": traj["ep_return"].sum() / jnp.maximum(n_done, 1), "episodes": n_done}
        return (nets, env_state, obs, ep_ret, key), metrics

    def train(key):
        k_net, k_env, key = jax.random.split(key, 3)
        nets = init(k_net, env.OBS_DIM, env.N_ACTIONS)
        obs, env_state = env_reset_all(jax.random.split(k_env, N))
        carry, metrics = jax.lax.scan(update, (nets, env_state, obs, jnp.zeros(N), key), None, cfg.num_updates)
        return {"actor": carry[0][0], "critic": carry[0][1], "metrics": metrics}

    return train
