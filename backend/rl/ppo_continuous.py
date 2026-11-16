"""PPO with a Gaussian policy, for continuous actions (MuJoCo: Walker2d-v5).

The algorithm is the "ppo" rung of rl/ppo.py (2,048-step batches, 10 epochs of minibatches of 64, clip 0.2,
GAE λ = 0.95, advantages normalised per minibatch). What is new for continuous actions:

  actor     the network outputs the mean μ(s) of each action dimension (Walker2d: 6 joint torques); the spread
            σ is one learned number per dimension, the same in every state (log σ starts at 0, so σ = 1)
  sample    a = μ(s) + σ·ε, ε ~ N(0, 1). The environment gets clip(a, −1, 1); log π uses the UNCLIPPED a,
            because that is what the Gaussian produced
  log π     log π(a|s) = Σ_j log N(a_j; μ_j, σ_j²): one number per step, so the ratio π/π_old and the clip
            work exactly as before
  entropy   Σ_j ½·log(2πe·σ_j²): falls as σ shrinks (the policy gets sure)
  KL        between two Gaussians, in closed form (logged: how far one update moved the policy)

And the standard MuJoCo settings (Huang et al. 2022, "The 37 implementation details of PPO"):
  - observation normalisation: running mean and std over every state seen so far, clipped to ±10 (joint
    velocities are ~10× larger than angles). Frozen during the update; saved with the weights.
  - reward scaling: each reward divided by the running std of the discounted return, clipped to ±10, so the
    critic's targets stay around ±10 whatever the reward size. Scores are always logged with the real reward.
  - two 64-unit tanh layers, orthogonal start (hidden gain √2, mean head 0.01, value head 1.0)
  - one Adam for both networks (lr 3e-4, eps 1e-5) on L_π + 0.5·L_V, gradients clipped to norm 0.5,
    learning rate falling linearly to 0 over the run
  - the 1,000-step time limit is not an end: V(s') fills in the future (as in rl/ppo.py)
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from pydantic import BaseModel, Field

from .diagnostics import explained_variance
from .networks import Critic, GaussianActor
from .ppo import actor_loss


class PPOContinuousConfig(BaseModel):
    kind: str = "ppo_continuous"
    env_id: str = "Walker2d-v5"
    seed: int = 0
    hidden: int = Field(64, ge=2, le=1024)
    layers: int = Field(2, ge=1, le=4)
    lr: float = Field(3e-4, gt=0)
    lr_anneal: bool = True  # linearly to 0 over total_steps
    adam_eps: float = Field(1e-5, gt=0)
    gamma: float = Field(0.99, ge=0, le=1)
    gae_lambda: float = Field(0.95, ge=0, le=1)
    clip: float = Field(0.2, gt=0, lt=1)
    epochs: int = Field(10, ge=1)
    batch_steps: int = Field(2048, ge=1)
    minibatch: int = Field(64, ge=1)
    vf_coef: float = Field(0.5, ge=0)
    ent_coef: float = Field(0.0, ge=0)
    max_grad_norm: float | None = Field(0.5, gt=0)
    log_std_init: float = 0.0
    norm_obs: bool = True
    norm_reward: bool = True
    obs_clip: float = Field(10.0, gt=0)
    reward_clip: float = Field(10.0, gt=0)
    total_steps: int | None = Field(None, ge=1)


class RunningMeanStd:
    """Running mean and variance (parallel/Welford update), as in Gymnasium's normalisation wrappers."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape)
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load(self, st: dict) -> None:
        self.mean, self.var, self.count = np.array(st["mean"]), np.array(st["var"]), float(st["count"])


def orthogonal_init(net, out_gain: float) -> None:
    for name, layer in net.linear_layers():
        torch.nn.init.orthogonal_(layer.weight, gain=out_gain if name == "out" else math.sqrt(2))
        torch.nn.init.zeros_(layer.bias)


@torch.no_grad()
def gaussian_kl(mu0: torch.Tensor, ls0: torch.Tensor, mu1: torch.Tensor, ls1: torch.Tensor) -> torch.Tensor:
    """KL(N(μ0, σ0²) ‖ N(μ1, σ1²)) per state, summed over the action dimensions."""
    v0, v1 = (2 * ls0).exp(), (2 * ls1).exp()
    return (ls1 - ls0 + (v0 + (mu0 - mu1) ** 2) / (2 * v1) - 0.5).sum(-1)


class PPOContinuousSession:
    def __init__(self, cfg: PPOContinuousConfig, env_kwargs: dict | None = None):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.env = gym.make(cfg.env_id, **(env_kwargs or {}))
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high
        hidden = (cfg.hidden,) * cfg.layers
        self.actor = GaussianActor(self.obs_dim, self.act_dim, hidden, log_std_init=cfg.log_std_init)
        self.critic = Critic(self.obs_dim, hidden)
        orthogonal_init(self.actor, 0.01)
        orthogonal_init(self.critic, 1.0)
        self.params = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr, eps=cfg.adam_eps)
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.ret_rms = RunningMeanStd(())
        self.disc_ret = 0.0  # running discounted return of the episode in progress (for reward scaling)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.shuffle_gen = torch.Generator().manual_seed(cfg.seed + 1)
        self.env.action_space.seed(cfg.seed)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.env_steps = 0
        self.ep_len, self.ep_ret = 0, 0.0
        self.k = 0

    # ------------------------------------------------------------------ helpers
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if not self.cfg.norm_obs:
            return np.asarray(obs, np.float32)
        z = (np.asarray(obs, np.float64) - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(z, -self.cfg.obs_clip, self.cfg.obs_clip).astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """The action for the environment (clipped to its range). deterministic: the mean μ(s)."""
        d = self.actor.dist(torch.as_tensor(self.normalize(obs)))
        a = d.mean if deterministic else d.sample()
        return np.clip(a.numpy(), self.act_low, self.act_high)

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "obs_rms": self.obs_rms.state(), "ret_rms": self.ret_rms.state(), "config": self.cfg.model_dump()}

    def load_state_dict(self, st: dict) -> None:
        self.actor.load_state_dict(st["actor"])
        self.critic.load_state_dict(st["critic"])
        self.obs_rms.load(st["obs_rms"])
        self.ret_rms.load(st["ret_rms"])

    # ------------------------------------------------------------------ collect
    @torch.no_grad()
    def _collect(self):
        cfg = self.cfg
        X, A, logp, rews, ends, X_next = [], [], [], [], [], []
        finished = []  # one dict per episode that ended in this batch
        for t in range(cfg.batch_steps):
            if cfg.norm_obs:
                self.obs_rms.update(self.obs)
            x = self.normalize(self.obs)
            d = self.actor.dist(torch.as_tensor(x))
            a = d.mean + d.stddev * torch.randn(d.mean.shape, generator=self.gen)
            obs2, r, term, trunc, info = self.env.step(np.clip(a.numpy(), self.act_low, self.act_high))
            self.ep_ret += float(r)
            self.ep_len += 1
            r_train = float(r)
            if cfg.norm_reward:
                self.disc_ret = self.disc_ret * cfg.gamma + float(r)
                self.ret_rms.update(np.array([self.disc_ret]))
                r_train = float(np.clip(r / math.sqrt(self.ret_rms.var + 1e-8), -cfg.reward_clip, cfg.reward_clip))
            X.append(x)
            A.append(a)
            logp.append(float(d.log_prob(a).sum()))
            rews.append(r_train)
            X_next.append(self.normalize(obs2))
            end = "terminated" if term else "truncated" if trunc else None
            if end:
                finished.append({"ret": self.ep_ret, "len": self.ep_len, "fell": bool(term),
                                 "x": float(info.get("x_position", float("nan")))})
                self.ep_len, self.ep_ret, self.disc_ret = 0, 0.0, 0.0
                self.obs, _ = self.env.reset()
            else:
                self.obs = obs2
            if t == cfg.batch_steps - 1 and not end:
                end = "cut"  # the episode continues in the next batch; V(s') fills in its future
            ends.append(end)
        return (torch.as_tensor(np.array(X)), torch.stack(A), torch.tensor(logp), rews, ends,
                torch.as_tensor(np.array(X_next)), finished)

    @torch.no_grad()
    def _advantages(self, X, rews, ends, X_next):
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        V = self.critic(X).numpy().astype(np.float64)
        V_next = self.critic(X_next).numpy().astype(np.float64)
        adv = np.zeros(len(rews))
        carry = 0.0
        for t in range(len(rews) - 1, -1, -1):
            if ends[t] == "terminated":  # fell: nothing comes after
                nv, carry = 0.0, 0.0
            elif ends[t] in ("truncated", "cut"):  # the future exists: the critic estimates it
                nv, carry = V_next[t], 0.0
            else:
                nv = V_next[t]
            carry = rews[t] + g * nv - V[t] + g * lam * carry
            adv[t] = carry
        return V, adv, adv + V

    # ------------------------------------------------------------------ iterate
    def iterate(self) -> dict:
        cfg = self.cfg
        frac = min(1.0, self.env_steps / cfg.total_steps) if cfg.total_steps else 0.0
        lr_now = cfg.lr * (1.0 - frac) if cfg.lr_anneal and cfg.total_steps else cfg.lr
        for group in self.opt.param_groups:
            group["lr"] = lr_now
        X, A, logp_old, rews, ends, X_next, finished = self._collect()
        n = len(rews)
        self.env_steps += n
        V, adv_np, target_np = self._advantages(X, rews, ends, X_next)
        adv_all = torch.tensor(adv_np, dtype=torch.float32)
        tgt_all = torch.tensor(target_np, dtype=torch.float32)
        with torch.no_grad():
            d0 = self.actor.dist(X)
            mu0, ls0 = d0.mean.clone(), self.actor.log_std.detach().clone().expand_as(d0.mean)
            critic_loss0 = float(0.5 * (tgt_all - self.critic(X)).pow(2).mean())
            entropy = float(d0.entropy().sum(-1).mean())

        grad_norms, pi_losses = [], []
        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=self.shuffle_gen)
            for start in range(0, n, cfg.minibatch):
                idx = order[start:start + cfg.minibatch]
                adv = adv_all[idx]
                if len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                d = self.actor.dist(X[idx])
                lp = d.log_prob(A[idx]).sum(-1)
                loss_pi = actor_loss(lp, logp_old[idx], adv, cfg.clip, ratio=True)
                loss_v = 0.5 * (tgt_all[idx] - self.critic(X[idx])).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                self.opt.zero_grad()
                (loss_pi - cfg.ent_coef * ent + cfg.vf_coef * loss_v).backward()
                gn = float(torch.norm(torch.stack([p.grad.norm() for p in self.params if p.grad is not None])))
                if cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.opt.step()
                grad_norms.append(gn)
                pi_losses.append(loss_pi.item())

        with torch.no_grad():
            d1 = self.actor.dist(X)
            ls1 = self.actor.log_std.expand_as(d1.mean)
            kl = float(gaussian_kl(mu0, ls0, d1.mean, ls1).mean())
            ratio = (d1.log_prob(A).sum(-1) - logp_old).exp()
            clip_frac = float(((ratio - 1).abs() > cfg.clip).float().mean())
            critic_loss1 = float(0.5 * (tgt_all - self.critic(X)).pow(2).mean())
        self.k += 1
        return {
            "k": self.k, "steps": n, "env_steps": self.env_steps, "finished": finished,
            "critic_loss": critic_loss0, "critic_loss_after": critic_loss1, "actor_loss": float(np.mean(pi_losses)),
            "entropy": entropy, "std": [float(s) for s in self.actor.log_std.detach().exp()],
            "ev": explained_variance(V.tolist(), target_np.tolist()), "mean_abs_adv": float(np.abs(adv_np).mean()),
            "kl": kl, "clip_frac": clip_frac, "grad_norm": float(np.mean(grad_norms)), "lr": lr_now,
            "action_abs_mean": float(A.abs().mean()), "action_clipped": float((A.abs() > 1).float().mean()),
            "reward_scale": math.sqrt(self.ret_rms.var + 1e-8) if cfg.norm_reward else 1.0,
        }
