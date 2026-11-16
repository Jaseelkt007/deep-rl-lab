"""Off-policy actor-critic for continuous actions: DDPG (then TD3 and SAC, one change at a time).

DDPG (Lillicrap et al. 2015) is DQN for continuous actions. DQN needs max_a Q(s, a), which is a lookup over a few
buttons; with 6 continuous torques it is an optimisation problem. DDPG learns that argmax with a second network:

  actor   μ(s): 17 → 256 → 256 → 6, tanh, so every torque is in [−1, 1]. Deterministic: one action per state.
  critic  Q(s, a): (17 + 6) → 256 → 256 → 1. The ACTION is an input now (DQN: one output per button).
  explore a = clip(μ(s) + N(0, 0.1²), −1, 1): the actor is deterministic, so noise is added by hand.
          The first 25,000 steps use uniform random torques to fill the buffer.
  buffer  the last 1,000,000 transitions (s, a, r, s′, fell); minibatches of 256 drawn at random.
  critic  y = r + γ·(1 − fell)·Q_tgt(s′, μ_tgt(s′))       (the time limit is not a fall: the future exists)
          loss = mean (Q(s, a) − y)²
  actor   loss = −mean Q(s, μ(s)): move the torques in the direction that raises the critic's Q,
          i.e. the gradient ∂Q/∂a flows from the critic into the actor (the critic is not changed by this step)
  targets μ_tgt and Q_tgt: slow copies, θ_tgt ← 0.995·θ_tgt + 0.005·θ after every update ("soft" target networks;
          DQN copied all at once every 1,000 steps)
  one critic update and one actor update per environment step; no observation or reward normalisation

Logged per chunk of 2,048 env steps (the size of a PPO batch, so both line up on the page). The overestimation
check: Q(s, a) next to the discounted return that REALLY followed, on episodes that started and ended inside the
chunk, at steps where the episode either fell (a real end) or had ≥ 459 steps left (the part cut by the time
limit weighs < 1 %), so the comparison is fair.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from .diagnostics import explained_variance
from .networks import MLP


class OffPolicyConfig(BaseModel):
    kind: str = "offpolicy"
    algo: Literal["ddpg"] = "ddpg"
    env_id: str = "Walker2d-v5"
    seed: int = 0
    hidden: int = Field(256, ge=2, le=2048)
    actor_lr: float = Field(3e-4, gt=0)
    critic_lr: float = Field(3e-4, gt=0)
    gamma: float = Field(0.99, ge=0, le=1)
    tau: float = Field(0.005, gt=0, le=1)  # soft target update
    buffer_size: int = Field(1_000_000, ge=1)
    batch_size: int = Field(256, ge=1)
    learning_starts: int = Field(25_000, ge=0)  # uniform random actions before the first update
    expl_noise: float = Field(0.1, ge=0)  # std of the Gaussian added to μ(s), in units of the action range / 2
    chunk_steps: int = Field(2048, ge=1)
    total_steps: int | None = Field(None, ge=1)
    threads: int = Field(1, ge=1, le=64)


class Actor(MLP):
    """μ(s) = tanh(net(s)), scaled to the action range."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, low: np.ndarray, high: np.ndarray):
        super().__init__(obs_dim, (hidden, hidden), act_dim, "relu")
        self.register_buffer("scale", torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        return torch.tanh(super().forward(x)) * self.scale + self.bias


class QNet(MLP):
    """Q(s, a): the state and the action go in, one number comes out."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__(obs_dim + act_dim, (hidden, hidden), 1, "relu")

    def forward(self, s: torch.Tensor, a: torch.Tensor):  # type: ignore[override]
        return super().forward(torch.cat([s, a], -1)).squeeze(-1)


class Buffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, seed: int):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.n, self.i = size, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d) -> None:
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k: int):
        idx = self.rng.integers(0, self.n, k)
        return tuple(torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d))


@torch.no_grad()
def soft_update(target: torch.nn.Module, net: torch.nn.Module, tau: float) -> None:
    for pt, p in zip(target.parameters(), net.parameters()):
        pt.mul_(1 - tau).add_(p, alpha=tau)


class OffPolicySession:
    def __init__(self, cfg: OffPolicyConfig, env_kwargs: dict | None = None):
        torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.env = gym.make(cfg.env_id, **(env_kwargs or {}))
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        self.low, self.high = self.env.action_space.low, self.env.action_space.high
        self.actor = Actor(self.obs_dim, self.act_dim, cfg.hidden, self.low, self.high)
        self.critic = QNet(self.obs_dim, self.act_dim, cfg.hidden)
        self.actor_tgt, self.critic_tgt = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = Buffer(min(cfg.buffer_size, cfg.total_steps or cfg.buffer_size), self.obs_dim, self.act_dim, cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.env.action_space.seed(cfg.seed)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.env_steps = 0
        self.updates = 0
        self.ep_len, self.ep_ret = 0, 0.0
        self.k = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        a = self.actor(torch.as_tensor(np.asarray(obs, np.float32))).numpy()
        if not deterministic:
            a = a + self.rng.normal(0, self.cfg.expl_noise, self.act_dim) * (self.high - self.low) / 2
        return np.clip(a, self.low, self.high)

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "config": self.cfg.model_dump()}

    def load_state_dict(self, st: dict) -> None:
        self.actor.load_state_dict(st["actor"])
        self.critic.load_state_dict(st["critic"])

    # ------------------------------------------------------------------ learning
    def _update(self, log: dict) -> None:
        cfg = self.cfg
        s, a, r, s2, d = self.buffer.sample(cfg.batch_size)
        with torch.no_grad():
            y = r + cfg.gamma * (1 - d) * self.critic_tgt(s2, self.actor_tgt(s2))
        q = self.critic(s, a)
        loss_q = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        loss_q.backward()
        gn_c = float(torch.norm(torch.stack([p.grad.norm() for p in self.critic.parameters()])))
        self.critic_opt.step()
        # actor: raise Q(s, μ(s)); only the actor's weights change (the critic's gradients are discarded)
        mu = self.actor(s)
        loss_pi = -self.critic(s, mu).mean()
        self.actor_opt.zero_grad()
        loss_pi.backward()
        gn_a = float(torch.norm(torch.stack([p.grad.norm() for p in self.actor.parameters()])))
        self.actor_opt.step()
        soft_update(self.critic_tgt, self.critic, cfg.tau)
        soft_update(self.actor_tgt, self.actor, cfg.tau)
        self.updates += 1
        log["loss_q"].append(loss_q.item())
        log["loss_pi"].append(loss_pi.item())
        log["gn_c"].append(gn_c)
        log["gn_a"].append(gn_a)
        log["sat"].append(float((mu.detach().abs() > 0.99 * torch.as_tensor(self.high)).float().mean()))
        if len(log["q"]) < 40:  # a sample of predictions and targets for the explained variance
            log["q"].extend(q.detach().tolist())
            log["y"].extend(y.tolist())

    def iterate(self) -> dict:
        cfg = self.cfg
        log: dict[str, list] = {k: [] for k in ("loss_q", "loss_pi", "gn_c", "gn_a", "sat", "q", "y")}
        finished, episodes = [], []  # episodes: the steps of each episode that started inside this chunk
        cur = {"S": [], "A": [], "R": [], "fresh": self.ep_len == 0}
        abs_a, clipped = [], []
        for t in range(cfg.chunk_steps):
            if self.env_steps < cfg.learning_starts:
                a_raw = self.rng.uniform(self.low, self.high)
            else:
                with torch.no_grad():
                    mu = self.actor(torch.as_tensor(np.asarray(self.obs, np.float32))).numpy()
                a_raw = mu + self.rng.normal(0, cfg.expl_noise, self.act_dim) * (self.high - self.low) / 2
            a = np.clip(a_raw, self.low, self.high).astype(np.float32)
            clipped.append(float(np.mean(np.abs(a_raw) > self.high)))
            abs_a.append(float(np.mean(np.abs(a))))
            obs2, r, term, trunc, info = self.env.step(a)
            self.buffer.add(np.asarray(self.obs, np.float32), a, float(r), np.asarray(obs2, np.float32), float(term))
            cur["S"].append(np.asarray(self.obs, np.float32))
            cur["A"].append(a)
            cur["R"].append(float(r))
            self.env_steps += 1
            self.ep_len += 1
            self.ep_ret += float(r)
            if term or trunc:
                finished.append({"ret": self.ep_ret, "len": self.ep_len, "fell": bool(term),
                                 "x": float(info.get("x_position", float("nan")))})
                episodes.append({**cur, "fell": bool(term)})
                cur = {"S": [], "A": [], "R": [], "fresh": True}
                self.ep_len, self.ep_ret = 0, 0.0
                self.obs, _ = self.env.reset()
            else:
                self.obs = obs2
            if self.env_steps >= cfg.learning_starts and self.buffer.n >= cfg.batch_size:
                self._update(log)

        # overestimation check: Q(s, a) against the discounted return that really followed
        q_pred, g_real = [], []
        horizon = math.ceil(math.log(0.01) / math.log(cfg.gamma)) if cfg.gamma < 1 else 10**9  # 459 for γ = 0.99
        for ep in episodes:
            if not ep["fresh"] or not ep["R"]:
                continue
            n = len(ep["R"])
            G, g = np.zeros(n), 0.0
            for j in range(n - 1, -1, -1):
                g = ep["R"][j] + cfg.gamma * g
                G[j] = g
            keep = [j for j in range(n) if ep["fell"] or n - j >= horizon]
            if keep:
                with torch.no_grad():
                    qs = self.critic(torch.as_tensor(np.array(ep["S"])[keep]), torch.as_tensor(np.array(ep["A"])[keep]))
                q_pred.extend(qs.tolist())
                g_real.extend(G[keep].tolist())
        self.k += 1
        mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
        return {
            "k": self.k, "steps": cfg.chunk_steps, "env_steps": self.env_steps, "finished": finished,
            "critic_loss": mean(log["loss_q"]), "critic_loss_after": None, "actor_loss": mean(log["loss_pi"]),
            "ev": explained_variance(log["q"], log["y"]) if len(log["q"]) > 1 else None,
            "q_mean": mean(log["q"]), "q_pred": mean(q_pred), "g_real": mean(g_real), "n_compared": len(q_pred),
            "gn_critic": mean(log["gn_c"]), "gn_actor": mean(log["gn_a"]), "updates": self.updates,
            "buffer": self.buffer.n, "noise": cfg.expl_noise if self.env_steps > cfg.learning_starts else None,
            "action_abs_mean": mean(abs_a), "action_clipped": mean(clipped), "actor_saturated": mean(log["sat"]),
            "learning": self.env_steps > cfg.learning_starts,
        }
