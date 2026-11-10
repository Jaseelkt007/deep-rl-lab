"""PPO, built up from our batch actor-critic one switch at a time (rl/explain.py stays as it is).

One iteration:
  1. Collect a batch with the frozen policy: N complete episodes, or exactly N steps
     (then the episode running at the end is CUT and continues in the next batch).
     For every step store the state, the action and log π_old(a|s).
  2. With the critic BEFORE any update, compute per step (backwards along each episode):
        δ_t = r_t + γ·V(s_t+1) − V(s_t)              (V(s_t+1) = 0 if the pole fell)
        A_t = δ_t + γλ·A_t+1                          (stops at the end of the episode / batch)
        target_t = A_t + V(s_t)                       (λ = 1: target = the return G_t, A = G − V)
     These stay fixed for the whole update.
  3. For each of K epochs: shuffle the steps, split them into minibatches, and for each
     minibatch take one gradient step for the critic and one for the actor:
        critic:  L_V = mean ½ (target − V(s))²
        actor:   plain:    L_π = −mean A · log π(a|s)              (ignores that the data came from π_old)
                 ratio:    L_π = −mean r · A,  r = π(a|s) / π_old(a|s)   (corrects for it; r = 1 in epoch 1,
                                                                    so the first step equals plain)
                 clip ε:   L_π = −mean min(r·A, clip(r, 1−ε, 1+ε)·A)    (ratio + a brake)
     After every epoch, measure on the whole batch how far the policy has moved from π_old
     (KL, the ratio r on steps with A > 0 and A < 0, the share of steps past the clip limit).
  4. Throw the batch away.

With epochs = 1, minibatch = whole batch, clip = None, λ = 1 and 5-episode batches this is
the batch actor-critic of rl/explain.py.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from .config import RunConfig
from .diagnostics import explained_variance
from .envs import ObsEncoder, make_env
from .networks import Actor, Critic


class PPOConfig(BaseModel):
    env_id: Literal["CartPole-v1", "MountainCar-v0"] = "CartPole-v1"
    seed: int = 0
    hidden: int = Field(8, ge=2, le=256)
    actor_lr: float = Field(0.01, gt=0)
    critic_lr: float = Field(0.03, gt=0)
    gamma: float = Field(0.99, ge=0, le=1)
    gae_lambda: float = Field(1.0, ge=0, le=1)  # 1 = Monte Carlo returns (our batch actor-critic)
    epochs: int = Field(1, ge=1, le=100)  # passes over the same batch
    minibatch: int | None = Field(None, ge=1)  # None = the whole batch in one gradient step
    ratio: bool = False  # True: −r·A with r = π/π_old (no clip); False: plain −A·log π
    clip: float | None = Field(None, gt=0, lt=1)  # ε: −min(r·A, clip(r, 1−ε, 1+ε)·A); uses the ratio
    batch_episodes: int | None = Field(5, ge=1)  # a batch = N complete episodes ...
    batch_steps: int | None = Field(None, ge=1)  # ... or exactly N steps (cut episodes continue)
    adv_norm: bool = False  # normalise A to mean 0, std 1 per minibatch


# The ladder from our batch actor-critic to PPO: each rung adds one thing to the one before.
VARIANTS: dict[str, dict] = {
    "epochs": {"epochs": 10},
    "ratio": {"epochs": 10, "ratio": True},
    "clip": {"epochs": 10, "ratio": True, "clip": 0.2},
    "gae": {"epochs": 10, "ratio": True, "clip": 0.2, "gae_lambda": 0.95},
    "ppo": {"epochs": 10, "ratio": True, "clip": 0.2, "gae_lambda": 0.95, "batch_episodes": None, "batch_steps": 2048,
            "minibatch": 64, "adv_norm": True, "actor_lr": 1e-3, "critic_lr": 3e-3},
}


def actor_loss(logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float | None,
               ratio: bool = False) -> torch.Tensor:
    """plain: −mean A·log π (our actor-critic).  ratio: −mean r·A, r = π/π_old, no limit.
    clip ε: −mean min(r·A, clip(r, 1−ε, 1+ε)·A): where r has already moved past 1±ε in the
    direction A pushes, the gradient is 0."""
    if clip is None and not ratio:
        return -(adv * logp).mean()
    r = torch.exp(logp - logp_old)
    if clip is None:
        return -(r * adv).mean()
    return -torch.min(r * adv, torch.clamp(r, 1 - clip, 1 + clip) * adv).mean()


@torch.no_grad()
def policy_move(log_probs_old: torch.Tensor, log_probs_new: torch.Tensor, acts: torch.Tensor, adv: torch.Tensor,
                eps: float = 0.2) -> dict:
    """How far the policy has moved from π_old on the batch states (all actions, not only the taken ones)."""
    idx = torch.arange(len(acts))
    kl = float((log_probs_old.exp() * (log_probs_old - log_probs_new)).sum(-1).mean())
    r = torch.exp(log_probs_new[idx, acts] - log_probs_old[idx, acts])
    pos, neg = adv > 0, adv < 0
    past = (pos & (r > 1 + eps)) | (neg & (r < 1 - eps))  # where the clip would stop the push
    return {"kl": kl, "past": float(past.float().mean()),
            "r_pos": float(r[pos].mean()) if pos.any() else None,
            "r_neg": float(r[neg].mean()) if neg.any() else None,
            # the 10 % of steps that moved most in the direction A pushes (the mean hides them:
            # similar states with different actions pull against each other)
            "r_hi": float(torch.quantile(r[pos], 0.9)) if pos.any() else None,
            "r_lo": float(torch.quantile(r[neg], 0.1)) if neg.any() else None,
            "r_max": float(r.max()), "r_min": float(r.min())}


class PPOSession:
    """Same interface as rl.explain.ExplainSession, so the training script treats both alike."""

    def __init__(self, cfg: PPOConfig):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        run_cfg = RunConfig(env_id=cfg.env_id, gamma=cfg.gamma, seed=cfg.seed)
        self.env = make_env(run_cfg)
        self.encoder = ObsEncoder(run_cfg)
        n_actions = int(self.env.action_space.n)
        self.actor = Actor(self.encoder.dim, n_actions, (cfg.hidden,))
        self.critic = Critic(self.encoder.dim, (cfg.hidden,))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.gen = torch.Generator().manual_seed(cfg.seed)  # actions
        self.shuffle_gen = torch.Generator().manual_seed(cfg.seed + 1)  # minibatch order
        self.env.action_space.seed(cfg.seed)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.ep_len = 0  # steps of the episode in progress (it may continue into the next batch)
        self.k = 0
        self.last_mean_return = 0.0

    # ------------------------------------------------------------------ collect
    @torch.no_grad()
    def _collect(self):
        """Play with the frozen policy. Returns per-step arrays and the episode segments."""
        cfg = self.cfg
        X, acts, logp_old, rews, ends, next_obs = [], [], [], [], [], []
        segments, seg = [], {"states": []}
        finished_lengths = []
        n_done_episodes = 0
        while True:
            x = torch.as_tensor(self.encoder(self.obs, 0))
            p = F.softmax(self.actor(x), -1)
            a = int(torch.multinomial(p, 1, generator=self.gen))
            obs2, r, term, trunc, _ = self.env.step(a)
            X.append(x)
            acts.append(a)
            logp_old.append(float(torch.log(p[a])))
            rews.append(float(r))
            seg["states"].append(np.asarray(self.obs, dtype=float).tolist())
            next_obs.append(np.asarray(obs2, dtype=float))
            self.ep_len += 1
            end = "terminated" if term else "truncated" if trunc else None
            if end:
                finished_lengths.append(self.ep_len)
                n_done_episodes += 1
                self.ep_len = 0
                self.obs, _ = self.env.reset()
            else:
                self.obs = obs2
            full = (cfg.batch_steps is not None and len(acts) >= cfg.batch_steps) or (
                cfg.batch_episodes is not None and n_done_episodes >= cfg.batch_episodes)
            if full and not end:
                end = "cut"  # the episode continues in the next batch; its future is filled in with V
            ends.append(end)
            if end:
                segments.append(seg)
                seg = {"states": []}
            if full:
                break
        return X, acts, logp_old, rews, ends, next_obs, segments, finished_lengths

    # ------------------------------------------------------------------ advantages
    @torch.no_grad()
    def _advantages(self, X, rews, ends, next_obs):
        """GAE backwards along each episode, with the critic before any update."""
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        V = self.critic(torch.stack(X)).numpy().astype(np.float64)
        V_next = self.critic(torch.as_tensor(self.encoder(np.array(next_obs), 0))).numpy().astype(np.float64)
        A = np.zeros(len(rews))
        carry = 0.0
        for t in range(len(rews) - 1, -1, -1):
            if ends[t] == "terminated":  # the pole fell: nothing comes after
                nv, carry = 0.0, 0.0
            elif ends[t] in ("truncated", "cut"):  # the future exists: the critic estimates it
                nv, carry = V_next[t], 0.0
            else:
                nv = V_next[t]
            delta = rews[t] + g * nv - V[t]
            carry = delta + g * lam * carry
            A[t] = carry
        return V, A, A + V

    # ------------------------------------------------------------------ iterate
    def iterate(self, record: bool = True) -> dict:
        cfg = self.cfg
        X, acts, logp_old_l, rews, ends, next_obs, segments, finished = self._collect()
        V, A, target = self._advantages(X, rews, ends, next_obs)
        Xb = torch.stack(X)
        Ab = torch.tensor(acts)
        logp_old = torch.tensor(logp_old_l)
        adv_all = torch.tensor(A, dtype=torch.float32)
        tgt_all = torch.tensor(target, dtype=torch.float32)
        n = len(acts)

        with torch.no_grad():
            log_probs0 = F.log_softmax(self.actor(Xb), -1)
            entropy = float(-(log_probs0.exp() * log_probs0).sum(-1).mean())
            critic_loss0 = float(0.5 * (tgt_all - self.critic(Xb)).pow(2).mean())
            actor_loss0 = float(-(adv_all * log_probs0[torch.arange(n), Ab]).mean())

        mb = cfg.minibatch or n
        steps = 0
        # the sign A has in the loss (with normalisation: relative to the batch mean)
        adv_sign = adv_all - adv_all.mean() if cfg.adv_norm else adv_all
        per_epoch = []
        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=self.shuffle_gen)
            for start in range(0, n, mb):
                idx = order[start:start + mb]
                adv = adv_all[idx]
                if cfg.adv_norm and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                # critic: regression to the fixed targets
                v = self.critic(Xb[idx])
                loss_v = 0.5 * (tgt_all[idx] - v).pow(2).mean()
                self.critic_opt.zero_grad()
                loss_v.backward()
                self.critic_opt.step()
                # actor
                logp = F.log_softmax(self.actor(Xb[idx]), -1)[torch.arange(len(idx)), Ab[idx]]
                loss_pi = actor_loss(logp, logp_old[idx], adv, cfg.clip, cfg.ratio)
                self.actor_opt.zero_grad()
                loss_pi.backward()
                self.actor_opt.step()
                steps += 1
            with torch.no_grad():  # after this epoch: how far from π_old?
                per_epoch.append(policy_move(log_probs0, F.log_softmax(self.actor(Xb), -1), Ab, adv_sign, cfg.clip or 0.2))

        with torch.no_grad():
            log_probs1 = F.log_softmax(self.actor(Xb), -1)
            critic_loss1 = float(0.5 * (tgt_all - self.critic(Xb)).pow(2).mean())
            ratio_end = torch.exp(log_probs1[torch.arange(n), Ab] - logp_old)
            clip_frac = float(((ratio_end - 1).abs() > (cfg.clip or 0.2)).float().mean())

        self.k += 1
        if finished:
            self.last_mean_return = float(np.mean(finished))  # CartPole: return = episode length
        summary = {
            "k": self.k, "mean_return": self.last_mean_return, "episodes_finished": len(finished), "steps": n,
            "critic_loss": critic_loss0, "critic_loss_after": critic_loss1, "actor_loss": actor_loss0,
            "entropy": entropy, "ev": explained_variance(V.tolist(), target.tolist()),
            "mean_abs_adv": float(np.abs(A).mean()), "grad_steps": steps, "clip_frac": clip_frac,
            "per_epoch": per_epoch,
        }
        return {
            "k": self.k, "summary": summary, "episodes": segments,
            "values": V.tolist(), "returns": target.tolist(), "advantages": A.tolist(),
            "actor": {"probs_before": log_probs0.exp().tolist(), "probs_after": log_probs1.exp().tolist(),
                      "logp_before": logp_old.tolist(), "logp_after": log_probs1[torch.arange(n), Ab].tolist()},
        }
