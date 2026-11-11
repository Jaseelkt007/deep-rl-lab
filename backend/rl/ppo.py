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

"ppo_1step" (an experiment) is PPO with ONE gradient step per 64-step batch: in that step r = 1, so the clip never acts and the
step is a plain policy-gradient step, −A·log π. It also uses one Adam for both networks (loss L_π + 0.5·L_V,
gradients clipped together to norm 0.5), a learning rate decaying along a cosine to 0, λ rising from 0.9
towards 0.98, advantages normalised over the whole batch and clipped to ±10, the time limit counted as
the end of an episode (no V after it), and orthogonal starting weights (policy head ×0.01).
Adam moves each weight by at most about one learning rate per step, so with 2,048-step batches and a
200k-step budget (98 steps of ≤ 5e-4) the networks barely leave their start; hence 64-step batches
(200k steps → 3,125 gradient steps).
"""

from __future__ import annotations

from typing import Literal

import math

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from .config import RunConfig
from .diagnostics import explained_variance
from .envs import ObsEncoder, make_env
from .networks import Actor, Critic


class PPOConfig(BaseModel):
    env_id: Literal["CartPole-v1", "MountainCar-v0", "DoubleCartPole-v0", "DoubleCartPoleJoint-v0", "LunarLander-v3"] = "CartPole-v1"
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
    shaping: Literal["none", "energy"] = "none"  # MountainCar: + 10·(γΦ(s′) − Φ(s)), Φ = energy (rl/envs.py)
    env_kwargs: dict[str, float | bool] = Field(default_factory=dict)  # gym.make options (LunarLander wind, gravity)
    # one Adam for both networks on L_π + vf_coef·L_V (lr = actor_lr), gradients clipped TOGETHER to max_grad_norm;
    # False: two separate Adams (actor_lr, critic_lr), no clipping
    joint_opt: bool = False
    vf_coef: float = Field(0.5, ge=0)
    max_grad_norm: float | None = Field(None, gt=0)
    adam_eps: float = Field(1e-8, gt=0)  # 1e-8 = PyTorch's default
    lr_decay: Literal["none", "cosine"] = "none"  # cosine: the learning rate falls to 0 over total_steps
    total_steps: int | None = Field(None, ge=1)  # the run's budget in environment steps (for the schedules)
    gae_lambda_end: float | None = Field(None, ge=0, le=1)  # λ rises from gae_lambda towards this over the run
    adv_clip: float | None = Field(None, gt=0)  # after normalising: clip A to ±adv_clip
    trunc_as_end: bool = False  # the time limit counts as the end of the episode: no V after it
    # starting weights. default: PyTorch's. orthogonal: hidden layers gain √2, policy head 0.01 (all actions
    # start almost equally likely), value head 1.0, biases 0
    init: Literal["default", "orthogonal"] = "default"


# The ladder from our batch actor-critic to PPO: each rung adds one thing to the one before.
VARIANTS: dict[str, dict] = {
    "epochs": {"epochs": 10},
    "ratio": {"epochs": 10, "ratio": True},
    "clip": {"epochs": 10, "ratio": True, "clip": 0.2},
    "gae": {"epochs": 10, "ratio": True, "clip": 0.2, "gae_lambda": 0.95},
    "ppo": {"epochs": 10, "ratio": True, "clip": 0.2, "gae_lambda": 0.95, "batch_episodes": None, "batch_steps": 2048,
            "minibatch": 64, "adv_norm": True, "actor_lr": 1e-3, "critic_lr": 3e-3},
    # not a rung on the ladder: PPO with a single gradient step per batch, and its own optimiser settings
    "ppo_1step": {"epochs": 1, "ratio": True, "clip": 0.2, "gae_lambda": 0.9, "gae_lambda_end": 0.98,
                  "batch_episodes": None, "batch_steps": 64, "minibatch": None, "adv_norm": True, "adv_clip": 10.0,
                  "joint_opt": True, "vf_coef": 0.5, "max_grad_norm": 0.5, "adam_eps": 1e-5,
                  "actor_lr": 5e-4, "critic_lr": 5e-4, "lr_decay": "cosine", "trunc_as_end": True, "init": "orthogonal"},
}


def orthogonal_init(net: torch.nn.Module, out_gain: float) -> None:
    """Orthogonal weights (hidden layers gain √2, the output layer `out_gain`) and zero biases."""
    for name, layer in net.linear_layers():
        torch.nn.init.orthogonal_(layer.weight, gain=out_gain if name == "out" else math.sqrt(2))
        torch.nn.init.zeros_(layer.bias)


def lambda_at(cfg: PPOConfig, frac: float) -> float:
    """λ after `frac` of the run: fixed, or rising from gae_lambda towards gae_lambda_end along
    1 − sech(3·frac) (fast at first, then slowly: 0.9 → about 0.972 at the end for 0.9 → 0.98)."""
    if cfg.gae_lambda_end is None:
        return cfg.gae_lambda
    return cfg.gae_lambda + (1 - 1 / math.cosh(3 * frac)) * (cfg.gae_lambda_end - cfg.gae_lambda)


def lr_factor(cfg: PPOConfig, frac: float) -> float:
    """The learning rate's multiplier after `frac` of the run: 1, or a cosine from 1 down to 0."""
    if cfg.lr_decay == "none":
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


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
        run_cfg = RunConfig(env_id=cfg.env_id, gamma=cfg.gamma, seed=cfg.seed, shaping=cfg.shaping, env_kwargs=cfg.env_kwargs)
        self.env = make_env(run_cfg)
        self.encoder = ObsEncoder(run_cfg)
        n_actions = int(self.env.action_space.n)
        self.actor = Actor(self.encoder.dim, n_actions, (cfg.hidden,))
        self.critic = Critic(self.encoder.dim, (cfg.hidden,))
        if cfg.init == "orthogonal":
            orthogonal_init(self.actor, 0.01)
            orthogonal_init(self.critic, 1.0)
        if cfg.joint_opt:
            self.params = [*self.actor.parameters(), *self.critic.parameters()]
            self.opt = torch.optim.Adam(self.params, lr=cfg.actor_lr, eps=cfg.adam_eps)
        else:
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, eps=cfg.adam_eps)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, eps=cfg.adam_eps)
        self.env_steps = 0  # collected so far (the schedules run over cfg.total_steps)
        self.frac = 0.0  # share of the run done before the current batch
        self.lam = cfg.gae_lambda  # λ used for the current batch
        self.gen = torch.Generator().manual_seed(cfg.seed)  # actions
        self.shuffle_gen = torch.Generator().manual_seed(cfg.seed + 1)  # minibatch order
        self.env.action_space.seed(cfg.seed)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.ep_len = 0  # steps of the episode in progress (it may continue into the next batch)
        self.ep_ret = 0.0  # its return so far, with the OFFICIAL reward (no shaping bonus)
        self.k = 0
        self.last_mean_return = 0.0

    # ------------------------------------------------------------------ collect
    @torch.no_grad()
    def _collect(self):
        """Play with the frozen policy. Returns per-step arrays and the episode segments."""
        cfg = self.cfg
        X, acts, logp_old, rews, ends, next_obs = [], [], [], [], [], []
        # a segment = one episode's steps inside this batch; `fresh`: it started with a reset (not cut off
        # from the previous batch), `end`: how it ended (terminated / truncated / cut)
        segments, seg = [], {"states": [], "actions": [], "probs": [], "fresh": self.ep_len == 0}
        finished_lengths, finished_terminated, finished_returns = [], [], []
        n_done_episodes = 0
        while True:
            x = torch.as_tensor(self.encoder(self.obs, 0))
            p = F.softmax(self.actor(x), -1)
            a = int(torch.multinomial(p, 1, generator=self.gen))
            obs2, r, term, trunc, info = self.env.step(a)
            self.ep_ret += float(info.get("r_env", r))  # shaping wrappers keep the official reward in r_env
            X.append(x)
            acts.append(a)
            logp_old.append(float(torch.log(p[a])))
            rews.append(float(r))
            seg["states"].append(np.asarray(self.obs, dtype=float).tolist())
            seg["actions"].append(a)
            seg["probs"].append(p.tolist())
            next_obs.append(np.asarray(obs2, dtype=float))
            self.ep_len += 1
            end = "terminated" if term else "truncated" if trunc else None
            if end:
                finished_lengths.append(self.ep_len)
                finished_terminated.append(end == "terminated")
                finished_returns.append(self.ep_ret)
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
                seg["end"] = end
                seg["ret"] = self.ep_ret  # the whole episode's return if the segment is `fresh`
                segments.append(seg)
                seg = {"states": [], "actions": [], "probs": [], "fresh": True}
                if end != "cut":  # the episode is over; a cut one continues (with its return so far)
                    self.ep_ret = 0.0
            if full:
                break
        self._finished_terminated = finished_terminated
        self._finished_returns = finished_returns
        return X, acts, logp_old, rews, ends, next_obs, segments, finished_lengths

    # ------------------------------------------------------------------ advantages
    @torch.no_grad()
    def _advantages(self, X, rews, ends, next_obs):
        """GAE backwards along each episode, with the critic before any update."""
        g, lam = self.cfg.gamma, lambda_at(self.cfg, self.frac)
        self.lam = lam
        V = self.critic(torch.stack(X)).numpy().astype(np.float64)
        V_next = self.critic(torch.as_tensor(self.encoder(np.array(next_obs), 0))).numpy().astype(np.float64)
        A = np.zeros(len(rews))
        carry = 0.0
        for t in range(len(rews) - 1, -1, -1):
            if ends[t] == "terminated" or (ends[t] == "truncated" and self.cfg.trunc_as_end):  # nothing comes after
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
        self.frac = frac = min(1.0, self.env_steps / cfg.total_steps) if cfg.total_steps else 0.0
        lr_now = cfg.actor_lr * lr_factor(cfg, frac)
        X, acts, logp_old_l, rews, ends, next_obs, segments, finished = self._collect()
        self.env_steps += len(acts)
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
        grad_norms: list[tuple[float, float]] = []  # joint optimiser: (actor, critic) gradient norm before clipping
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
                    if cfg.adv_clip is not None:
                        adv = adv.clamp(-cfg.adv_clip, cfg.adv_clip)
                if cfg.joint_opt:
                    # one loss, one optimiser, the gradient of both networks clipped together
                    loss_v = 0.5 * (tgt_all[idx] - self.critic(Xb[idx])).pow(2).mean()
                    logp = F.log_softmax(self.actor(Xb[idx]), -1)[torch.arange(len(idx)), Ab[idx]]
                    loss_pi = actor_loss(logp, logp_old[idx], adv, cfg.clip, cfg.ratio)
                    for group in self.opt.param_groups:
                        group["lr"] = lr_now
                    self.opt.zero_grad()
                    (loss_pi + cfg.vf_coef * loss_v).backward()
                    gn_actor = float(torch.norm(torch.stack([p.grad.norm() for p in self.actor.parameters()])))
                    gn_critic = float(torch.norm(torch.stack([p.grad.norm() for p in self.critic.parameters()])))
                    if cfg.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                    self.opt.step()
                    grad_norms.append((gn_actor, gn_critic))
                    steps += 1
                    continue
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
            self.last_mean_return = float(np.mean(finished))  # the mean LENGTH of the finished episodes
        summary = {
            "k": self.k, "mean_return": self.last_mean_return, "episodes_finished": len(finished), "steps": n,
            "critic_loss": critic_loss0, "critic_loss_after": critic_loss1, "actor_loss": actor_loss0,
            "entropy": entropy, "ev": explained_variance(V.tolist(), target.tolist()),
            "mean_abs_adv": float(np.abs(A).mean()), "grad_steps": steps, "clip_frac": clip_frac,
            "finished_lengths": finished, "finished_terminated": self._finished_terminated,
            "finished_returns": self._finished_returns,
            "per_epoch": per_epoch,
            "lam": self.lam, "lr": lr_now if cfg.joint_opt else None,
            # the share of the joint gradient that is the critic's (before clipping)
            "gn_actor": float(np.mean([g[0] for g in grad_norms])) if grad_norms else None,
            "gn_critic": float(np.mean([g[1] for g in grad_norms])) if grad_norms else None,
        }
        return {
            "k": self.k, "summary": summary, "episodes": segments,
            "values": V.tolist(), "returns": target.tolist(), "advantages": A.tolist(),
            "actor": {"probs_before": log_probs0.exp().tolist(), "probs_after": log_probs1.exp().tolist(),
                      "logp_before": logp_old.tolist(), "logp_after": log_probs1[torch.arange(n), Ab].tolist()},
        }
