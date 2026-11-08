"""Plain one-step actor-critic (Sutton & Barto 2018, ch. 13.5) and its optional extensions.

Two separate networks: an actor π_θ(a|s) and a critic V_w(s).
At EVERY environment step t (no replay buffer, no batches, no target network):

    δ_t = r_t + γ · V_w(s_{t+1}) · (1 − terminated_t) − V_w(s_t)        TD error

    critic loss  L_V = ½ δ_t²              V_w(s_{t+1}) is a constant (semi-gradient TD)
                 ∇_w L_V = −δ_t ∇_w V_w(s_t)
    actor loss   L_π = −δ_t · log π_θ(a_t|s_t)       δ_t is a constant (.detach())
                 ∇_θ L_π = −δ_t ∇_θ log π_θ(a_t|s_t)

Gradient DESCENT on L_π moves θ along +δ ∇log π: if δ > 0 (things went better than the
critic expected) a_t becomes more likely in s_t; if δ < 0 it becomes less likely.
So the critic is the actor's teacher: a wrong critic means a noisy δ, and the actor learns noise.

terminated: s_{t+1} is a real terminal state (goal reached, pole fell). Its value is 0
            by definition, so we do NOT bootstrap.
truncated:  the time limit cut the episode off. s_{t+1} still has a future, so we DO
            bootstrap from V(s_{t+1}). Treating it as terminal is a common bug
            (switch: bootstrap_on_truncation=False).

About I: S&B's boxed algorithm also scales the actor step by I = γ^t (t = time since the
episode started, I ← γI): θ ← θ + α I δ ∇log π. It comes from the episodic objective
J = V(s_0), which discounts what happens late in an episode. Almost every implementation
drops it, and so does the plain setting here. With γ = 0.99 and 200 steps I falls to
0.13, so late-episode states teach the actor much less. Switch: actor_discount_I.

Optional (all off by default):
  n-step:   G = r_t + γ r_{t+1} + … + γ^{n−1} r_{t+n−1} + γ^n V(s_{t+n}),   A = G − V(s_t)
  GAE(λ):   A = Σ_{k<n} (γλ)^k δ_{t+k}     (λ = 1: the n-step A above; n = 1: A = δ_t)
  entropy:  L_π −= β · H(π(·|s_t))   (pushes towards keeping all actions possible)
  advantage normalisation, gradient clipping (separate or joint), LR schedules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .config import RunConfig
from .networks import Actor, Critic


def td_target(r: float, v_next: float, terminated: bool, gamma: float) -> float:
    """One-step TD target  r + γ · V(s') · (1 − terminated)."""
    return r + gamma * v_next * (1.0 - float(terminated))


def should_bootstrap(terminated: bool, truncated: bool, bootstrap_on_truncation: bool = True) -> bool:
    """Does the target include V(s')? Never after termination; after truncation only if correct."""
    if terminated:
        return False
    if truncated:
        return bootstrap_on_truncation
    return True


def lr_factor(schedule: str, progress: float) -> float:
    """Learning-rate multiplier for progress ∈ [0, 1] through training."""
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        return 1.0 - progress
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


class RunningNorm:
    """EMA estimate of mean and std (bias-corrected like Adam). Used for advantage normalisation:
    an online learner sees one advantage per update, so there is no batch to normalise over."""

    def __init__(self, beta: float = 0.99):
        self.beta, self.k, self.m, self.s = beta, 0, 0.0, 0.0

    def __call__(self, x: float) -> float:
        self.k += 1
        self.m = self.beta * self.m + (1 - self.beta) * x
        self.s = self.beta * self.s + (1 - self.beta) * x * x
        corr = 1.0 - self.beta**self.k
        mean, var = self.m / corr, max(self.s / corr - (self.m / corr) ** 2, 0.0)
        return (x - mean) / (math.sqrt(var) + 1e-8)


def _norm(tensors) -> float:
    return math.sqrt(sum(float(t.pow(2).sum()) for t in tensors))


class ActorCritic:
    def __init__(self, obs_dim: int, n_actions: int, cfg: RunConfig):
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.lam = 1.0 if cfg.gae_lambda is None else cfg.gae_lambda
        self.actor = Actor(obs_dim, n_actions, cfg.hidden_sizes, cfg.activation)
        self.critic = Critic(obs_dim, cfg.hidden_sizes, cfg.activation)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.adv_norm = RunningNorm() if cfg.adv_norm else None

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, x: torch.Tensor, generator: torch.Generator) -> tuple[int, torch.Tensor]:
        """Sample a ~ π(·|s). Returns the action and the probabilities it was sampled from."""
        probs = F.softmax(self.actor(x), dim=-1)
        a = int(torch.multinomial(probs, 1, generator=generator))
        return a, probs

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Logits, probabilities and V for a batch of states (for maps and probes)."""
        logits = self.actor(x)
        return logits, F.softmax(logits, dim=-1), self.critic(x)

    def set_lr_factor(self, f: float) -> tuple[float, float]:
        for g in self.actor_opt.param_groups:
            g["lr"] = self.cfg.actor_lr * f
        for g in self.critic_opt.param_groups:
            g["lr"] = self.cfg.critic_lr * f
        return self.cfg.actor_lr * f, self.cfg.critic_lr * f

    # ----------------------------------------------------------------- learning
    def update(
        self, xs: torch.Tensor, action: int, rewards: Sequence[float], bootstrap: bool, actor_weight: float = 1.0
    ) -> dict:
        """One actor and one critic step for the state s_t = xs[0].

        xs:        features of s_t, s_{t+1}, …, s_{t+m}   (m = len(rewards), m = 1 for plain)
        rewards:   r_t, …, r_{t+m−1}  (the rewards the agent learns from, i.e. shaped if shaping is on)
        bootstrap: whether s_{t+m} is non-terminal, i.e. V(s_{t+m}) is part of the target.
        actor_weight: multiplies the policy-gradient term; I = γ^t with actor_discount_I, else 1.
        Returns a flat dict of numbers describing this update, plus per-layer stats.
        Gradient norms gn_* are measured BEFORE clipping; gn_*_clipped after.
        """
        m, gamma, lam = len(rewards), self.gamma, self.lam

        # --- critic values. Only V(s_t) gets a gradient; later V's are constants.
        v = self.critic(xs[0:1]).squeeze(0)  # V_w(s_t), with grad
        with torch.no_grad():
            v_seq = self.critic(xs).tolist()  # V(s_t) … V(s_{t+m}) before this update

        # --- TD errors along the window:  δ_k = r_k + γ V(s_{k+1}) c_k − V(s_k)
        #     c_k = 1 inside the window; for the last one c = bootstrap (0 after termination).
        deltas = []
        for k in range(m):
            cont = 1.0 if (k < m - 1 or bootstrap) else 0.0
            deltas.append(rewards[k] + gamma * v_seq[k + 1] * cont - v_seq[k])
        # --- advantage  A = Σ_k (γλ)^k δ_k   (m = 1: A = δ_t;  λ = 1: A = G − V(s_t))
        adv_raw = sum((gamma * lam) ** k * d for k, d in enumerate(deltas))
        target = v_seq[0] + adv_raw  # the critic's regression target (a constant)

        # --- critic loss  ½ (target − V(s_t))²   (= ½ δ² for the plain algorithm)
        critic_loss = 0.5 * (target - v).pow(2)

        # --- actor loss  −I · A · log π(a_t|s_t) − β H(π(·|s_t))     (I = 1 unless actor_discount_I)
        adv = self.adv_norm(adv_raw) if self.adv_norm else adv_raw
        logits = self.actor(xs[0:1]).squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()
        actor_loss = -actor_weight * adv * log_probs[action] - self.cfg.entropy_beta * entropy

        # --- gradients (two separate graphs, two separate optimisers)
        self.actor_opt.zero_grad(set_to_none=False)
        self.critic_opt.zero_grad(set_to_none=False)
        actor_loss.backward()
        critic_loss.backward()
        a_params, c_params = list(self.actor.parameters()), list(self.critic.parameters())
        gn_a, gn_c = _norm(p.grad for p in a_params), _norm(p.grad for p in c_params)
        layer_grads = {
            "actor": [_norm([l.weight.grad, l.bias.grad]) for _, l in self.actor.linear_layers()],
            "critic": [_norm([l.weight.grad, l.bias.grad]) for _, l in self.critic.linear_layers()],
        }
        clip_a, clip_c = self._clip(gn_a, gn_c, a_params, c_params)

        # --- optimiser steps, measuring how far the weights moved
        before = {id(p): p.detach().clone() for p in a_params + c_params}
        self.actor_opt.step()
        self.critic_opt.step()

        def moved(ps):
            return _norm(p.detach() - before[id(p)] for p in ps)

        dw_a, dw_c = moved(a_params), moved(c_params)
        layer_updates = {
            "actor": [moved([l.weight, l.bias]) for _, l in self.actor.linear_layers()],
            "critic": [moved([l.weight, l.bias]) for _, l in self.critic.linear_layers()],
        }

        # --- the same state after the update: did the action become more likely?
        with torch.no_grad():
            log_probs_after = F.log_softmax(self.actor(xs[0:1]).squeeze(0), dim=-1)
            v_after = float(self.critic(xs[0:1]))
            kl = float((probs * (log_probs - log_probs_after)).sum())  # KL(π_old ‖ π_new) at s_t
        p_old, p_new = probs.detach(), log_probs_after.exp()

        stats = {
            "v": v_seq[0], "v_next": v_seq[1], "delta": deltas[0], "adv_raw": adv_raw, "adv": adv,
            "target": target, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "logp": log_probs[action].item(), "logp_after": float(log_probs_after[action]),
            "p_a": float(p_old[action]), "p_a_after": float(p_new[action]),
            "p_after": p_new.tolist(), "v_after": v_after, "kl": kl,
            "gn_actor": gn_a, "gn_critic": gn_c, "clip_actor": clip_a, "clip_critic": clip_c,
            "gn_actor_clipped": gn_a * clip_a, "gn_critic_clipped": gn_c * clip_c, "actor_weight": actor_weight,
            "critic_share": gn_c**2 / (gn_a**2 + gn_c**2 + 1e-12),
            "dw_actor": dw_a, "dw_critic": dw_c,
        }
        return {"stats": stats, "layer_grads": layer_grads, "layer_updates": layer_updates}

    def _clip(self, gn_a, gn_c, a_params, c_params) -> tuple[float, float]:
        """Clip gradients by norm. Returns the factor each network's gradient was scaled by.

        separate: each network is clipped against its own norm.
        joint:    one global norm over BOTH networks (PPO code with a shared optimiser).
                  A large critic gradient then shrinks the actor's step too.
        """
        mode, c = self.cfg.grad_clip, self.cfg.grad_clip_norm
        if mode == "separate":
            fa, fc = min(1.0, c / (gn_a + 1e-6)), min(1.0, c / (gn_c + 1e-6))
        elif mode == "joint":
            fa = fc = min(1.0, c / (math.sqrt(gn_a**2 + gn_c**2) + 1e-6))
        else:
            return 1.0, 1.0
        for p in a_params:
            p.grad.mul_(fa)
        for p in c_params:
            p.grad.mul_(fc)
        return fa, fc

    # ------------------------------------------------------------- inspection
    def layer_weight_norms(self) -> dict[str, list[float]]:
        return {
            "actor": [_norm([l.weight, l.bias]) for _, l in self.actor.linear_layers()],
            "critic": [_norm([l.weight, l.bias]) for _, l in self.critic.linear_layers()],
        }

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
