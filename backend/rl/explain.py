"""Batch on-policy actor-critic, one iteration at a time, with EVERY intermediate number recorded.

This powers the "Learn" page. It is the textbook batch version (A2C-style, Monte Carlo
returns), small enough to show every weight:

  1. FIX the policy π_θ. Play N complete episodes with it (no learning while playing).
  2. Pass every visited state through the critic: V_w(s_t)  (random numbers at first).
  3. Returns, backwards from the end of each episode:  G_t = r_t + γ·G_{t+1}
        G_T = 0         after termination (pole fell / flag reached: nothing more to come)
        G_T = V_w(s_T)  after truncation  (time limit: the future exists, the critic estimates it)
  4. Advantages:  A_t = G_t − V_w(s_t)    "how much better than the critic expected"
  5. ONE critic gradient step on   L_V = mean_t ½ (G_t − V_w(s_t))²      (regression to G)
  6. ONE actor gradient step on    L_π = −mean_t A_t · log π_θ(a_t|s_t)  (A is a constant)
  7. Throw the batch away (on-policy: it was collected by the OLD policy). Back to 1.

The live dashboard uses the per-step version (rl/actor_critic.py); the maths is the same, only
the bookkeeping differs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from .config import RunConfig
from .diagnostics import explained_variance
from .envs import ENV_INFO, ObsEncoder, make_env
from .networks import Actor, Critic


class LearnConfig(BaseModel):
    env_id: Literal["MountainCar-v0", "CartPole-v1", "DoubleCartPole-v0", "DoubleCartPoleJoint-v0", "LunarLander-v3"] = "CartPole-v1"
    seed: int = 0
    batch_episodes: int | None = Field(None, ge=1, le=50)  # None: 2 for MountainCar, else 5
    hidden: int = Field(8, ge=2, le=64)  # ONE hidden layer, small enough to draw every weight
    optimizer: Literal["adam", "sgd"] = "adam"
    actor_lr: float = Field(0.01, gt=0)
    critic_lr: float = Field(0.03, gt=0)
    gamma: float = Field(0.99, ge=0, le=1)
    shaping: Literal["none", "energy"] = "none"  # MountainCar only
    grid: int = Field(25, ge=5, le=60)

    def n_episodes(self) -> int:
        return self.batch_episodes or (2 if self.env_id == "MountainCar-v0" else 5)


def _params(net) -> dict[str, torch.Tensor]:
    """Named weights in layer order: W1, b1 (input → hidden), W2, b2 (hidden → output)."""
    out = {}
    for i, (_, layer) in enumerate(net.linear_layers()):
        out[f"W{i + 1}"], out[f"b{i + 1}"] = layer.weight, layer.bias
    return out


def _dump(tensors: dict[str, torch.Tensor]) -> dict[str, list]:
    return {k: t.detach().tolist() for k, t in tensors.items()}


class ExplainSession:
    def __init__(self, cfg: LearnConfig):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.run_cfg = RunConfig(env_id=cfg.env_id, gamma=cfg.gamma, shaping=cfg.shaping, seed=cfg.seed)
        self.info = ENV_INFO[cfg.env_id]
        self.env = make_env(self.run_cfg)
        self.encoder = ObsEncoder(self.run_cfg)
        n_actions = len(self.info.action_names)
        self.actor = Actor(self.encoder.dim, n_actions, (cfg.hidden,))
        self.critic = Critic(self.encoder.dim, (cfg.hidden,))
        opt = (lambda p, lr: torch.optim.SGD(p, lr=lr)) if cfg.optimizer == "sgd" else (lambda p, lr: torch.optim.Adam(p, lr=lr))
        self.actor_opt = opt(self.actor.parameters(), cfg.actor_lr)
        self.critic_opt = opt(self.critic.parameters(), cfg.critic_lr)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.env.action_space.seed(cfg.seed)
        self.first_reset = True
        self.k = 0  # completed iterations
        self.summaries: list[dict] = []
        self.traces: dict[int, dict] = {}

    # ------------------------------------------------------------------ describe
    def describe(self) -> dict:
        """The problem's formulation and the untrained networks (shown before iteration 1)."""
        info = self.info
        mc = self.cfg.env_id == "MountainCar-v0"
        return {
            "config": self.cfg.model_dump(), "n_episodes": self.cfg.n_episodes(),
            "obs_names": list(info.obs_names), "action_names": list(info.action_names),
            "obs_low": info.obs_low.tolist(), "obs_high": info.obs_high.tolist(),
            "reward": "−1 every step" if mc else "+1 every step the pole is still up",
            "terminated": "the car reaches the flag (position ≥ 0.5)" if mc else "the pole tips past ±12° or the cart leaves the track",
            "truncated": f"after {info.official_max_steps} steps (time limit)",
            "max_steps": info.official_max_steps,
            "value_meaning": ("V(s) ≈ −(steps still needed to reach the flag), discounted; −100 if never"
                              if mc else "V(s) ≈ how many more steps the pole stays up, discounted; at most 100"),
            "weights": {"actor": _dump(_params(self.actor)), "critic": _dump(_params(self.critic))},
            "maps": self._maps(),
            "phase": {"dims": list(info.phase_dims), "ranges": [list(r) for r in info.phase_ranges],
                      "names": [info.obs_names[d] for d in info.phase_dims]},
        }

    @torch.no_grad()
    def _maps(self) -> dict:
        G = self.cfg.grid
        (x0, x1), (y0, y1) = self.info.phase_ranges
        dx, dy = self.info.phase_dims
        xs, ys = np.linspace(x0, x1, G), np.linspace(y0, y1, G)
        states = np.tile(np.array(self.info.phase_base), (G * G, 1))
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        states[:, dx], states[:, dy] = xx.ravel(), yy.ravel()
        x = torch.as_tensor(self.encoder(states, 0))
        return {"grid": G, "x": xs.tolist(), "y": ys.tolist(),
                "value": self.critic(x).tolist(), "p_last": F.softmax(self.actor(x), -1)[:, -1].tolist()}

    # ----------------------------------------------------------------- iterate
    @torch.no_grad()
    def _collect(self) -> list[dict]:
        """Play N complete episodes with the CURRENT, fixed policy. Nothing learns here."""
        episodes = []
        for _ in range(self.cfg.n_episodes()):
            obs, _ = self.env.reset(seed=self.cfg.seed) if self.first_reset else self.env.reset()
            self.first_reset = False
            ep = {"states": [], "features": [], "actions": [], "probs": [], "rewards": [], "rewards_env": []}
            while True:
                x = torch.as_tensor(self.encoder(obs, 0))
                p = F.softmax(self.actor(x), -1)
                a = int(torch.multinomial(p, 1, generator=self.gen))
                obs2, r, term, trunc, info = self.env.step(a)
                ep["states"].append(np.asarray(obs, dtype=float).tolist())
                ep["features"].append(x.tolist())
                ep["actions"].append(a)
                ep["probs"].append(p.tolist())
                ep["rewards"].append(float(r))
                ep["rewards_env"].append(float(info.get("r_env", r)))
                obs = obs2
                if term or trunc:
                    break
            ep.update(final_state=np.asarray(obs, dtype=float).tolist(), terminated=bool(term),
                      truncated=bool(trunc and not term), length=len(ep["actions"]),
                      ret=float(sum(ep["rewards_env"])))
            ep["final_value"] = float(self.critic(torch.as_tensor(self.encoder(obs, 0))))
            episodes.append(ep)
        return episodes

    def iterate(self, record: bool = True) -> dict:
        """One full iteration (collect → critic → returns → advantages → critic step → actor step)."""
        cfg = self.cfg
        before_maps = self._maps() if record else None
        w_before = {"actor": _dump(_params(self.actor)), "critic": _dump(_params(self.critic))} if record else None
        episodes = self._collect()

        # flatten the batch: one row per step
        X = torch.tensor([f for ep in episodes for f in ep["features"]], dtype=torch.float32)
        A = torch.tensor([a for ep in episodes for a in ep["actions"]])
        ep_index = [i for i, ep in enumerate(episodes) for _ in ep["actions"]]
        t_index = [t for ep in episodes for t in range(ep["length"])]

        # 2. critic on the batch (before any update)
        with torch.no_grad():
            values = self.critic(X)
        # 3. returns, backwards per episode
        returns = []
        for ep in episodes:
            g = 0.0 if ep["terminated"] else ep["final_value"]
            ep["tail"] = g
            gs = []
            for r in reversed(ep["rewards"]):
                g = r + cfg.gamma * g
                gs.append(g)
            returns += gs[::-1]
        G = torch.tensor(returns, dtype=torch.float32)
        # 4. advantages
        adv = G - values

        # 5. critic gradient step: L_V = mean ½ (G − V)²
        cp = _params(self.critic)
        v = self.critic(X)
        critic_loss = 0.5 * (G - v).pow(2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        c_grads = {k: p.grad.detach().clone() for k, p in cp.items()}
        c_before = {k: p.detach().clone() for k, p in cp.items()}
        self.critic_opt.step()
        c_delta = {k: cp[k].detach() - c_before[k] for k in cp}

        # 6. actor gradient step: L_π = −mean A · log π(a|s)
        ap = _params(self.actor)
        logits = self.actor(X)
        log_probs = F.log_softmax(logits, -1)
        logp = log_probs[torch.arange(len(A)), A]
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        actor_loss = -(adv * logp).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        a_grads = {k: p.grad.detach().clone() for k, p in ap.items()}
        a_before = {k: p.detach().clone() for k, p in ap.items()}
        self.actor_opt.step()
        a_delta = {k: ap[k].detach() - a_before[k] for k in ap}

        with torch.no_grad():
            values_after = self.critic(X)
            log_probs_after = F.log_softmax(self.actor(X), -1)
            critic_loss_after = 0.5 * (G - values_after).pow(2).mean()

        self.k += 1
        ev = explained_variance(values.tolist(), returns)
        summary = {
            "k": self.k, "mean_return": float(np.mean([ep["ret"] for ep in episodes])),
            "mean_length": float(np.mean([ep["length"] for ep in episodes])),
            "steps": len(returns), "critic_loss": critic_loss.item(), "critic_loss_after": float(critic_loss_after),
            "actor_loss": actor_loss.item(), "entropy": entropy.item(), "ev": ev,
            "mean_abs_adv": float(adv.abs().mean()), "mean_value": float(values.mean()), "mean_return_G": float(G.mean()),
        }
        self.summaries.append(summary)
        if not record:
            return {"summary": summary}

        trace = {
            "k": self.k, "summary": summary, "weights_before": w_before,
            "episodes": [{k: v for k, v in ep.items() if k != "features"} for ep in episodes],
            "features": X.tolist(), "ep_index": ep_index, "t_index": t_index, "actions": A.tolist(),
            "values": values.tolist(), "returns": returns, "advantages": adv.tolist(),
            "critic": {"loss": critic_loss.item(), "loss_after": float(critic_loss_after),
                       "grads": _dump(c_grads), "delta": _dump(c_delta), "weights_after": _dump(cp),
                       "values_after": values_after.tolist(), "lr": cfg.critic_lr},
            "actor": {"loss": actor_loss.item(), "grads": _dump(a_grads), "delta": _dump(a_delta),
                      "weights_after": _dump(ap), "probs_before": log_probs.exp().tolist(),
                      "probs_after": log_probs_after.exp().tolist(), "logp_before": logp.tolist(),
                      "logp_after": log_probs_after[torch.arange(len(A)), A].tolist(), "lr": cfg.actor_lr},
            "maps": {"before": before_maps, "after": self._maps()},
            "optimizer": cfg.optimizer, "gamma": cfg.gamma,
        }
        self.traces[self.k] = trace
        return trace

    def forward(self, n: int) -> dict:
        """Run n iterations quickly (only the last one fully recorded)."""
        for i in range(n):
            last = self.iterate(record=(i == n - 1))
        return last
