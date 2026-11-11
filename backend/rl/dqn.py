"""DQN, built up from online Q-learning one switch at a time (explained in docs/dqn_explained.html).

One Q-network with one output per action, Q(s, a). The policy is ε-greedy: with probability ε a random
action, otherwise argmax_a Q(s, a); ε falls from 1.0 to 0.05 over the first 10% of the run.

After every environment step (every `train_every` steps), ONE gradient step on a minibatch:
    y    = r + γ · (1 − terminated) · max_a′ Q_tgt(s′, a′)        (no bootstrap after a real end only;
                                                                   a time limit still has a future)
    loss = Huber(Q_θ(s, a) − y), averaged over the minibatch; gradient norm clipped to 10

The ladder:
    q_online   learn from the newest transition only (minibatch of 1), right after it happens;
               Q_tgt = the online network itself
    q_replay   + replay buffer: keep the last 100,000 transitions, learn from 64 random ones every 4 steps
               (after 1,000 steps), still Q_tgt = the online network
    dqn        + target network: Q_tgt = a frozen copy, refreshed every 1,000 env steps (DQN, 2015)

There are no batches or epochs: the loop never stops to collect. For the logs and the web page, one
"iteration" is a chunk of 2,048 environment steps (the size of a PPO batch), so both line up.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from .config import RunConfig
from .diagnostics import explained_variance
from .envs import ObsEncoder, make_env
from .networks import MLP, Actor


class DQNConfig(BaseModel):
    kind: Literal["dqn"] = "dqn"
    env_id: Literal["CartPole-v1", "MountainCar-v0", "DoubleCartPole-v0", "DoubleCartPoleJoint-v0", "LunarLander-v3"] = "LunarLander-v3"
    seed: int = 0
    hidden: int = Field(64, ge=2, le=512)
    lr: float = Field(5e-4, gt=0)
    gamma: float = Field(0.99, ge=0, le=1)
    replay: bool = False  # False: learn from the newest transition only
    buffer_size: int = Field(100_000, ge=1)
    batch_size: int = Field(64, ge=1)
    train_every: int = Field(1, ge=1)  # env steps between gradient steps
    learning_starts: int = Field(0, ge=0)  # env steps before the first gradient step (fills the buffer)
    target_net: bool = False  # False: the target uses the online network itself
    target_every: int = Field(1_000, ge=1)  # env steps between copies θ⁻ ← θ
    eps_start: float = Field(1.0, ge=0, le=1)
    eps_end: float = Field(0.05, ge=0, le=1)
    eps_fraction: float = Field(0.1, gt=0, le=1)  # ε reaches eps_end after this share of total_steps
    total_steps: int | None = Field(None, ge=1)  # the run's budget (for the ε schedule)
    chunk_steps: int = Field(2048, ge=1)  # env steps per logged iteration
    huber: bool = True
    max_grad_norm: float | None = Field(10.0, gt=0)
    shaping: Literal["none", "energy"] = "none"
    env_kwargs: dict[str, float | bool] = Field(default_factory=dict)


DQN_VARIANTS: dict[str, dict] = {
    "q_online": {},
    "q_replay": {"replay": True, "train_every": 4, "learning_starts": 1_000},
    "dqn": {"replay": True, "train_every": 4, "learning_starts": 1_000, "target_net": True},
}


class QNet(Actor):
    """The Q-network. Same layers as our actor (so the window viewer can load it and take the argmax),
    but forward() returns the log-probabilities of the ε-greedy policy, so every tool that reads "the
    policy" (replays, the explored tag, evaluation by argmax) sees what DQN really does."""

    eps: float = 0.0

    def qvalues(self, x: torch.Tensor) -> torch.Tensor:
        return MLP.forward(self, x)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        q = self.qvalues(x)
        n = q.shape[-1]
        greedy = F.one_hot(q.argmax(-1), n).to(q.dtype)
        p = (1 - self.eps) * greedy + self.eps / n
        return torch.log(p.clamp_min(1e-12))


class MaxQ(torch.nn.Module):
    """V(s) = max_a Q(s, a): the value of the greedy policy (what the training script calls the critic)."""

    def __init__(self, q: QNet):
        super().__init__()
        self.q = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q.qvalues(x).max(-1).values


class Replay:
    """A ring buffer of single transitions (s, a, r, s′, terminated); the oldest is overwritten when full."""

    def __init__(self, size: int, dim: int, seed: int):
        self.s = np.zeros((size, dim), np.float32)
        self.s2 = np.zeros((size, dim), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.n, self.ptr = size, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k: int):
        idx = self.rng.integers(0, self.n, size=k)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]), torch.from_numpy(self.d[idx]))


class DQNSession:
    """Same interface as PPOSession (iterate() → summary + trace), so the training script treats both alike."""

    def __init__(self, cfg: DQNConfig):
        torch.set_num_threads(1)
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        run_cfg = RunConfig(env_id=cfg.env_id, gamma=cfg.gamma, seed=cfg.seed, shaping=cfg.shaping, env_kwargs=cfg.env_kwargs)
        self.env = make_env(run_cfg)
        self.encoder = ObsEncoder(run_cfg)
        self.n_actions = int(self.env.action_space.n)
        self.qnet = QNet(self.encoder.dim, self.n_actions, (cfg.hidden,))
        self.target = copy.deepcopy(self.qnet) if cfg.target_net else None
        self.opt = torch.optim.Adam(self.qnet.parameters(), lr=cfg.lr)
        self.buffer = Replay(cfg.buffer_size, self.encoder.dim, cfg.seed) if cfg.replay else None
        self.actor = self.qnet  # what the training script plays, records and saves
        self.critic = MaxQ(self.qnet)
        self.rng = np.random.default_rng(cfg.seed)
        self.env.action_space.seed(cfg.seed)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.steps = 0  # env steps so far
        self.grad_steps = 0
        self.ep_len, self.ep_ret = 0, 0.0
        self.k = 0
        self.last_mean_return = 0.0
        self.qnet.eps = self.epsilon()

    def epsilon(self) -> float:
        c = self.cfg
        horizon = c.eps_fraction * (c.total_steps or 1_000_000)
        return c.eps_end + (c.eps_start - c.eps_end) * max(0.0, 1 - self.steps / horizon)

    def _q_target(self) -> QNet:
        return self.target if self.target is not None else self.qnet

    def _learn(self, s, a, r, s2, d) -> tuple[float, np.ndarray, np.ndarray]:
        """One gradient step on a minibatch. Returns the loss and (prediction, target) for the logs."""
        with torch.no_grad():
            y = r + self.cfg.gamma * (1 - d) * self._q_target().qvalues(s2).max(-1).values
        q = self.qnet.qvalues(s).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y) if self.cfg.huber else 0.5 * F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.grad_steps += 1
        return loss.item(), q.detach().numpy(), y.numpy()

    def iterate(self, record: bool = True) -> dict:
        cfg, n = self.cfg, self.cfg.chunk_steps
        before = copy.deepcopy(self.qnet)  # the policy at the start of the chunk (for "how far did it move")
        before.eps = self.epsilon()
        X, X2, acts, rews, terms, losses, preds, targs = [], [], [], [], [], [], [], []
        segments, seg = [], {"states": [], "actions": [], "probs": [], "fresh": self.ep_len == 0}
        finished_lengths, finished_terminated, finished_returns = [], [], []
        grad0 = self.grad_steps
        for t in range(n):
            eps = self.epsilon()
            self.qnet.eps = eps
            x = self.encoder(self.obs, 0).astype(np.float32)
            with torch.no_grad():
                q = self.qnet.qvalues(torch.as_tensor(x))
            explore = self.rng.random() < eps
            a = int(self.rng.integers(self.n_actions)) if explore else int(q.argmax())
            obs2, r, term, trunc, info = self.env.step(a)
            x2 = self.encoder(obs2, 0).astype(np.float32)
            self.steps += 1
            self.ep_len += 1
            self.ep_ret += float(info.get("r_env", r))
            probs = np.full(self.n_actions, eps / self.n_actions)
            probs[int(q.argmax())] += 1 - eps
            seg["states"].append(np.asarray(self.obs, dtype=float).tolist())
            seg["actions"].append(a)
            seg["probs"].append(probs.tolist())
            X.append(x)
            X2.append(x2)
            acts.append(a)
            rews.append(float(r))
            terms.append(float(term))
            # learn
            if self.buffer is not None:
                self.buffer.add(x, a, float(r), x2, float(term))
                if self.steps >= cfg.learning_starts and self.steps % cfg.train_every == 0 and self.buffer.n >= cfg.batch_size:
                    loss, qp, yt = self._learn(*self.buffer.sample(cfg.batch_size))
                    losses.append(loss)
                    preds.append(qp)
                    targs.append(yt)
            elif self.steps >= cfg.learning_starts and self.steps % cfg.train_every == 0:
                one = (torch.as_tensor(x)[None], torch.tensor([a]), torch.tensor([float(r)]), torch.as_tensor(x2)[None],
                       torch.tensor([float(term)]))
                loss, qp, yt = self._learn(*one)
                losses.append(loss)
                preds.append(qp)
                targs.append(yt)
            if self.target is not None and self.steps % cfg.target_every == 0:
                self.target.load_state_dict(self.qnet.state_dict())
            # episode bookkeeping (as in PPOSession: a chunk can cut an episode, which continues)
            end = "terminated" if term else "truncated" if trunc else None
            if end:
                finished_lengths.append(self.ep_len)
                finished_terminated.append(bool(term))
                finished_returns.append(self.ep_ret)
                self.ep_len = 0
                self.obs, _ = self.env.reset()
            else:
                self.obs = obs2
            if not end and t == n - 1:
                end = "cut"
            if end:
                seg["end"] = end
                seg["ret"] = self.ep_ret if end == "cut" else finished_returns[-1]
                segments.append(seg)
                seg = {"states": [], "actions": [], "probs": [], "fresh": True}
                if end != "cut":
                    self.ep_ret = 0.0

        self.qnet.eps = self.epsilon()
        Xb, X2b = torch.as_tensor(np.array(X)), torch.as_tensor(np.array(X2))
        Ab, Rb, Db = torch.tensor(acts), torch.tensor(rews), torch.tensor(terms)
        with torch.no_grad():
            lp0 = before(Xb)  # ε-greedy log-probabilities, policy at the start of the chunk
            lp1 = self.qnet(Xb)  # … and now
            q_sa = self.qnet.qvalues(Xb).gather(1, Ab[:, None]).squeeze(1)
            y = Rb + cfg.gamma * (1 - Db) * self._q_target().qvalues(X2b).max(-1).values
            td = (y - q_sa).numpy()
        # overestimation check: Q(s, a) against the discounted return that REALLY followed, on the
        # episodes that started and ended inside this chunk
        q_pred, g_real = [], []
        i = 0
        for sg in segments:
            k = len(sg["actions"])
            if sg["fresh"] and sg["end"] in ("terminated", "truncated"):
                g, G = 0.0, np.zeros(k)
                for j in range(k - 1, -1, -1):
                    g = rews[i + j] + cfg.gamma * g
                    G[j] = g
                q_pred.extend(q_sa[i:i + k].tolist())
                g_real.extend(G.tolist())
            i += k
        eps = self.epsilon()
        ent = -((1 - eps + eps / self.n_actions) * math.log(1 - eps + eps / self.n_actions)
                + (self.n_actions - 1) * (eps / self.n_actions) * math.log(max(eps / self.n_actions, 1e-12)))
        P = np.concatenate(preds) if preds else np.zeros(0)
        Y = np.concatenate(targs) if targs else np.zeros(0)
        self.k += 1
        if finished_lengths:
            self.last_mean_return = float(np.mean(finished_lengths))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        summary = {
            "k": self.k, "mean_return": self.last_mean_return, "episodes_finished": len(finished_lengths), "steps": n,
            "critic_loss": mean_loss, "critic_loss_after": float(losses[-1]) if losses else 0.0,
            "actor_loss": float(q_sa.mean()),  # no actor in DQN: the mean Q(s, a) on the chunk instead
            "entropy": ent, "ev": explained_variance(P.tolist(), Y.tolist()) if len(P) > 1 else None,
            "mean_abs_adv": float(np.abs(td).mean()), "grad_steps": self.grad_steps - grad0,
            "finished_lengths": finished_lengths, "finished_terminated": finished_terminated,
            "finished_returns": finished_returns, "per_epoch": [],
            "eps": eps, "buffer": self.buffer.n if self.buffer is not None else 1,
            "q_pred": float(np.mean(q_pred)) if q_pred else None, "g_real": float(np.mean(g_real)) if g_real else None,
        }
        return {
            "k": self.k, "summary": summary, "episodes": segments,
            "values": q_sa.tolist(), "returns": y.tolist(), "advantages": td.tolist(),
            "actor": {"probs_before": lp0.exp().tolist(), "probs_after": lp1.exp().tolist(),
                      "logp_before": lp0[torch.arange(n), Ab].tolist(), "logp_after": lp1[torch.arange(n), Ab].tolist()},
        }
