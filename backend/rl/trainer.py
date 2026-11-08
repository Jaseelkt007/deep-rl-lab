"""Training loop. Runs the algorithm, measures everything, and emits plain-dict records
through a callback. No file or network IO here: the logger and the server consume records.

Record types (all JSON-serialisable except `checkpoint`):
  run_start   config, label, names of state dims and actions
  step        one transition (s, a, r, s') and the update made for it
  episode     per-episode summary. Scores use the ORIGINAL reward r_env.
  snapshot    phase-plane maps: V(s), greedy action, P(last action), mean δ, last trajectory
  probe       fixed-ranking probe: does the actor's choice depend on the state?
  checkpoint  network weights (the logger saves them, the server drops them)
  run_end
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .actor_critic import ActorCritic, lr_factor, should_bootstrap
from .config import RunConfig
from .diagnostics import correlation, discounted_returns, explained_variance, hidden_stats
from .envs import ENV_INFO, ObsEncoder, action_map, make_env

Emit = Callable[[dict], None]

# Per-step keys averaged into the episode record as mean_<key>.
EPISODE_MEANS = (
    "delta", "entropy", "actor_loss", "critic_loss", "gn_actor", "gn_critic", "gn_actor_clipped",
    "gn_critic_clipped", "critic_share", "clip_actor", "clip_critic", "dw_actor", "dw_critic", "kl", "p_a",
    "adv_raw", "v",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    def __init__(self, cfg: RunConfig, emit: Emit | None = None, env: gym.Env | None = None):
        torch.set_num_threads(1)  # small networks: one thread is faster, and deterministic
        seed_everything(cfg.seed)
        self.cfg = cfg
        self.emit: Emit = emit or (lambda rec: None)
        self.info = ENV_INFO[cfg.env_id]
        self.env = env if env is not None else make_env(cfg)
        self.max_steps = cfg.max_episode_steps or self.info.official_max_steps
        self.encoder = ObsEncoder(cfg)
        self.amap = action_map(cfg)
        self.action_names = [self.info.action_names[i] for i in self.amap]
        self.agent = ActorCritic(self.encoder.dim, len(self.amap), cfg)  # init uses torch's seed
        self.gen = torch.Generator().manual_seed(cfg.seed)  # action sampling
        self.env.action_space.seed(cfg.seed)
        self.eval_env: gym.Env | None = None

        self.episode = 0  # completed episodes
        self.total_steps = 0
        self.finished = False
        self._seen_since_snapshot: list[tuple[float, float, float]] = []  # (x, y, δ) for the δ map
        self._seen_since_probe: deque = deque(maxlen=5000)  # (obs, t) for the probe
        self.last_traj: list[list[float]] = []
        self.started = time.time()

        self.emit({
            "type": "run_start", "config": cfg.model_dump(mode="json"), "label": cfg.label(),
            "helpers": cfg.helpers(), "obs_names": list(self.info.obs_names),
            "action_names": self.action_names, "max_steps": self.max_steps,
            "phase": {"dims": list(self.info.phase_dims), "ranges": [list(r) for r in self.info.phase_ranges],
                      "names": [self.info.obs_names[d] for d in self.info.phase_dims]},
        })
        self._snapshot()  # episode 0: the untrained networks
        self._start_episode(first=True)

    # ------------------------------------------------------------------ helpers
    def _x(self, obs, t) -> torch.Tensor:
        return torch.as_tensor(self.encoder(obs, t))

    def _start_episode(self, first: bool = False) -> None:
        obs, _ = self.env.reset(seed=self.cfg.seed) if first else self.env.reset()
        self.obs, self.t = np.asarray(obs, dtype=np.float64), 0
        self.pending: deque[dict] = deque()  # transitions waiting for their n-step update
        self.ep_steps: list[dict] = []
        self.ep_x: list[torch.Tensor] = []
        self.ep_layer = {"actor": None, "critic": None}
        self.ep_upd = {"actor": None, "critic": None}
        f = lr_factor(self.cfg.lr_schedule, self.episode / self.cfg.num_episodes)
        self.lr_actor, self.lr_critic = self.agent.set_lr_factor(f)

    # --------------------------------------------------------------------- step
    def step(self) -> bool:
        """One environment step, plus every update it makes possible. False when finished."""
        if self.finished:
            return False
        cfg = self.cfg
        x = self._x(self.obs, self.t)
        a, probs = self.agent.act(x, self.gen)
        obs2, r_shaped, terminated, truncated, info = self.env.step(self.amap[a])
        obs2 = np.asarray(obs2, dtype=np.float64)
        r_env = float(info.get("r_env", r_shaped))
        self.pending.append({
            "x": x, "s": self.obs, "t": self.t, "a": a, "p": probs.tolist(),
            "r_env": r_env, "r_shaped": float(r_shaped), "r_train": float(r_shaped) * cfg.reward_scale,
            "s_next": obs2,
        })
        self.ep_x.append(x)
        self._seen_since_probe.append((self.obs, self.t))
        self.t += 1
        self.total_steps += 1
        x2 = self._x(obs2, self.t)

        if terminated or truncated:
            # Flush: the last states get SHORTER windows that end at the episode end.
            # Nothing crosses into the next episode.
            boot = should_bootstrap(terminated, truncated, cfg.bootstrap_on_truncation)
            while self.pending:
                self._update_oldest(x2, boot, terminated, truncated)
        elif len(self.pending) >= cfg.n_step:
            self._update_oldest(x2, True, False, False)

        self.obs = obs2
        if terminated or truncated:
            self._end_episode(terminated, truncated)
        return not self.finished

    def run(self) -> None:
        while self.step():
            pass

    def _update_oldest(self, x_last: torch.Tensor, bootstrap: bool, terminated: bool, truncated: bool) -> None:
        """Update the oldest pending state with the window that follows it."""
        xs = torch.stack([p["x"] for p in self.pending] + [x_last])
        rewards = [p["r_train"] for p in self.pending]  # shaped and scaled: what the agent learns from
        tr = self.pending.popleft()
        is_last = not self.pending
        weight = self.cfg.gamma ** tr["t"] if self.cfg.actor_discount_I else 1.0
        out = self.agent.update(xs, tr["a"], rewards, bootstrap, actor_weight=weight)
        self._accumulate_layers(out)
        rec = {
            "type": "step", "ep": self.episode + 1, "t": tr["t"], "s": tr["s"].tolist(), "a": tr["a"],
            "a_env": self.amap[tr["a"]], "p": tr["p"], "r_env": tr["r_env"], "r_shaped": tr["r_shaped"],
            "r_train": tr["r_train"],
            "s_next": tr["s_next"].tolist(), "window": len(rewards), "bootstrap": bootstrap,
            "term": terminated and is_last, "trunc": truncated and is_last, **out["stats"],
        }
        self.ep_steps.append(rec)
        dx, dy = self.info.phase_dims
        self._seen_since_snapshot.append((float(tr["s"][dx]), float(tr["s"][dy]), rec["delta"]))
        self.emit(rec)

    def _accumulate_layers(self, out: dict) -> None:
        for net in ("actor", "critic"):
            g, u = np.array(out["layer_grads"][net]), np.array(out["layer_updates"][net])
            self.ep_layer[net] = g if self.ep_layer[net] is None else self.ep_layer[net] + g
            self.ep_upd[net] = u if self.ep_upd[net] is None else self.ep_upd[net] + u

    # ------------------------------------------------------------------ episode
    def _end_episode(self, terminated: bool, truncated: bool) -> None:
        cfg, steps = self.cfg, self.ep_steps
        self.episode += 1
        n = len(steps)
        ret = sum(s["r_env"] for s in steps)

        # Critic quality: V(s_t) vs the discounted return the critic is trained to predict
        # (r_train: shaped and scaled if those switches are on). After truncation the tail is filled with V(s_T),
        # always, even with the bug switch on, so the bug shows up as a worse EV.
        tail = steps[-1]["v_next"] if truncated and not terminated else 0.0
        g = discounted_returns([s["r_train"] for s in steps], cfg.gamma, tail)
        v = [s["v"] for s in steps]

        if cfg.env_id == "MountainCar-v0":
            success = bool(terminated)
        else:  # CartPole: success = the pole stayed up until the time limit
            success = bool(truncated and not terminated)

        rec = {
            "type": "episode", "ep": self.episode, "return": ret,
            "return_shaped": sum(s["r_shaped"] for s in steps), "return_train": sum(s["r_train"] for s in steps),
            "length": n, "success": success,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "mean_abs_delta": float(np.mean([abs(s["delta"]) for s in steps])),
            "adv_std": float(np.std([s["adv_raw"] for s in steps])),
            "ev": explained_variance(v, g), "corr": correlation(v, g.tolist()),
            "g_mean": float(g.mean()), "lr_actor": self.lr_actor, "lr_critic": self.lr_critic,
            "total_steps": self.total_steps, "elapsed": time.time() - self.started,
            "max_pos": max(max(s["s"][0], s["s_next"][0]) for s in steps) if cfg.env_id == "MountainCar-v0" else None,
        }
        for k in EPISODE_MEANS:
            rec[f"mean_{k}"] = float(np.mean([s[k] for s in steps]))
        rec["layers"] = self._layer_summary(n)
        rec["hidden"] = self._hidden_summary()
        self.emit(rec)

        self.last_traj = [s["s"] for s in steps] + [steps[-1]["s_next"]]
        if self.episode % cfg.snapshot_every == 0:
            self._snapshot()
        if self.episode % cfg.state_probe_every == 0:
            self._probe()
        done = self.episode >= cfg.num_episodes
        if self.episode % cfg.checkpoint_every == 0 or done:
            self.emit({"type": "checkpoint", "ep": self.episode, "state": self.agent.state_dict()})
        if done:
            self.finished = True
            self.emit({"type": "run_end", "ep": self.episode, "total_steps": self.total_steps})
        else:
            self._start_episode()

    def _layer_summary(self, n: int) -> dict:
        """Per network and layer: mean grad norm, mean update norm, weight norm, update/weight."""
        weights = self.agent.layer_weight_norms()
        out = {}
        for net, model in (("actor", self.agent.actor), ("critic", self.agent.critic)):
            names = [name for name, _ in model.linear_layers()]
            out[net] = [
                {"name": name, "grad": float(self.ep_layer[net][i] / n), "update": float(self.ep_upd[net][i] / n),
                 "weight": weights[net][i], "ratio": float(self.ep_upd[net][i] / n / (weights[net][i] + 1e-12))}
                for i, name in enumerate(names)
            ]
        return out

    @torch.no_grad()
    def _hidden_summary(self) -> dict:
        xs = torch.stack(self.ep_x)
        _, h_actor = self.agent.actor(xs, return_hidden=True)
        _, h_critic = self.agent.critic(xs, return_hidden=True)
        return {"actor": hidden_stats(h_actor, self.cfg.activation),
                "critic": hidden_stats(h_critic, self.cfg.activation)}

    # ----------------------------------------------------------------- snapshot
    @torch.no_grad()
    def _snapshot(self) -> None:
        """Evaluate V, π over a G×G grid of the two phase-plane dims (others at phase_base).
        Arrays are row-major [row = y (velocity) index, col = x (position) index]."""
        G = self.cfg.grid_size
        (x0, x1), (y0, y1) = self.info.phase_ranges
        dx, dy = self.info.phase_dims
        xs, ys = np.linspace(x0, x1, G), np.linspace(y0, y1, G)
        states = np.tile(np.array(self.info.phase_base, dtype=np.float64), (G * G, 1))
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        states[:, dx], states[:, dy] = xx.ravel(), yy.ravel()
        _, probs, values = self.agent.evaluate(torch.as_tensor(self.encoder(states, 0)))

        # mean δ the actor received in each cell since the last snapshot (coarser grid)
        B = max(G // 2, 4)
        d_sum, d_cnt = np.zeros((B, B)), np.zeros((B, B))
        for px, py, d in self._seen_since_snapshot:
            i = int(np.clip((px - x0) / (x1 - x0) * B, 0, B - 1))
            j = int(np.clip((py - y0) / (y1 - y0) * B, 0, B - 1))
            d_sum[j, i] += d
            d_cnt[j, i] += 1
        d_mean = np.where(d_cnt > 0, d_sum / np.maximum(d_cnt, 1), np.nan)
        self._seen_since_snapshot = []

        self.emit({
            "type": "snapshot", "ep": self.episode, "grid": G, "x": xs.tolist(), "y": ys.tolist(),
            "value": values.tolist(), "best_action": probs.argmax(dim=1).tolist(),
            "p_right": probs[:, -1].tolist(), "delta_bins": B,
            "delta_mean": [None if np.isnan(v) else float(v) for v in d_mean.ravel()],
            "delta_count": d_cnt.ravel().astype(int).tolist(),
            "traj": [[s[dx], s[dy]] for s in self.last_traj],
        })

    # -------------------------------------------------------------------- probe
    @torch.no_grad()
    def _probe(self) -> None:
        """Fixed-ranking probe. Replace π(·|s) by one state-independent policy built from the
        mean logits over recently visited states, and compare scores. If the fixed policy
        scores as well as the real one, the actor is not using the state."""
        if not self._seen_since_probe:
            return
        obs = np.array([o for o, _ in self._seen_since_probe])
        ts = np.array([t for _, t in self._seen_since_probe])
        logits = self.agent.actor(torch.as_tensor(self.encoder(obs, ts)))
        mean_logits = logits.mean(dim=0)
        fixed_probs = F.softmax(mean_logits, dim=-1)
        fixed_a = int(mean_logits.argmax())
        state_dependence = float((logits.argmax(dim=1) != fixed_a).float().mean())
        self._seen_since_probe.clear()

        if self.eval_env is None:
            self.eval_env = make_env(self.cfg.model_copy(update={"shaping": "none"}))
        gen = torch.Generator().manual_seed(self.cfg.seed + 7919 * self.episode)

        def policy(mode):
            def choose(o, t):
                if mode == "fixed":
                    return int(torch.multinomial(fixed_probs, 1, generator=gen))
                if mode == "fixed_greedy":
                    return fixed_a
                p = F.softmax(self.agent.actor(self._x(o, t)), dim=-1)
                return int(p.argmax()) if mode == "greedy" else int(torch.multinomial(p, 1, generator=gen))
            return choose

        returns = {}
        for mode in ("policy", "greedy", "fixed", "fixed_greedy"):
            scores = []
            for i in range(self.cfg.probe_episodes):  # same start states for every mode
                o, _ = self.eval_env.reset(seed=10_000 + 100 * self.cfg.seed + i)
                choose, total, t = policy(mode), 0.0, 0
                while True:
                    o, r, term, trunc, _ = self.eval_env.step(self.amap[choose(o, t)])
                    total, t = total + float(r), t + 1
                    if term or trunc:
                        break
                scores.append(total)
            returns[mode] = float(np.mean(scores))

        self.emit({
            "type": "probe", "ep": self.episode, "mean_logits": mean_logits.tolist(),
            "fixed_probs": fixed_probs.tolist(), "fixed_action": fixed_a,
            "state_dependence": state_dependence, "returns": returns,
        })
