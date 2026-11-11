"""Train CartPole (or the double cart-pole) from the terminal: batch actor-critic, and the ladder from it
to PPO. No web server needed.

    uv run python scripts/train_cartpole.py --algo batch_ac   # logs + game window every 25 iterations
                                                              # + one plot opened at the end
    uv run python scripts/train_cartpole.py --algo all --render-every 0   # all 5, one after another
    uv run python scripts/train_cartpole.py --help            # options (budget, learning rates, ...)
    uv run python scripts/train_cartpole.py --env double_cartpole --algo ppo   # two sticks (rl/double_cartpole.py)

--algo (each adds one thing to the one before; rl/ppo.py):
    batch_ac  5 episodes → 1 gradient step → throw away (rl/explain.py)
    epochs    + 10 epochs: 10 gradient steps on the same batch, plain loss −A·log π
    ratio     + importance ratio r = π/π_old: loss −r·A (corrects for the data coming from π_old, no limit)
    clip      + clip: loss −min(r·A, clip(r, 1−ε, 1+ε)·A), ε = 0.2 (stops the push once r is past 1±ε)
    gae       + GAE λ = 0.95: A from the critic's one-step errors δ instead of G − V
    ppo       + 2,048-step batches, minibatches of 64, normalised advantages, lr 1e-3 / 3e-3
Every algorithm trains for the same --budget of environment steps (default 200,000).

The LATEST run of each algorithm is saved in runs/cli/<algo>/ (double cart-pole: runs/double_cartpole/<algo>/),
cleared when that algorithm runs again:
    log.csv, settings.json, status.json, plot.png, scorecard.json, final_weights.pt,
    episodes/iterNNNN.json + .gif   the current policy playing one episode at iteration 1, every
                                    --record-every iterations and at the end (JSON = for the web page)
    episodes/good.json + .gif       the hand-written good policy on the same start, for comparison
    per_epoch.csv                   inside every iteration, after each epoch: KL(π_old ‖ π), the share of
                                    steps past the clip limit (A > 0 and r > 1.2, or A < 0 and r < 0.8),
                                    the ratio r on steps with A > 0 (mean, 90th percentile) and A < 0
                                    (mean, 10th percentile), the largest and smallest r
The web page http://localhost:3000 shows and compares them (double cart-pole: http://localhost:3000/double),
live while training.

One batch_ac iteration (same code as the Learn page, rl/explain.py):
    1. the FROZEN policy plays --batch complete episodes
    2. critic target per step: the discounted return that actually followed,
           G_t = r_t + γ r_t+1 + γ² r_t+2 + …   (0 after the pole falls, V(s_T) after the 500-step limit)
       advantage per step: A_t = G_t − V(s_t)
    3. one critic gradient step on  mean ½ (G_t − V(s_t))²
    4. one actor gradient step on   −mean A_t · log π(a_t|s_t)

Log columns:
    ret     mean episode return of the batch (played by the policy before this update)
    Lv      critic loss, before → after its gradient step on this batch
    Lpi     actor loss
    H       policy entropy: 0.69 = coin flip, 0 = always the same action
    EV      critic explained variance on the batch: 1 = V predicts G, 0 = no better than a constant
    |A|     mean |advantage|: the size of the actor's teaching signal
    KL      how much this update changed the policy: mean KL(π_old ‖ π_new) over the batch states
    clip_frac  share of batch steps where π(a|s) moved by more than ±20 % in this update (the clip's limit)
    agree   % of batch states where the policy's most likely action = the reference policy's action
    eval    every --eval-every iterations: return of the current policy, always taking its most likely
            action, on 10 fixed start states (500 = perfect)

Actor quality that does NOT grow with the episode length (the actor loss ≈ |A| × entropy does):
    p_good  average probability the actor gives to the good policy's action (0.5 = coin flip, 1 = perfect)
    ce_good cross-entropy to the good policy: mean −log π(good action | s)  (0.69 = coin flip, 0 = perfect).
            A real classification loss, possible only because we know a good policy here.
    dir     did this update move the right way? correlation between A_t and the change of
            log π(a_t|s_t) caused by the update (> 0: higher-advantage actions gained more probability)

Critic quality that does NOT grow with the episode length (the raw loss does):
    rel     relative error  √(2·Lv) / RMS(G): how wrong V is as a fraction of the size of the returns
            (on a FRESH batch: the critic has not trained on it yet)
    corr    correlation(V, G): does the critic rank states right, whatever the scale? (1 = perfectly)
    every --truth-every iterations, a ground-truth check:
            take 10 states from the batch, play 10 episodes from each with the current policy, average
            their discounted returns = the TRUE value V^π(s) of that state. Then
            true_err = RMS(V − V^π) / RMS(V^π)   the critic's real error, single-return noise removed
            noise    = RMS(scatter of single returns around V^π) / RMS(V^π)
                       = how much of `rel` NO critic can remove (the policy samples its actions)

Reference policy ("a good policy we know"): push right if  pole angle + 0.5 × angular velocity > 0,
else left. It scores 500 (the maximum) on every start state, so it is optimal for this task.
"""

from __future__ import annotations

import argparse
import math
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

from rl.explain import ExplainSession, LearnConfig  # noqa: E402
from rl.ppo import VARIANTS, PPOConfig, PPOSession  # noqa: E402

# The ladder: our batch actor-critic, then one more piece of PPO per rung.
ALGOS = {
    "batch_ac": "Batch Actor-Critic (1 gradient step per batch)",
    "epochs": "+ 10 epochs on each batch (plain −A·log π)",
    "ratio": "+ importance ratio r = π/π_old (loss −r·A, no clip)",
    "clip": "+ clip ε = 0.2 (loss −min(r·A, clip(r)·A))",
    "gae": "+ GAE λ = 0.95",
    "ppo": "PPO (2,048-step batches, minibatches of 64, advantage normalisation)",
}


def make_session(algo: str, args):
    """Build the algorithm. Options left at None use the algorithm's own defaults."""
    pick = lambda value, default: default if value is None else value  # noqa: E731
    if algo == "batch_ac":
        cfg = LearnConfig(env_id=SPEC.env_id, seed=args.seed, batch_episodes=pick(args.batch, 5),
                          hidden=pick(args.hidden, SPEC.hidden), actor_lr=pick(args.actor_lr, 0.01),
                          critic_lr=pick(args.critic_lr, 0.03), gamma=args.gamma, optimizer=args.optimizer,
                          shaping=SPEC.shaping, env_kwargs=SPEC.env_kwargs)
        return ExplainSession(cfg), cfg.model_dump()
    base = {**PPOConfig().model_dump(), **VARIANTS[algo]}
    cfg = PPOConfig(**{**base, "env_id": SPEC.env_id, "shaping": SPEC.shaping, "env_kwargs": SPEC.env_kwargs, "seed": args.seed, "gamma": args.gamma, "hidden": pick(args.hidden, SPEC.hidden),
                       "actor_lr": pick(args.actor_lr, base["actor_lr"]), "critic_lr": pick(args.critic_lr, base["critic_lr"]),
                       **({"batch_episodes": args.batch} if args.batch is not None and base["batch_episodes"] else {})})
    return PPOSession(cfg), cfg.model_dump()
from rl.logger import runs_dir  # noqa: E402
from rl.double_cartpole import from_obs, reference_controller  # noqa: E402
from rl.envs import potential  # noqa: E402


class EnvSpec:
    """Everything that differs between the environments; chosen with --env."""

    def __init__(self, key: str):
        self.key = key
        # CartPole-like: success = still balancing at the time limit, longer is better.
        # MountainCar: success = reaching the flag (the episode TERMINATES), shorter is better.
        mc = key.startswith("mountaincar")
        self.success_is_terminal = mc
        self.n_actions = 3 if mc else 2
        self.quality_steps = 200 if mc else 10_000  # default length of the final test
        # reward shaping used in TRAINING only; evaluation and the scorecard use the official reward
        self.shaping = "energy" if key == "mountaincar_energy" else "none"
        self.measure = "length"  # what an episode is judged by: its length, or its score (return)
        self.can_set_state = True  # the ground-truth critic check needs to start rollouts in a given state
        self.budget, self.hidden = 200_000, 8  # defaults for --budget and --hidden
        self.env_kwargs: dict = {}  # gym.make options (LunarLander: wind, turbulence, gravity)
        if key.startswith("lunarlander"):
            from gymnasium.envs.box2d.lunar_lander import heuristic

            class _Discrete:  # Gymnasium's heuristic only reads env.unwrapped.continuous
                continuous = False

                @property
                def unwrapped(self):
                    return self

            self.env_id, self.folder = "LunarLander-v3", key
            self.title = "LunarLander + wind" if key == "lunarlander_wind" else "LunarLander"
            if key == "lunarlander_wind":  # Gymnasium's own wind: a steady push + random gusts
                self.env_kwargs = {"enable_wind": True, "wind_power": 15.0, "turbulence_power": 1.5}
            self.measure, self.n_actions, self.quality_steps = "score", 4, 1000
            self.can_set_state = False  # Box2D: terrain, legs and joints are not in the 8 numbers
            self.budget, self.hidden = 1_000_000, 64  # a harder task: more steps, a bigger network
            self.reference_text = "Gymnasium's hand-written heuristic"
            self.reference = lambda s: int(heuristic(_Discrete(), np.asarray(s)))
            self.angles = lambda s: abs(float(s[4]))  # for the final test: mean |lander angle|
        elif key == "cartpole":
            self.env_id, self.title, self.folder = "CartPole-v1", "CartPole", "cli"
            self.reference_text = "angle + 0.5·ang_vel > 0 → right"
            self.reference = lambda s: 1 if s[2] + 0.5 * s[3] > 0 else 0
            self.angles = lambda s: abs(float(s[2]))  # for the quality test: mean |pole angle|
        elif mc:
            self.env_id, self.folder = "MountainCar-v0", key
            self.title = "MountainCar + energy bonus" if self.shaping == "energy" else "MountainCar"
            self.reference_text = "push in the direction you are moving"
            self.reference = lambda s: 2 if s[1] >= 0 else 0  # 0 = left, 1 = no push, 2 = right
            self.angles = lambda s: 0.0
        else:
            joint = key == "double_cartpole_joint"
            self.env_id = "DoubleCartPoleJoint-v0" if joint else "DoubleCartPole-v0"
            self.title = "Double CartPole (joint rule)" if joint else "Double CartPole"
            self.folder = key
            self.reference_text = "the LQR controller, ±10 N"
            ctrl = reference_controller()
            self.reference = lambda s: ctrl(np.asarray(s))
            # mean |stick angle| from vertical: both sticks (θ1 and θ2 = θ1 + φ)
            self.angles = lambda s: 0.5 * (abs(float(s[2])) + abs(float(s[2] + s[4])))

    def succeeded(self, terminated: bool, score: float | None = None) -> bool:
        """Did this episode succeed? CartPole-like: it was NOT terminated (reached the time limit).
        MountainCar: it terminated (the flag). Score-judged envs (LunarLander): score ≥ 200."""
        if self.measure == "score":
            return score is not None and score >= 200
        return terminated if self.success_is_terminal else not terminated

    def good_batch(self, row: dict) -> bool:
        """A batch that counts as 'solved' (CartPole: mean length ≥ 475; MountainCar: all reach the flag;
        LunarLander: mean score ≥ 200)."""
        if self.measure == "score":
            return (row.get("score") or -1e9) >= 200
        return (row.get("success") or 0) >= 0.95 if self.success_is_terminal else row["ret"] >= 475

    def bad_batch(self, row: dict) -> bool:
        """A drop after solving (CartPole: mean length < 250; MountainCar: fewer than half reach the flag;
        LunarLander: mean score below 0)."""
        if self.measure == "score":
            return (row.get("score") or 0) < 0
        return (row.get("success") or 0) < 0.5 if self.success_is_terminal else row["ret"] < 250

    def solved_eval(self, row: dict) -> bool:
        """The greedy policy succeeds on all 10 eval starts (LunarLander: averages ≥ 200 over them)."""
        if self.measure == "score":
            return row.get("eval_score") is not None and row["eval_score"] >= 200
        if row.get("eval_success") is not None:
            return row["eval_success"] >= 1.0
        return row["eval"] is not None and row["eval"] >= 500  # logs from before eval_success existed

    def outcome(self, env, obs, terminated: bool) -> str | None:
        """How an episode ended, where the length does not say it (LunarLander)."""
        if not self.key.startswith("lunarlander"):
            return None
        u = env.unwrapped
        if not terminated:
            return "time limit"  # still flying (hovering) after 1,000 steps
        if u.game_over:
            return "crashed"  # the body touched the ground
        if abs(float(obs[0])) >= 1.0:
            return "flew away"
        return "landed"  # came to rest

    def set_state(self, env, st) -> None:
        """Put an unwrapped env exactly into the observed state `st`."""
        if self.key == "cartpole" or self.key.startswith("mountaincar"):
            env.state = np.array(st, dtype=np.float64)
        else:
            env.q, env.qd = from_obs(np.asarray(st))


SPEC = EnvSpec("cartpole")  # replaced in main() by --env


def reference_action(state) -> int:
    """The hand-written good policy of the current environment."""
    return SPEC.reference(state)


@torch.no_grad()
def greedy_action(session: ExplainSession, state) -> int:
    x = torch.as_tensor(session.encoder(np.asarray(state), 0))
    return int(session.actor(x).argmax())


def evaluate(policy, episodes: int = 10, render: bool = False) -> tuple[float, float, float]:
    """`policy(state) -> action` on fixed start states: (mean episode LENGTH, share that succeeded,
    mean SCORE = return with the official reward). CartPole: length = score (500 = perfect).
    MountainCar: length = steps to the flag (200 = never)."""
    env = gym.make(SPEC.env_id, render_mode="human" if render else None, **SPEC.env_kwargs)
    lengths, wins, scores = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        n, score = 0, 0.0
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            n += 1
            score += float(r)
            if term or trunc:
                break
        lengths.append(n)
        scores.append(score)
        wins.append(SPEC.succeeded(term, score))
    env.close()
    return float(np.mean(lengths)), float(np.mean(wins)), float(np.mean(scores))


@torch.no_grad()
def record_episode(session: ExplainSession | None, it: int, folder: Path, name: str | None = None) -> dict:
    """Play one episode from a fixed start state and save it: trajectory as JSON (the web page
    replays it) and frames as a GIF. session = None plays the hand-written good policy."""
    from PIL import Image

    env = gym.make(SPEC.env_id, render_mode="rgb_array", **SPEC.env_kwargs)
    obs, _ = env.reset(seed=10_000)  # the same start as the first eval episode
    states, actions, probs, frames, good_actions = [], [], [], [], []
    score = 0.0
    while True:
        if session is None:  # the good policy: certain of its action
            a = reference_action(obs)
            p = [0.0] * int(env.action_space.n)
            p[a] = 1.0
        else:  # the agent's policy, always taking its most likely action
            pt = F.softmax(session.actor(torch.as_tensor(session.encoder(np.asarray(obs), 0))), -1)
            a, p = int(pt.argmax()), pt.tolist()
        states.append(np.asarray(obs, dtype=float).tolist())
        actions.append(a)
        probs.append(p)
        good_actions.append(reference_action(obs))
        if len(states) % 2 == 1:  # every 2nd frame keeps the GIF small
            frames.append(Image.fromarray(env.render()).resize((300, 200)))
        obs, r, term, trunc, _ = env.step(a)
        score += float(r)
        if term or trunc:
            break
    outcome = SPEC.outcome(env, obs, term)
    env.close()
    name = name or f"iter{it:04d}"
    info = {"name": name, "iter": it, "steps": len(actions), "terminated": bool(term),
            "kind": "good" if session is None else "greedy", "score": score, "outcome": outcome}
    (folder / f"{name}.json").write_text(json.dumps({**info, "states": states, "actions": actions, "probs": probs, "good_actions": good_actions,
                                                    "final_state": np.asarray(obs, dtype=float).tolist()}))
    frames[0].save(folder / f"{name}.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
    return info


def record_training_episode(tr: dict, it: int, folder: Path) -> dict | None:
    """Save one REAL training episode of this batch: the actions as they were sampled from π (not the most
    likely ones), with the probabilities at that moment. It starts wherever that episode started."""
    eps = tr["episodes"]
    whole = [e for e in eps if e.get("fresh", True) and e.get("end", "terminated" if e.get("terminated") else "truncated") != "cut"
             and e.get("actions")]
    ep = whole[0] if whole else max((e for e in eps if e.get("actions")), key=lambda e: len(e["actions"]), default=None)
    if ep is None:
        return None
    terminated = bool(ep["terminated"]) if "terminated" in ep else ep.get("end") == "terminated"
    states = ep["states"]
    info = {"name": f"train{it:04d}", "iter": it, "steps": len(ep["actions"]), "terminated": terminated, "kind": "training",
            "complete": bool(whole), "score": ep.get("ret")}
    (folder / f"{info['name']}.json").write_text(json.dumps({
        **info, "states": states, "actions": [int(a) for a in ep["actions"]], "probs": ep["probs"],
        "good_actions": [reference_action(np.asarray(s)) for s in states]}))
    return info


@torch.no_grad()
def quality_test(policy, starts: int = 20, max_steps: int = 10_000) -> dict:
    """The final policy, tested harder than training: up to `max_steps` (training stops at 500)
    from `starts` different start states. Survival, centering, calmness and action switching.
    MountainCar: 'survived' = reached the flag, and the curve = share that reached it by step t."""
    env = gym.make(SPEC.env_id, max_episode_steps=max_steps, **SPEC.env_kwargs)
    lengths, abs_x, abs_th, ok, scores, outcomes = [], [], [], [], [], []
    switches = survived = fell_cart = fell_pole = 0
    for i in range(starts):
        obs, _ = env.reset(seed=20_000 + i)
        prev, n, score = None, 0, 0.0
        while True:
            a = policy(obs)
            switches += int(prev is not None and a != prev)
            prev = a
            abs_x.append(abs(float(obs[0])))
            abs_th.append(SPEC.angles(obs))
            obs, r, term, trunc, _ = env.step(a)
            n += 1
            score += float(r)
            if term or trunc:
                break
        lengths.append(n)
        scores.append(score)
        outcomes.append(SPEC.outcome(env, obs, term))
        ok.append(SPEC.succeeded(term, score))
        if ok[-1]:
            survived += 1
        elif SPEC.success_is_terminal or SPEC.measure == "score":
            fell_pole += 1  # MountainCar: did not reach the flag; LunarLander: scored below 200
        elif abs(obs[0]) > 2.4:
            fell_cart += 1
        else:
            fell_pole += 1
    env.close()
    grid = np.linspace(0, max_steps, 101)
    score_grid = np.linspace(-300, 320, 125)
    return {
        "starts": starts, "max_steps": max_steps, "lengths": lengths, "survived": survived,
        "scores": scores, "median_score": float(np.median(scores)),
        "outcomes": {o: outcomes.count(o) for o in sorted({o for o in outcomes if o is not None})},
        "fell_cart": fell_cart, "fell_pole": fell_pole, "median_len": float(np.median(lengths)),
        "mean_abs_x": float(np.mean(abs_x)), "mean_abs_theta": float(np.mean(abs_th)),
        "switch_rate": switches / max(1, sum(lengths) - starts),
        # the curve: CartPole-like = share still balancing at step t; MountainCar = share at the flag by step t;
        # LunarLander = share of the starts scoring at least s
        "survival_x": (score_grid if SPEC.measure == "score" else grid).tolist(),
        "survival": ([float(np.mean([g >= s for g in scores])) for s in score_grid] if SPEC.measure == "score"
                     else [float(np.mean([o and L <= g for o, L in zip(ok, lengths)])) for g in grid] if SPEC.success_is_terminal
                     else [float(np.mean([L >= g for L in lengths])) for g in grid]),
    }


def speed_stability(rows: list[dict]) -> dict:
    """Speed: environment steps until the policy first scores 500. Stability: what happens after."""
    first_eval = next((r for r in rows if r["eval"] is not None and SPEC.solved_eval(r)), None)
    first_batch = next((r for r in rows if SPEC.good_batch(r)), None)
    after = rows[rows.index(first_batch) + 1:] if first_batch else []
    evals_after = [r for r in after if r["eval"] is not None]
    return {
        "total_env_steps": rows[-1]["env_steps"], "iters": rows[-1]["iter"],
        "eval500_iter": first_eval["iter"] if first_eval else None,
        "eval500_steps": first_eval["env_steps"] if first_eval else None,
        "batch475_iter": first_batch["iter"] if first_batch else None,
        "batch475_steps": first_batch["env_steps"] if first_batch else None,
        "stay_475": float(np.mean([SPEC.good_batch(r) for r in after])) if after else None,
        "drops_below_250": sum(SPEC.bad_batch(r) for r in after),
        "evals_500_after": float(np.mean([SPEC.solved_eval(r) for r in evals_after])) if evals_after else None,
    }


@torch.no_grad()
def true_values(session: ExplainSession, states: list, rollouts: int = 10, horizon: int = 600, seed: int = 0):
    """Monte Carlo ground truth: average discounted return of the CURRENT policy from each state.

    No time limit (the critic bootstraps at the 500-step cut, so it estimates the never-ending
    task); 600 steps cover it (0.99^600 ≈ 0.002). Returns (mean, std) per state."""
    env = gym.make(SPEC.env_id, **SPEC.env_kwargs).unwrapped  # no TimeLimit wrapper
    gen = torch.Generator().manual_seed(seed)
    g = session.cfg.gamma
    means, stds = [], []
    for k, st in enumerate(states):
        rets = []
        for r in range(rollouts):
            env.reset(seed=seed + 1000 * k + r)
            SPEC.set_state(env, st)  # start exactly in this state
            obs, ret, disc = np.array(st), 0.0, 1.0
            for _ in range(horizon):
                p = F.softmax(session.actor(torch.as_tensor(session.encoder(obs, 0))), -1)
                prev = obs
                obs, rew, term, _, _ = env.step(int(torch.multinomial(p, 1, generator=gen)))
                if SPEC.shaping != "none":  # the same bonus the critic was trained with
                    rew += 10.0 * (g * (0.0 if term else potential(obs, SPEC.env_id, SPEC.shaping))
                                   - potential(prev, SPEC.env_id, SPEC.shaping))
                ret += disc * rew
                disc *= g
                if term:
                    break
            rets.append(ret)
        means.append(float(np.mean(rets)))
        stds.append(float(np.std(rets)))
    env.close()
    return np.array(means), np.array(stds)


def web_page() -> str:
    return "http://localhost:3000" + {"cartpole": "", "double_cartpole": "/double", "mountaincar": "/mountaincar",
                                      "mountaincar_energy": "/mountaincar-energy", "lunarlander": "/lunarlander",
                                      "lunarlander_wind": "/lunarlander-wind"}.get(
        SPEC.key, "/double-joint")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=["cartpole", "double_cartpole", "double_cartpole_joint", "mountaincar", "mountaincar_energy",
                                      "lunarlander", "lunarlander_wind"],
                    default="cartpole",
                    help="cartpole (one stick), double_cartpole (two sticks on a free hinge, each ≤ 15° from "
                         "vertical), double_cartpole_joint (lower stick ≤ 30°, top stick bent ≤ 20° from it) "
                         "mountaincar (official MountainCar-v0: −1 per step, 200-step limit) or mountaincar_energy "
                         "(+ a bonus for gaining energy in training: 10·(γΦ(s′) − Φ(s))) or lunarlander (LunarLander-v3, "
                         "judged by its score, 200 = solved) or lunarlander_wind (the same with wind and turbulence)")
    ap.add_argument("--algo", choices=[*ALGOS, "all"], default="batch_ac",
                    help="batch_ac | epochs | ratio | clip | gae | ppo, or all of them one after another")
    ap.add_argument("--budget", type=int, default=None,
                    help="environment steps to train for, the same for every algorithm (default: 200,000; LunarLander 1,000,000)")
    ap.add_argument("--iters", type=int, default=None, help="optional maximum number of iterations")
    ap.add_argument("--batch", type=int, default=None, help="episodes per batch (episode-batch algorithms)")
    ap.add_argument("--hidden", type=int, default=None, help="hidden units (one hidden layer); default 8, LunarLander 64")
    ap.add_argument("--actor-lr", type=float, default=None, help="default: the algorithm's own")
    ap.add_argument("--critic-lr", type=float, default=None, help="default: the algorithm's own")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--render-every", type=int, default=25, help="show the current policy playing every N iterations (0 = never)")
    ap.add_argument("--render-every-steps", type=int, default=0,
                    help="instead: show it each time training passes another N environment steps, so every algorithm is "
                         "shown at the same points (e.g. 100000 = 10 windows in 1M steps); replaces --render-every")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="plot to <run folder>/plot.png and open it at the end")
    ap.add_argument("--truth-every", type=int, default=10, help="ground-truth critic check every N iterations (0 = never)")
    ap.add_argument("--record-every", type=int, default=10, help="save one episode of the current policy every N iterations")
    ap.add_argument("--open", action=argparse.BooleanOptionalAction, default=True, help="open the plot at the end")
    ap.add_argument("--quality-steps", type=int, default=None,
                    help="length of the final-policy test (default: 10,000 for the cart-poles, 200 for MountainCar)")
    ap.add_argument("--quality-starts", type=int, default=20, help="start states in the final-policy test")
    ap.add_argument("--gravity", type=float, default=None, help="LunarLander: gravity (default −10; −5 = moon-like)")
    ap.add_argument("--wind-power", type=float, default=None, help="lunarlander_wind: steady wind strength (default 15)")
    ap.add_argument("--turbulence", type=float, default=None, help="lunarlander_wind: random gusts (default 1.5)")
    args = ap.parse_args()
    global SPEC
    SPEC = EnvSpec(args.env)
    if args.quality_steps is None:
        args.quality_steps = SPEC.quality_steps
    if args.budget is None:
        args.budget = SPEC.budget
    if args.gravity is not None:
        if not args.env.startswith("lunarlander"):
            ap.error("--gravity is for the LunarLander environments")
        SPEC.env_kwargs["gravity"] = args.gravity
    for flag, key in (("wind_power", "wind_power"), ("turbulence", "turbulence_power")):
        if getattr(args, flag) is not None:
            if args.env != "lunarlander_wind":
                ap.error(f"--{flag.replace('_', '-')} needs --env lunarlander_wind")
            SPEC.env_kwargs[key] = getattr(args, flag)
    if not SPEC.can_set_state:
        args.truth_every = 0  # no ground-truth critic check (rollouts cannot start in a given state)
    if args.algo == "all":  # the whole ladder, one after another, no windows
        args.render_every, args.render_every_steps, args.open = 0, 0, False
        for algo in ALGOS:
            print(f"\n========== {algo}: {ALGOS[algo]} ==========")
            train(algo, args)
        print(f"\nDone. Compare them on {web_page()}")
    else:
        train(args.algo, args)


def train(algo: str, args) -> None:
    session, resolved = make_session(algo, args)
    out_dir = runs_dir() / SPEC.folder / algo  # the latest run of this algorithm in this environment
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "episodes").mkdir(parents=True)
    csv_path = out_dir / "log.csv"
    (out_dir / "settings.json").write_text(json.dumps(
        {"algo": algo, "algo_name": ALGOS[algo], "env": SPEC.key, "env_id": SPEC.env_id, "budget": args.budget,
         "seed": args.seed, "config": resolved}, indent=2))
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    def status(state: str, it: int, steps: int = 0) -> None:
        (out_dir / "status.json").write_text(json.dumps(
            {"state": state, "iter": it, "env_steps": steps, "budget": args.budget, "started": started}))

    status("running", 0)
    record_episode(None, 0, out_dir / "episodes", name="good")  # the hand-written policy, for comparison

    target = ("steps to the flag, lower is better (200 = never)" if SPEC.success_is_terminal
              else "score, 200 = solved" if SPEC.measure == "score" else "steps balanced (500 = perfect)")
    ref_len, ref_ok, ref_score = evaluate(reference_action)
    ref_v = ref_score if SPEC.measure == "score" else ref_len
    print(f"{SPEC.title} · reference policy ({SPEC.reference_text}): eval {ref_v:.0f} {target}, succeeds {ref_ok:.0%}")
    un_len, un_ok, un_score = evaluate(lambda s: greedy_action(session, s))
    un_v = un_score if SPEC.measure == "score" else un_len
    hidden = SPEC.hidden if args.hidden is None else args.hidden
    print(f"untrained policy: eval {un_v:.0f}, succeeds {un_ok:.0%}   (budget {args.budget:,} env steps, hidden {hidden})\n")
    cols = ["iter", "ret", "env_steps", "Lv", "Lv_after", "rel", "corr", "Lpi", "H", "EV", "absA", "KL", "clip_frac", "agree", "p_good",
            "ce_good", "dir", "eval", "success", "eval_success", "score", "eval_score", "max_x", "true_err", "noise", "V_mean",
            "Vtrue_mean"]
    print(f"{'iter':>4} {'ret':>6} {'Lv (before → after)':>22} {'rel':>5} {'corr':>5} {'Lpi':>8} {'H':>5} {'EV':>6} "
          f"{'|A|':>6} {'KL':>8} {'agree':>6} {'p_good':>6} {'ce_good':>7} {'dir':>5} {'eval':>5}")

    rows: list[dict] = []
    fig = axes = None
    live = False
    if args.plot:
        import matplotlib
        import matplotlib.pyplot as plt

        live = matplotlib.get_backend().lower() != "agg"
        if not live:
            print("(the plot opens when training ends; for a plot that updates DURING training: sudo apt install python3-tk)\n")
        plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        fig.suptitle(f"{SPEC.title} · {ALGOS[algo]}")

    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    # inside each iteration: how far the policy is from π_old after every epoch (1 row per epoch)
    f_ep = open(out_dir / "per_epoch.csv", "w", newline="")
    ep_cols = ["iter", "epoch", "kl", "past", "r_pos", "r_neg", "r_hi", "r_lo", "r_max", "r_min"]
    ep_writer = csv.DictWriter(f_ep, fieldnames=ep_cols)
    ep_writer.writeheader()
    it = 0
    env_steps = 0  # environment steps used so far: the fair x-axis between algorithms
    completed = False
    try:
        last = False
        while not last:
            it += 1
            tr = session.iterate()  # collect a batch → gradient step(s) → discard
            s = tr["summary"]
            last = env_steps + s["steps"] >= args.budget or (args.iters is not None and it >= args.iters)
            p_old = np.array(tr["actor"]["probs_before"])  # π_k on the batch states
            p_new = np.array(tr["actor"]["probs_after"])  # π_k+1 on the same states
            kl = float(np.mean(np.sum(p_old * (np.log(p_old + 1e-12) - np.log(p_new + 1e-12)), axis=1)))
            states = [st for ep in tr["episodes"] for st in ep["states"]]
            agree = float(np.mean([int(np.argmax(p)) == reference_action(st) for p, st in zip(p_old, states)]))
            good = np.array([p[reference_action(st)] for p, st in zip(p_old, states)])  # π(good action | s)
            p_good, ce_good = float(good.mean()), float(np.mean(-np.log(good + 1e-12)))
            adv = np.array(tr["advantages"])
            dlogp = np.array(tr["actor"]["logp_after"]) - np.array(tr["actor"]["logp_before"])
            clip_frac = float(np.mean(np.abs(np.exp(dlogp) - 1) > 0.2))  # share of steps where π(a|s) moved > ±20 %
            direction = float(np.corrcoef(adv, dlogp)[0, 1]) if adv.std() > 1e-9 and dlogp.std() > 1e-12 else float("nan")
            ev = s["ev"] if s["ev"] is not None else float("nan")
            G, V = np.array(tr["returns"]), np.array(tr["values"])  # V: critic BEFORE its step, on fresh data
            rel = float(np.sqrt(np.mean((G - V) ** 2)) / (np.sqrt(np.mean(G**2)) + 1e-12))
            corr = float(np.corrcoef(G, V)[0, 1]) if G.std() > 1e-9 and V.std() > 1e-9 else float("nan")
            prev_steps = env_steps
            env_steps += s["steps"]
            # episode LENGTH (CartPole: = return) and the share of the finished episodes that succeeded
            if "finished_lengths" in s:  # PPO session: the episodes that ended in this batch
                lens, terms, rets = s["finished_lengths"], s["finished_terminated"], s["finished_returns"]
            else:  # batch actor-critic: complete episodes
                lens = [len(ep["states"]) for ep in tr["episodes"]]
                terms = [bool(ep["terminated"]) for ep in tr["episodes"]]
                rets = [float(ep["ret"]) for ep in tr["episodes"]]  # official reward (no shaping bonus)
            mean_len = float(np.mean(lens)) if lens else (rows[-1]["ret"] if rows else float("nan"))
            success = (float(np.mean([SPEC.succeeded(t, g) for t, g in zip(terms, rets)])) if terms
                       else (rows[-1]["success"] if rows else None))
            score = float(np.mean(rets)) if rets else (rows[-1]["score"] if rows else None)
            row = {"iter": it, "ret": mean_len, "env_steps": env_steps, "Lv": s["critic_loss"], "Lv_after": s["critic_loss_after"],
                   "rel": rel, "corr": corr, "Lpi": s["actor_loss"], "H": s["entropy"], "EV": ev,
                   "absA": s["mean_abs_adv"], "KL": kl, "clip_frac": clip_frac, "agree": agree, "p_good": p_good, "ce_good": ce_good,
                   "dir": direction, "eval": None, "success": success, "eval_success": None,
                   "score": score, "eval_score": None,
                   # MountainCar: the highest position each episode (piece) reached, averaged; the flag is at 0.5
                   "max_x": (float(np.mean([max(st[0] for st in ep["states"]) for ep in tr["episodes"] if ep["states"]]))
                             if SPEC.success_is_terminal else None),
                   "true_err": None, "noise": None, "V_mean": None, "Vtrue_mean": None}
            if it % args.eval_every == 0 or it == 1 or last:
                row["eval"], row["eval_success"], row["eval_score"] = evaluate(lambda st: greedy_action(session, st))
            truth_line = ""
            if args.truth_every and (it % args.truth_every == 0 or it == 1):
                probe = [states[i] for i in np.linspace(0, len(states) - 1, 10).astype(int)]
                vt, vs = true_values(session, probe, seed=args.seed + it)
                with torch.no_grad():
                    vc = session.critic(torch.as_tensor(session.encoder(np.array(probe), 0))).numpy()
                scale = np.sqrt(np.mean(vt**2)) + 1e-12
                row.update(true_err=float(np.sqrt(np.mean((vc - vt) ** 2)) / scale),
                           noise=float(np.sqrt(np.mean(vs**2)) / scale), V_mean=float(vc.mean()), Vtrue_mean=float(vt.mean()))
                truth_line = (f"     ↳ critic vs TRUE value on 10 states: critic says {row['V_mean']:.1f}, truth {row['Vtrue_mean']:.1f} "
                              f"→ true error {row['true_err']:.0%} of the value (single-return noise {row['noise']:.0%})")
            rows.append(row)
            writer.writerow(row)
            f.flush()
            if "per_epoch" in s:
                moves = s["per_epoch"]
            else:  # batch actor-critic: one gradient step = one "epoch"
                r = np.exp(dlogp)
                pos, neg = adv > 0, adv < 0
                moves = [{"kl": kl, "past": float(np.mean((pos & (r > 1.2)) | (neg & (r < 0.8)))),
                          "r_pos": float(r[pos].mean()) if pos.any() else None,
                          "r_neg": float(r[neg].mean()) if neg.any() else None,
                          "r_hi": float(np.quantile(r[pos], 0.9)) if pos.any() else None,
                          "r_lo": float(np.quantile(r[neg], 0.1)) if neg.any() else None,
                          "r_max": float(r.max()), "r_min": float(r.min())}]
            for e, m in enumerate(moves, start=1):
                ep_writer.writerow({"iter": it, "epoch": e, **m})
            f_ep.flush()
            ev_txt = f"{ev:6.2f}" if ev == ev else "   n/a"
            ev_shown = row["eval_score"] if SPEC.measure == "score" else row["eval"]  # LunarLander: the eval score
            eval_txt = f"{ev_shown:5.0f}" if ev_shown is not None else ""
            corr_txt = f"{corr:5.2f}" if corr == corr else "  n/a"
            shown = score if SPEC.measure == "score" else mean_len  # LunarLander: the batch's mean score
            print(f"{it:4d} {shown:6.1f} {s['critic_loss']:10.2f} → {s['critic_loss_after']:9.2f} {rel:5.0%} {corr_txt} "
                  f"{s['actor_loss']:8.3f} {s['entropy']:5.2f} {ev_txt} {s['mean_abs_adv']:6.2f} {kl:8.5f} {agree:6.0%} "
                  f"{p_good:6.2f} {ce_good:7.3f} {direction:5.2f} {eval_txt}")
            if truth_line:
                print(truth_line)

            if it == 1 or it % args.record_every == 0 or last:
                ep = record_episode(session, it, out_dir / "episodes")
                record_training_episode(tr, it, out_dir / "episodes")
                print(f"     ↳ recorded the current policy playing: {ep['steps']} steps (episodes/{ep['name']}.gif)")
            by_steps = args.render_every_steps and env_steps // args.render_every_steps > prev_steps // args.render_every_steps
            by_iters = not args.render_every_steps and args.render_every and it % args.render_every == 0
            if by_steps or by_iters:  # independent of --record-every; always the same start and the most likely action
                print(f"     ↳ showing the current policy in a window (iteration {it}, {env_steps:,} env steps) …")
                try:
                    w_len, _, w_score = evaluate(lambda st: greedy_action(session, st), episodes=1, render=True)
                    shown = f"score {w_score:.0f}" if SPEC.measure == "score" else f"{w_len:.0f} steps"
                    print(f"     ↳ that episode: {shown}")
                except Exception as err:  # no display available
                    print(f"     ↳ cannot open a window ({err}); use --render-every 0")
                    args.render_every = args.render_every_steps = 0
            status("running", it, env_steps)
            if fig is not None and (live or last):
                _draw(fig, axes, rows, live)
    except KeyboardInterrupt:
        print("\nstopped by Ctrl-C")
        status("stopped", it, env_steps)
    else:
        completed = True
    finally:
        f.close()
        f_ep.close()

    if not rows:
        return
    _diagnose(rows)
    if completed:
        status("testing", it, env_steps)
        torch.save({"actor": session.actor.state_dict(), "critic": session.critic.state_dict()}, out_dir / "final_weights.pt")
        print(f"\nScorecard (final policy tested for up to {args.quality_steps} steps from {args.quality_starts} start states) …")
        card = {
            "speed_stability": speed_stability(rows),
            "quality": {
                "learned": quality_test(lambda st: greedy_action(session, st), args.quality_starts, args.quality_steps),
                "good": quality_test(reference_action, args.quality_starts, args.quality_steps),
            },
        }
        (out_dir / "scorecard.json").write_text(json.dumps(card))
        _print_scorecard(card)
        status("finished", it, env_steps)
    print(f"\n{ALGOS[algo]}: everything saved in {out_dir}   (web page: {web_page()})")
    if fig is not None:
        if not live:
            _draw(fig, axes, rows, live)
        png = out_dir / "plot.png"
        fig.savefig(png, dpi=110)
        print(f"plot saved: {png}")
        if live:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.show()
        elif args.open and shutil.which("explorer.exe"):  # WSL: open it in the Windows image viewer
            win_path = subprocess.run(["wslpath", "-w", str(png)], capture_output=True, text=True).stdout.strip()
            subprocess.run(["explorer.exe", win_path], check=False)


def _draw(fig, axes, rows: list[dict], live: bool) -> None:
    import matplotlib.pyplot as plt

    it = [r["iter"] for r in rows]
    ev_it = [r["iter"] for r in rows if r["eval"] is not None]
    t_it = [r["iter"] for r in rows if r["true_err"] is not None]
    panels = [
        (axes[0, 0], "1. Is the agent learning?  " + ("steps to the flag (200 = never reached)" if SPEC.success_is_terminal
                                                       else "score (200 = solved)" if SPEC.measure == "score"
                                                       else "episode length (500 = perfect)"), [
            ("batch: mean " + ("score" if SPEC.measure == "score" else "episode length"),
             [r["score" if SPEC.measure == "score" else "ret"] for r in rows], it),
            ("eval: most likely action, 10 fixed starts",
             [r["eval_score" if SPEC.measure == "score" else "eval"] for r in rows if r["eval"] is not None], ev_it)], False),
        (axes[0, 1], "2. Is the actor learning?  (does not grow with episode length)", [
            (f"P(good action): {1 / SPEC.n_actions:.2f} = random → 1 perfect", [r["p_good"] for r in rows], it),
            ("most likely action = good action", [r["agree"] for r in rows], it),
            (f"entropy ({math.log(SPEC.n_actions):.2f} = random, 0 = always the same)", [r["H"] for r in rows], it)], False),
        (axes[1, 0], "3. Critic loss ½(G−V)²  (grows with the returns)", [
            ("critic loss on the new batch", [r["Lv"] for r in rows], it)], True),
        (axes[1, 1], "4. Is the critic learning?  error as a fraction of the value", [
            ("TRUE error: |V − real average return|", [r["true_err"] for r in rows if r["true_err"] is not None], t_it),
            ("noise floor (no critic can go below)", [r["noise"] for r in rows if r["noise"] is not None], t_it),
            ("error on the batch (rel)", [r["rel"] for r in rows], it)], False),
    ]
    for ax, title, series, log in panels:
        ax.clear()
        for label, ys, xs in series:
            ax.plot(xs, ys, label=label, lw=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration (one batch + one update)", fontsize=8)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=8)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 1].set_ylim(0, 1.2)
    fig.tight_layout()
    if live:
        plt.pause(0.001)


def _print_scorecard(card: dict) -> None:
    ss = card["speed_stability"]
    fmt_steps = lambda v: f"{v:,}" if v is not None else "never"  # noqa: E731
    solved, good = (("all 10 eval starts reach the flag", "all batch episodes reach the flag") if SPEC.success_is_terminal
                    else ("eval score ≥ 200", "batch score ≥ 200") if SPEC.measure == "score"
                    else ("eval 500", "batch ≥ 475"))
    print(f"  speed      {solved}: first at {fmt_steps(ss['eval500_steps'])} env steps (iteration {ss['eval500_iter']}); "
          f"{good}: first at {fmt_steps(ss['batch475_steps'])} (iteration {ss['batch475_iter']})")
    if ss["stay_475"] is not None:
        print(f"  stability  after that: {ss['stay_475']:.0%} of batches stayed solved, {ss['drops_below_250']} drops")
    for name, q in card["quality"].items():
        if SPEC.measure == "score":
            print(f"  quality    {name:7s}: score ≥ 200 on {q['survived']}/{q['starts']} starts · median score {q['median_score']:.0f} · "
                  f"{', '.join(f'{v} {k}' for k, v in q['outcomes'].items())} · switches action {q['switch_rate']:.0%} of steps")
        elif SPEC.success_is_terminal:
            print(f"  quality    {name:7s}: reached the flag {q['survived']}/{q['starts']} within {q['max_steps']} steps · "
                  f"median {q['median_len']:.0f} steps · switches action {q['switch_rate']:.0%} of steps")
        else:
            print(f"  quality    {name:7s}: survived {q['survived']}/{q['starts']} · median {q['median_len']:.0f} steps · "
                  f"mean |x| {q['mean_abs_x']:.2f} m · mean |θ| {q['mean_abs_theta']:.3f} rad · switches action {q['switch_rate']:.0%} of steps")


def _diagnose(rows: list[dict]) -> None:
    """A few plain-language checks on the last 20 iterations."""
    last = rows[-20:]
    first = rows[:20]
    mean = lambda k, rs: float(np.nanmean([r[k] for r in rs]))  # noqa: E731
    print("\nIs it learning? (last 20 iterations vs first 20)")
    if SPEC.measure == "score":
        better = mean("score", last) > mean("score", first) + 50
        print(f"  batch score {mean('score', first):.0f} → {mean('score', last):.0f}"
              f"   ({'improving' if better else 'not clearly improving'})")
    else:
        better = (mean("ret", last) < 0.8 * mean("ret", first)) if SPEC.success_is_terminal else (mean("ret", last) > 1.5 * mean("ret", first))
        print(f"  batch episode length {mean('ret', first):.0f} → {mean('ret', last):.0f}"
              f"   ({'improving' if better else 'not clearly improving'})")
    print(f"  agreement with the reference policy {mean('agree', first):.0%} → {mean('agree', last):.0%}")
    if mean("EV", last) < 0.2:
        print("  ! critic explained variance is low: V does not predict the returns, so A is mostly noise")
    if mean("H", last) < 0.1:
        print("  ! entropy ~0: the policy is (nearly) deterministic and has stopped exploring")
    if mean("KL", last) < 1e-5:
        print("  ! the policy barely changes per update (tiny KL): advantages ~0 or learning rate too small")
    if mean("absA", last) < 0.05 * max(1.0, abs(mean("ret", last)) / 100):
        print("  ! |A| is tiny: the critic predicts the returns, so there is little left to learn from")


if __name__ == "__main__":
    main()
