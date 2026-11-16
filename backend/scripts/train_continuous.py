"""Train on a continuous-action MuJoCo task from the terminal (Walker2d): PPO, then the off-policy ladder.

    uv run python scripts/train_continuous.py --algo ppo                      # 1M steps (~17 min), window every 100k steps
    uv run python scripts/train_continuous.py --algo ddpg                     # 1M steps (~2 h: one update per env step)
    uv run python scripts/train_continuous.py --algo ddpg --threads 4         # faster updates, if the CPU is free
    uv run python scripts/train_continuous.py --help

--algo
    ppo   PPO with a Gaussian policy (rl/ppo_continuous.py): the actor outputs a mean torque per joint, σ is
          learned per joint; 2,048-step batches, 10 epochs × minibatches of 64, clip 0.2, GAE 0.95, observation
          normalisation and reward scaling, one Adam (lr 3e-4 → 0), gradient clip 0.5
    ddpg  DDPG (rl/offpolicy.py): DQN for continuous actions. A deterministic actor μ(s) (tanh) learns the argmax of
          a critic Q(s, a); replay buffer 1M, minibatches of 256, one update per env step, soft target networks
          (τ = 0.005), exploration noise N(0, 0.1²) added to μ(s), 25,000 random steps first. 2×256 ReLU networks.

The LATEST run of each algorithm is saved in runs/<env>/<algo>/, replaced by the next run:
    log.csv, settings.json, status.json, plot.png, scorecard.json, final_weights.pt (incl. the normalisation),
    episodes/iterNNNN.json (+ .gif) the current policy (mean action) from a fixed start, every --record-every
                                    iterations; the JSON holds the body's capsules and the 6 torques per frame;
                                    a GIF every --gif-every iterations
Watch it learn on the web page: http://localhost:3000/walker2d (live while training)
Watch a saved policy:  uv run python scripts/play_walker2d.py --policy trained

Log columns (one iteration = one batch of 2,048 env steps + its update):
    score     mean return of the episodes that ended in this batch (sampled actions; the real reward)
    len       their mean length (1,000 = the time limit, never fell)
    fell      share of them that ended by falling
    dist      how far they walked (m, final x position)
    eval      every --eval-every iterations: the MEAN action (no sampling), 5 fixed starts: score, length, distance
    Lv        critic loss before → after the update (scaled rewards)      EV  explained variance of V
    H         entropy of the Gaussian (falls as σ shrinks)                 σ   mean spread of the 6 torques
    KL        KL(π_old ‖ π_new) on the batch states                        clip  share of steps past the 1 ± 0.2 limit
    |a|>1     share of sampled torques outside [−1, 1] (clipped before they reach the robot)
  off-policy (one row = 2,048 env steps, with one update after each step):
    Lv        critic loss mean (Q − y)²                                    EV  explained variance of Q vs its targets y
    Q / real  the critic's Q(s, a) next to the discounted return that really followed (> real = overestimation)
    noise     std of the exploration noise                                 sat  share of actor torques at the tanh limit
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

from rl.logger import runs_dir  # noqa: E402
from rl.offpolicy import OffPolicyConfig, OffPolicySession  # noqa: E402
from rl.ppo_continuous import PPOContinuousConfig, PPOContinuousSession  # noqa: E402

ENVS = {"walker2d": ("Walker2d-v5", "Walker2d")}
ALGOS = {"ppo": "PPO with a Gaussian policy (2,048-step batches, 10 epochs, clip 0.2, GAE 0.95)",
         "ddpg": "DDPG: deterministic actor + Q(s, a) critic, replay buffer, soft target networks, Gaussian noise 0.1"}
OFF_POLICY = {"ddpg"}

_viewer = None


def show_window(session, label: str, env_id: str) -> None:
    """Play the current policy in a window in a separate process (training goes on)."""
    import multiprocessing as mp

    from rl.viewer import play_continuous

    global _viewer
    if _viewer is not None and _viewer.is_alive():
        print(f"     ↳ (the previous window is still playing; no window for {label})")
        return
    st = session.state_dict()
    st = {**st, "actor": {k: v.detach().clone() for k, v in st["actor"].items()},
          "critic": {k: v.detach().clone() for k, v in st["critic"].items()}}
    _viewer = mp.get_context("spawn").Process(target=play_continuous, daemon=True, args=(env_id, st, label))
    _viewer.start()
    print(f"     ↳ window opened for the policy at {label} (training goes on)")


def evaluate(session, env_id: str, episodes: int = 5, seed0: int = 10_000, max_steps: int | None = None) -> dict:
    """The mean action (no sampling) from fixed starts. Score = real reward."""
    env = gym.make(env_id, **({"max_episode_steps": max_steps} if max_steps else {}))
    scores, lens, fell, dist = [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        score, n = 0.0, 0
        while True:
            obs, r, term, trunc, info = env.step(session.act(obs))
            score += float(r)
            n += 1
            if term or trunc:
                break
        scores.append(score)
        lens.append(n)
        fell.append(bool(term))
        dist.append(float(info["x_position"]))
    env.close()
    return {"scores": scores, "lens": lens, "fell": fell, "dist": dist}


# Walker2d's 7 capsules, drawn by the web page: behind (left leg), torso, front (right leg)
GEOMS = ("thigh_left_geom", "leg_left_geom", "foot_left_geom", "torso_geom", "thigh_geom", "leg_geom", "foot_geom")
FRAME_EVERY = 2  # keep every 2nd step (62.5 frames per simulated second)


def body_segments(env) -> list[float]:
    """Both end points (x, z) of every capsule, as MuJoCo places them: centre ± half-length along the capsule axis."""
    import mujoco

    m, d = env.unwrapped.model, env.unwrapped.data
    mujoco.mj_kinematics(m, d)
    out = []
    for name in GEOMS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        c, axis = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
        out += [c[0] - axis[0], c[2] - axis[2], c[0] + axis[0], c[2] + axis[2]]
    return [round(float(v), 3) for v in out]


def record_episode(session, env_id: str, folder: Path, it: int, gif: bool = True) -> dict:
    """One episode of the mean action from a fixed start: a GIF, and the body's capsules per frame as JSON
    (the web page draws and replays the walker from them, with the 6 torques)."""
    import mujoco
    from PIL import Image

    env = gym.make(env_id, **({"render_mode": "rgb_array", "width": 320, "height": 240} if gif else {}))
    obs, _ = env.reset(seed=10_000)
    m = env.unwrapped.model
    radii = [round(float(m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)][0]), 3) for n in GEOMS]
    images, frames, actions, score, n = [], [], [], 0.0, 0
    while True:
        a = session.act(obs)
        if n % FRAME_EVERY == 0:
            frames.append(body_segments(env))
            actions.append([round(float(v), 3) for v in a])
        if gif and n % 4 == 0:  # every 4th frame keeps the GIF small (125 fps → ~31 fps)
            images.append(Image.fromarray(env.render()))
        obs, r, term, trunc, info = env.step(a)
        score += float(r)
        n += 1
        if term or trunc:
            break
    frames.append(body_segments(env))  # the final pose
    actions.append(actions[-1])
    env.close()
    name = f"iter{it:04d}"
    info_out = {"name": name, "iter": it, "kind": "greedy", "steps": n, "terminated": bool(term), "score": score,
                "distance": float(info["x_position"])}
    (folder / f"{name}.json").write_text(json.dumps({
        **info_out, "dt": float(env.unwrapped.dt), "every": FRAME_EVERY, "env_steps": session.env_steps,
        "geoms": [{"name": g, "radius": r} for g, r in zip(GEOMS, radii)], "frames": frames, "actions": actions}))
    if gif:  # ~4 MB for a 1,000-step episode, so only every --gif-every iterations
        images[0].save(folder / f"{name}.gif", save_all=True, append_images=images[1:], duration=32, loop=0)
    return info_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=list(ENVS), default="walker2d")
    ap.add_argument("--algo", choices=list(ALGOS), default="ppo")
    ap.add_argument("--budget", type=int, default=1_000_000, help="environment steps (default 1,000,000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=None, help="units per hidden layer, 2 layers (default: ppo 64, ddpg 256)")
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (off-policy: actor and critic)")
    ap.add_argument("--threads", type=int, default=1, help="off-policy: CPU threads for the network updates")
    ap.add_argument("--learning-starts", type=int, default=25_000, help="off-policy: random steps before learning")
    ap.add_argument("--expl-noise", type=float, default=0.1, help="ddpg: std of the exploration noise")
    ap.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus (default 0, as in standard PPO for MuJoCo)")
    ap.add_argument("--no-norm-obs", action="store_true", help="turn off observation normalisation")
    ap.add_argument("--no-norm-reward", action="store_true", help="turn off reward scaling")
    ap.add_argument("--eval-every", type=int, default=10, help="evaluate the mean action every N iterations")
    ap.add_argument("--record-every", type=int, default=10, help="record the mean action (GIF + JSON for the web page) every N iterations")
    ap.add_argument("--gif-every", type=int, default=50, help="also save a GIF every N iterations (and the first and last)")
    ap.add_argument("--render-every-steps", type=int, default=100_000,
                    help="show the policy in a window each time training passes another N env steps (0 = never)")
    ap.add_argument("--quality-starts", type=int, default=20, help="start states in the final test")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--open", action=argparse.BooleanOptionalAction, default=True, help="open the plot at the end")
    args = ap.parse_args()

    env_id, title = ENVS[args.env]
    off = args.algo in OFF_POLICY
    if args.hidden is None:
        args.hidden = 256 if off else 64
    if off:
        cfg = OffPolicyConfig(algo=args.algo, env_id=env_id, seed=args.seed, hidden=args.hidden, actor_lr=args.lr,
                              critic_lr=args.lr, learning_starts=args.learning_starts, expl_noise=args.expl_noise,
                              total_steps=args.budget, threads=args.threads)
        session = OffPolicySession(cfg)
    else:
        cfg = PPOContinuousConfig(env_id=env_id, seed=args.seed, hidden=args.hidden, lr=args.lr, ent_coef=args.ent_coef,
                                  norm_obs=not args.no_norm_obs, norm_reward=not args.no_norm_reward, total_steps=args.budget)
        session = PPOContinuousSession(cfg)
    out = runs_dir() / args.env / args.algo
    shutil.rmtree(out, ignore_errors=True)
    (out / "episodes").mkdir(parents=True)
    (out / "settings.json").write_text(json.dumps(
        {"algo": args.algo, "algo_name": ALGOS[args.algo], "env": args.env, "env_id": env_id, "budget": args.budget,
         "seed": args.seed, "continuous": True, "config": cfg.model_dump()}, indent=2))
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    def status(state: str, it: int, steps: int = 0) -> None:
        (out / "status.json").write_text(json.dumps(
            {"state": state, "iter": it, "env_steps": steps, "budget": args.budget, "started": started}))

    status("running", 0)
    un = evaluate(session, env_id)
    print(f"{title} · {ALGOS[args.algo]}")
    print(f"untrained policy (mean action): score {np.mean(un['scores']):.0f}, {np.mean(un['lens']):.0f} steps before "
          f"falling, {np.mean(un['dist']):+.2f} m   (budget {args.budget:,} env steps, hidden 2×{args.hidden})\n")
    cols = ["iter", "env_steps", "score", "len", "fell", "dist", "eval_score", "eval_len", "eval_fell", "eval_dist",
            "Lv", "Lv_after", "Lpi", "H", "std", "EV", "absA", "KL", "clip_frac", "grad_norm", "lr", "act_abs",
            "act_clipped", "reward_scale", "seconds", "q_mean", "q_pred", "g_real", "noise", "sat", "buffer", "updates",
            "gn_actor", "gn_critic"]
    f = open(out / "log.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    if off:
        print(f"{'iter':>4} {'steps':>9} {'score':>7} {'len':>5} {'fell':>5} {'dist':>6} {'Lv':>9} {'EV':>5} "
              f"{'Q':>7} {'real':>7} {'sat':>5} {'|a|>1':>5} {'eval':>6}")
    else:
        print(f"{'iter':>4} {'steps':>9} {'score':>7} {'len':>5} {'fell':>5} {'dist':>6} {'Lv (before → after)':>20} {'EV':>5} "
              f"{'H':>6} {'σ':>5} {'KL':>7} {'clip':>5} {'|a|>1':>5} {'eval':>6}")
    rows: list[dict] = []
    t0 = time.time()
    it, completed = 0, False
    try:
        while session.env_steps < args.budget:
            it += 1
            prev = session.env_steps
            s = session.iterate()
            fin = s["finished"]
            last_row = rows[-1] if rows else {}
            row = {"iter": it, "env_steps": s["env_steps"],
                   "score": float(np.mean([e["ret"] for e in fin])) if fin else last_row.get("score"),
                   "len": float(np.mean([e["len"] for e in fin])) if fin else last_row.get("len"),
                   "fell": float(np.mean([e["fell"] for e in fin])) if fin else last_row.get("fell"),
                   "dist": float(np.mean([e["x"] for e in fin])) if fin else last_row.get("dist"),
                   "eval_score": None, "eval_len": None, "eval_fell": None, "eval_dist": None,
                   "Lv": s["critic_loss"], "Lv_after": s.get("critic_loss_after"), "Lpi": s["actor_loss"], "H": s.get("entropy"),
                   # exploration spread: PPO's σ (mean of the 6 joints); DDPG's noise std once learning has started
                   "std": float(np.mean(s["std"])) if "std" in s else s.get("noise"),
                   "EV": s["ev"], "absA": s.get("mean_abs_adv"), "KL": s.get("kl"),
                   "clip_frac": s.get("clip_frac"), "grad_norm": s.get("grad_norm"), "lr": s.get("lr"), "act_abs": s["action_abs_mean"],
                   "act_clipped": s["action_clipped"], "reward_scale": s.get("reward_scale"), "seconds": round(time.time() - t0, 1),
                   "q_mean": s.get("q_mean"), "q_pred": s.get("q_pred"), "g_real": s.get("g_real"), "noise": s.get("noise"),
                   "sat": s.get("actor_saturated"), "buffer": s.get("buffer"), "updates": s.get("updates"),
                   "gn_actor": s.get("gn_actor"), "gn_critic": s.get("gn_critic")}
            last = session.env_steps >= args.budget
            if it == 1 or it % args.eval_every == 0 or last:
                e = evaluate(session, env_id)
                row.update(eval_score=float(np.mean(e["scores"])), eval_len=float(np.mean(e["lens"])),
                           eval_fell=float(np.mean(e["fell"])), eval_dist=float(np.mean(e["dist"])))
            rows.append(row)
            writer.writerow(row)
            f.flush()
            fmt = lambda v, spec: format(v, spec) if v is not None else " " * len(format(0, spec))  # noqa: E731
            ev = row["EV"] if row["EV"] is not None else float("nan")
            head = (f"{it:4d} {row['env_steps']:9,d} {fmt(row['score'], '7.0f')} {fmt(row['len'], '5.0f')} {fmt(row['fell'], '5.0%')} "
                    f"{fmt(row['dist'], '+6.1f')} ")
            if off:
                warm = "" if s.get("learning") else "   (random actions: filling the buffer)"
                print(head + f"{fmt(row['Lv'], '9.3f')} {ev:5.2f} {fmt(row['q_pred'], '7.1f')} {fmt(row['g_real'], '7.1f')} "
                      f"{fmt(row['sat'], '5.0%')} {row['act_clipped']:5.0%} {fmt(row['eval_score'], '6.0f')}{warm}")
            else:
                print(head + f"{row['Lv']:9.3f} → {row['Lv_after']:7.3f} {ev:5.2f} {row['H']:6.2f} "
                      f"{row['std']:5.2f} {row['KL']:7.4f} {row['clip_frac']:5.0%} {row['act_clipped']:5.0%} "
                      f"{fmt(row['eval_score'], '6.0f')}")
            if row["eval_score"] is not None and it > 1:
                print(f"     ↳ eval (mean action, 5 starts): score {row['eval_score']:.0f}, {row['eval_len']:.0f} steps, "
                      f"walked {row['eval_dist']:+.1f} m, fell {row['eval_fell']:.0%}   [{row['seconds'] / 60:.1f} min]")
            if it == 1 or it % args.record_every == 0 or last:
                gif = it == 1 or it % args.gif_every == 0 or last
                rec = record_episode(session, env_id, out / "episodes", it, gif)
                print(f"     ↳ recorded the mean action: {rec['steps']} steps, {rec['distance']:+.1f} m"
                      + (f" (episodes/iter{it:04d}.gif)" if gif else ""))
            if args.render_every_steps and session.env_steps // args.render_every_steps > prev // args.render_every_steps:
                show_window(session, f"{session.env_steps:,} env steps", env_id)
            status("running", it, session.env_steps)
    except KeyboardInterrupt:
        print("\nstopped by Ctrl-C")
        status("stopped", it, session.env_steps)
    else:
        completed = True
    finally:
        f.close()
    if not rows:
        return
    torch.save(session.state_dict(), out / "final_weights.pt")
    if completed:
        status("testing", it, session.env_steps)
        print(f"\nFinal test: the mean action from {args.quality_starts} start states, 1,000 steps each …")
        q = evaluate(session, env_id, args.quality_starts, seed0=20_000)
        stands = sum(1 for L, d in zip(q["lens"], q["dist"]) if L >= 1000 and d < 1.0)
        card = {"scores": q["scores"], "lens": q["lens"], "fell": q["fell"], "dist": q["dist"],
                "median_score": float(np.median(q["scores"])), "falls": int(sum(q["fell"])), "stands_still": stands,
                "median_dist": float(np.median(q["dist"])),
                "first_eval_steps": {str(t): next((r["env_steps"] for r in rows if (r["eval_score"] or -1e9) >= t), None)
                                     for t in (1000, 2000, 3000, 4000)}}
        (out / "scorecard.json").write_text(json.dumps(card))
        print(f"  median score {card['median_score']:.0f} · fell {card['falls']}/{args.quality_starts} · median distance "
              f"{card['median_dist']:.1f} m · stood still (no fall, < 1 m) {stands}/{args.quality_starts}")
        print("  eval score first reached: " + ", ".join(
            f"{t}: {'never' if v is None else f'{v:,} steps'}" for t, v in card["first_eval_steps"].items()))
        status("finished", it, session.env_steps)
    print(f"\neverything saved in {out}   (watch it: uv run python scripts/play_walker2d.py --policy trained --algo {args.algo})")
    if args.plot:
        _plot(rows, out / "plot.png", f"{title} · {ALGOS[args.algo]}")
        print(f"plot saved: {out / 'plot.png'}")
        if args.open and shutil.which("explorer.exe"):
            win = subprocess.run(["wslpath", "-w", str(out / "plot.png")], capture_output=True, text=True).stdout.strip()
            subprocess.run(["explorer.exe", win], check=False)
    if _viewer is not None and _viewer.is_alive():
        print("waiting for the last window to finish …")
        _viewer.join(60)


def _plot(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = lambda k, rs=rows, f=1.0: [np.nan if r.get(k) is None else r[k] * f for r in rs]  # noqa: E731
    x = [r["env_steps"] / 1e6 for r in rows]
    ev = [r for r in rows if r["eval_score"] is not None]
    xe = [r["env_steps"] / 1e6 for r in ev]
    off = any(r.get("q_pred") is not None for r in rows)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle(title)
    ax[0, 0].plot(x, v("score"), label="training episodes: mean score (with exploration)")
    ax[0, 0].plot(xe, v("eval_score", ev), "o-", ms=3, label="eval: deterministic action, 5 starts")
    ax[0, 0].set_title("1. Is it learning?  score (real reward)")
    ax[0, 1].plot(x, v("dist"), label="distance walked (m)")
    ax[0, 1].plot(x, v("len", f=0.01), label="episode length / 100 (10 = never fell)")
    ax[0, 1].set_title("2. Walking or just standing?")
    ax[1, 0].plot(x, v("std"), label="exploration spread (PPO σ / DDPG noise std)")
    ax[1, 0].plot(x, v("act_clipped"), label="share of torques clipped to ±1")
    ax[1, 0].set_title("3. Exploration")
    if off:
        ax[1, 1].plot(x, v("q_pred"), label="critic: Q(s, a)")
        ax[1, 1].plot(x, v("g_real"), label="the discounted return that really followed")
        ax[1, 1].set_title("4. Does the critic overestimate?")
    else:
        ax[1, 1].plot(x, v("EV"), label="critic explained variance")
        ax[1, 1].plot(x, v("clip_frac"), label="share past the clip")
        ax[1, 1].plot(x, [min(1.0, 10 * r["KL"]) if r.get("KL") is not None else np.nan for r in rows], label="10 × KL per update")
        ax[1, 1].set_title("4. Critic and update size")
        ax[1, 1].set_ylim(-0.2, 1.05)
    for a in ax.flat:
        a.set_xlabel("million env steps")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
