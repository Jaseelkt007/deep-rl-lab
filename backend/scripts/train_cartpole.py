"""Train CartPole with batch actor-critic from the terminal. No web server needed.

    uv run python scripts/train_cartpole.py        # the one command: logs + game window every 25 iterations
                                                   # + one plot opened at the end
    uv run python scripts/train_cartpole.py --help # options (learning rates, batch size, --render-every 0, ...)

Everything of the LATEST run is saved in runs/cli/latest/ (cleared when a new run starts):
    log.csv, settings.json, status.json, plot.png,
    episodes/iterNNNN.json + .gif   the current policy playing one episode at iteration 1, every
                                    --record-every iterations and at the end (JSON = for the web page)
    episodes/good.json + .gif       the hand-written good policy on the same start, for comparison
The web page http://localhost:3000/terminal shows it (live while training).

One iteration (same code as the Learn page, rl/explain.py):
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
from rl.logger import runs_dir  # noqa: E402


def reference_action(state) -> int:
    """A hand-written controller: push towards the side the pole is falling to."""
    return 1 if state[2] + 0.5 * state[3] > 0 else 0


@torch.no_grad()
def greedy_action(session: ExplainSession, state) -> int:
    x = torch.as_tensor(session.encoder(np.asarray(state), 0))
    return int(session.actor(x).argmax())


def evaluate(policy, episodes: int = 10, render: bool = False) -> float:
    """Average return of `policy(state) -> action` on fixed start states."""
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    total = []
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        ret = 0.0
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ret += r
            if term or trunc:
                break
        total.append(ret)
    env.close()
    return float(np.mean(total))


@torch.no_grad()
def record_episode(session: ExplainSession | None, it: int, folder: Path, name: str | None = None) -> dict:
    """Play one episode from a fixed start state and save it: trajectory as JSON (the web page
    replays it) and frames as a GIF. session = None plays the hand-written good policy."""
    from PIL import Image

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=10_000)  # the same start as the first eval episode
    states, actions, probs, frames = [], [], [], []
    while True:
        if session is None:  # the good policy: certain of its action
            a = reference_action(obs)
            p = [1.0 - a, float(a)]
        else:  # the agent's policy, always taking its most likely action
            pt = F.softmax(session.actor(torch.as_tensor(session.encoder(np.asarray(obs), 0))), -1)
            a, p = int(pt.argmax()), pt.tolist()
        states.append(np.asarray(obs, dtype=float).tolist())
        actions.append(a)
        probs.append(p)
        if len(states) % 2 == 1:  # every 2nd frame keeps the GIF small
            frames.append(Image.fromarray(env.render()).resize((300, 200)))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    env.close()
    name = name or f"iter{it:04d}"
    info = {"name": name, "iter": it, "steps": len(actions), "terminated": bool(term)}
    (folder / f"{name}.json").write_text(json.dumps({**info, "states": states, "actions": actions, "probs": probs,
                                                    "final_state": np.asarray(obs, dtype=float).tolist()}))
    frames[0].save(folder / f"{name}.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
    return info


@torch.no_grad()
def true_values(session: ExplainSession, states: list, rollouts: int = 10, horizon: int = 600, seed: int = 0):
    """Monte Carlo ground truth: average discounted return of the CURRENT policy from each state.

    No time limit (the critic bootstraps at the 500-step cut, so it estimates the never-ending
    task); 600 steps cover it (0.99^600 ≈ 0.002). Returns (mean, std) per state."""
    env = gym.make("CartPole-v1").unwrapped  # no TimeLimit wrapper
    gen = torch.Generator().manual_seed(seed)
    g = session.cfg.gamma
    means, stds = [], []
    for k, st in enumerate(states):
        rets = []
        for r in range(rollouts):
            env.reset(seed=seed + 1000 * k + r)
            env.state = np.array(st, dtype=np.float64)  # start exactly in this state
            obs, ret, disc = np.array(st), 0.0, 1.0
            for _ in range(horizon):
                p = F.softmax(session.actor(torch.as_tensor(session.encoder(obs, 0))), -1)
                obs, rew, term, _, _ = env.step(int(torch.multinomial(p, 1, generator=gen)))
                ret += disc * rew
                disc *= g
                if term:
                    break
            rets.append(ret)
        means.append(float(np.mean(rets)))
        stds.append(float(np.std(rets)))
    env.close()
    return np.array(means), np.array(stds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--batch", type=int, default=5, help="episodes per batch")
    ap.add_argument("--hidden", type=int, default=8, help="hidden units (one hidden layer)")
    ap.add_argument("--actor-lr", type=float, default=0.01)
    ap.add_argument("--critic-lr", type=float, default=0.03)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--render-every", type=int, default=25, help="show the current policy playing every N iterations (0 = never)")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="plot to runs/cli/latest/plot.png and open it at the end")
    ap.add_argument("--truth-every", type=int, default=10, help="ground-truth critic check every N iterations (0 = never)")
    ap.add_argument("--record-every", type=int, default=10, help="save one episode of the current policy every N iterations")
    ap.add_argument("--open", action=argparse.BooleanOptionalAction, default=True, help="open the plot at the end")
    args = ap.parse_args()

    cfg = LearnConfig(env_id="CartPole-v1", seed=args.seed, batch_episodes=args.batch, hidden=args.hidden,
                      actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma, optimizer=args.optimizer)
    session = ExplainSession(cfg)
    out_dir = runs_dir() / "cli" / "latest"  # only the latest run is kept
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "episodes").mkdir(parents=True)
    csv_path = out_dir / "log.csv"
    (out_dir / "settings.json").write_text(json.dumps(vars(args), indent=2))
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    def status(state: str, it: int) -> None:
        (out_dir / "status.json").write_text(json.dumps({"state": state, "iter": it, "iters": args.iters, "started": started}))

    status("running", 0)
    record_episode(None, 0, out_dir / "episodes", name="good")  # the hand-written policy, for comparison

    print(f"reference policy (angle + 0.5·ang_vel > 0 → right): eval return {evaluate(reference_action):.0f} / 500")
    print(f"untrained policy: eval return {evaluate(lambda s: greedy_action(session, s)):.0f} / 500\n")
    cols = ["iter", "ret", "Lv", "Lv_after", "rel", "corr", "Lpi", "H", "EV", "absA", "KL", "agree", "p_good",
            "ce_good", "dir", "eval", "true_err", "noise", "V_mean", "Vtrue_mean"]
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
        fig.suptitle("CartPole, batch actor-critic")

    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    it = 0
    try:
        for it in range(1, args.iters + 1):
            tr = session.iterate()  # collect → critic step → actor step (fully recorded)
            s = tr["summary"]
            p_old = np.array(tr["actor"]["probs_before"])  # π_k on the batch states
            p_new = np.array(tr["actor"]["probs_after"])  # π_k+1 on the same states
            kl = float(np.mean(np.sum(p_old * (np.log(p_old + 1e-12) - np.log(p_new + 1e-12)), axis=1)))
            states = [st for ep in tr["episodes"] for st in ep["states"]]
            agree = float(np.mean([int(np.argmax(p)) == reference_action(st) for p, st in zip(p_old, states)]))
            good = np.array([p[reference_action(st)] for p, st in zip(p_old, states)])  # π(good action | s)
            p_good, ce_good = float(good.mean()), float(np.mean(-np.log(good + 1e-12)))
            adv = np.array(tr["advantages"])
            dlogp = np.array(tr["actor"]["logp_after"]) - np.array(tr["actor"]["logp_before"])
            direction = float(np.corrcoef(adv, dlogp)[0, 1]) if adv.std() > 1e-9 and dlogp.std() > 1e-12 else float("nan")
            ev = s["ev"] if s["ev"] is not None else float("nan")
            G, V = np.array(tr["returns"]), np.array(tr["values"])  # V: critic BEFORE its step, on fresh data
            rel = float(np.sqrt(np.mean((G - V) ** 2)) / (np.sqrt(np.mean(G**2)) + 1e-12))
            corr = float(np.corrcoef(G, V)[0, 1]) if G.std() > 1e-9 and V.std() > 1e-9 else float("nan")
            row = {"iter": it, "ret": s["mean_return"], "Lv": s["critic_loss"], "Lv_after": s["critic_loss_after"],
                   "rel": rel, "corr": corr, "Lpi": s["actor_loss"], "H": s["entropy"], "EV": ev,
                   "absA": s["mean_abs_adv"], "KL": kl, "agree": agree, "p_good": p_good, "ce_good": ce_good,
                   "dir": direction, "eval": None,
                   "true_err": None, "noise": None, "V_mean": None, "Vtrue_mean": None}
            if it % args.eval_every == 0 or it == 1 or it == args.iters:
                row["eval"] = evaluate(lambda st: greedy_action(session, st))
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
            ev_txt = f"{ev:6.2f}" if ev == ev else "   n/a"
            eval_txt = f"{row['eval']:5.0f}" if row["eval"] is not None else ""
            corr_txt = f"{corr:5.2f}" if corr == corr else "  n/a"
            print(f"{it:4d} {s['mean_return']:6.1f} {s['critic_loss']:10.2f} → {s['critic_loss_after']:9.2f} {rel:5.0%} {corr_txt} "
                  f"{s['actor_loss']:8.3f} {s['entropy']:5.2f} {ev_txt} {s['mean_abs_adv']:6.2f} {kl:8.5f} {agree:6.0%} "
                  f"{p_good:6.2f} {ce_good:7.3f} {direction:5.2f} {eval_txt}")
            if truth_line:
                print(truth_line)

            if it == 1 or it % args.record_every == 0 or it == args.iters:
                ep = record_episode(session, it, out_dir / "episodes")
                print(f"     ↳ recorded the current policy playing: {ep['steps']} steps (episodes/{ep['name']}.gif)")
                if args.render_every and it % args.render_every == 0:
                    try:  # the same episode again, live in a window
                        evaluate(lambda st: greedy_action(session, st), episodes=1, render=True)
                    except Exception as err:  # no display available
                        print(f"     ↳ cannot open a window ({err}); use --render-every 0")
                        args.render_every = 0
            status("running", it)
            if fig is not None and (live or it == args.iters):
                _draw(fig, axes, rows, live)
    except KeyboardInterrupt:
        print("\nstopped by Ctrl-C")
        status("stopped", it)
    else:
        status("finished", it)
    finally:
        f.close()

    print(f"\neverything saved in: {out_dir}   (web page: http://localhost:3000/terminal)")
    if not rows:
        return
    _diagnose(rows)
    if fig is not None:
        if not live or it < args.iters:
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
        (axes[0, 0], "1. Is the agent learning?  episode length (500 = perfect)", [
            ("batch: mean episode length", [r["ret"] for r in rows], it),
            ("eval: most likely action, 10 fixed starts", [r["eval"] for r in rows if r["eval"] is not None], ev_it)], False),
        (axes[0, 1], "2. Is the actor learning?  (does not grow with episode length)", [
            ("P(good action): 0.5 coin flip → 1 perfect", [r["p_good"] for r in rows], it),
            ("most likely action = good action", [r["agree"] for r in rows], it),
            ("entropy (0.69 = coin flip, 0 = always same)", [r["H"] for r in rows], it)], False),
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


def _diagnose(rows: list[dict]) -> None:
    """A few plain-language checks on the last 20 iterations."""
    last = rows[-20:]
    first = rows[:20]
    mean = lambda k, rs: float(np.nanmean([r[k] for r in rs]))  # noqa: E731
    print("\nIs it learning? (last 20 iterations vs first 20)")
    print(f"  batch return {mean('ret', first):.0f} → {mean('ret', last):.0f}"
          f"   ({'improving' if mean('ret', last) > 1.5 * mean('ret', first) else 'not clearly improving'})")
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
