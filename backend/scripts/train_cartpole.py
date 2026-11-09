"""Train CartPole with batch actor-critic from the terminal. No web server needed.

    uv run python scripts/train_cartpole.py                       # 150 iterations, one log line each
    uv run python scripts/train_cartpole.py --plot                # + live plot window, saved as PNG at the end
    uv run python scripts/train_cartpole.py --render-every 25     # + watch the current policy play every 25 iterations
    uv run python scripts/train_cartpole.py --help                # all options (learning rates, batch size, ...)

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

Reference policy ("a good policy we know"): push right if  pole angle + 0.5 × angular velocity > 0,
else left. It scores 500 (the maximum) on every start state, so it is optimal for this task.
"""

from __future__ import annotations

import argparse
import csv
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
    ap.add_argument("--render-every", type=int, default=0, help="show the current policy playing every N iterations (0 = never)")
    ap.add_argument("--plot", action="store_true", help="live plot window (also saved as PNG)")
    args = ap.parse_args()

    cfg = LearnConfig(env_id="CartPole-v1", seed=args.seed, batch_episodes=args.batch, hidden=args.hidden,
                      actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma, optimizer=args.optimizer)
    session = ExplainSession(cfg)
    out_dir = runs_dir() / "cli"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"cartpole-{stamp}.csv"

    print(f"reference policy (angle + 0.5·ang_vel > 0 → right): eval return {evaluate(reference_action):.0f} / 500")
    print(f"untrained policy: eval return {evaluate(lambda s: greedy_action(session, s)):.0f} / 500\n")
    cols = ["iter", "ret", "Lv", "Lv_after", "Lpi", "H", "EV", "absA", "KL", "agree", "eval"]
    print(f"{'iter':>4} {'ret':>6} {'Lv (before → after)':>22} {'Lpi':>8} {'H':>5} {'EV':>6} {'|A|':>6} {'KL':>8} {'agree':>6} {'eval':>5}")

    rows: list[dict] = []
    fig = axes = None
    live = False
    if args.plot:
        import matplotlib
        import matplotlib.pyplot as plt

        live = matplotlib.get_backend().lower() != "agg"
        if not live:
            print("(no plot window available: install it with  sudo apt install python3-tk ; the plot is still saved as PNG)\n")
        plt.ion()
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
        fig.suptitle("CartPole, batch actor-critic")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for it in range(1, args.iters + 1):
            tr = session.iterate()  # collect → critic step → actor step (fully recorded)
            s = tr["summary"]
            p_old = np.array(tr["actor"]["probs_before"])  # π_k on the batch states
            p_new = np.array(tr["actor"]["probs_after"])  # π_k+1 on the same states
            kl = float(np.mean(np.sum(p_old * (np.log(p_old + 1e-12) - np.log(p_new + 1e-12)), axis=1)))
            states = [st for ep in tr["episodes"] for st in ep["states"]]
            agree = float(np.mean([int(np.argmax(p)) == reference_action(st) for p, st in zip(p_old, states)]))
            ev = s["ev"] if s["ev"] is not None else float("nan")
            row = {"iter": it, "ret": s["mean_return"], "Lv": s["critic_loss"], "Lv_after": s["critic_loss_after"],
                   "Lpi": s["actor_loss"], "H": s["entropy"], "EV": ev, "absA": s["mean_abs_adv"], "KL": kl,
                   "agree": agree, "eval": None}
            if it % args.eval_every == 0 or it == 1 or it == args.iters:
                row["eval"] = evaluate(lambda st: greedy_action(session, st))
            rows.append(row)
            writer.writerow(row)
            f.flush()
            ev_txt = f"{ev:6.2f}" if ev == ev else "   n/a"
            eval_txt = f"{row['eval']:5.0f}" if row["eval"] is not None else ""
            print(f"{it:4d} {s['mean_return']:6.1f} {s['critic_loss']:10.2f} → {s['critic_loss_after']:9.2f} "
                  f"{s['actor_loss']:8.3f} {s['entropy']:5.2f} {ev_txt} {s['mean_abs_adv']:6.2f} {kl:8.5f} {agree:6.0%} {eval_txt}")

            if args.render_every and it % args.render_every == 0:
                print(f"     ↳ showing the current policy (most likely action) in a window …")
                try:
                    r = evaluate(lambda st: greedy_action(session, st), episodes=1, render=True)
                    print(f"     ↳ it balanced for {r:.0f} steps")
                except Exception as err:  # no display available
                    print(f"     ↳ cannot open a window ({err}); use --render-every 0")
                    args.render_every = 0
            if fig is not None and (live or it == args.iters):
                _draw(fig, axes, rows, live)

    print(f"\nlog saved: {csv_path}")
    if fig is not None:
        png = csv_path.with_suffix(".png")
        fig.savefig(png, dpi=110)
        print(f"plot saved: {png}   (open it from Windows:  explorer.exe \"$(wslpath -w {png})\")")
        if live:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.show()
    _diagnose(rows)


def _draw(fig, axes, rows: list[dict], live: bool) -> None:
    import matplotlib.pyplot as plt

    it = [r["iter"] for r in rows]
    ev_it = [r["iter"] for r in rows if r["eval"] is not None]
    panels = [
        (axes[0, 0], "return", [("batch return", [r["ret"] for r in rows], it), ("eval (greedy)", [r["eval"] for r in rows if r["eval"] is not None], ev_it)], False),
        (axes[0, 1], "critic loss ½(G−V)²", [("before step", [r["Lv"] for r in rows], it), ("after step", [r["Lv_after"] for r in rows], it)], True),
        (axes[0, 2], "actor loss −A·log π", [("actor loss", [r["Lpi"] for r in rows], it)], False),
        (axes[1, 0], "critic explained variance", [("EV", [r["EV"] for r in rows], it)], False),
        (axes[1, 1], "policy: entropy and agreement with reference", [("entropy", [r["H"] for r in rows], it), ("agree with reference", [r["agree"] for r in rows], it)], False),
        (axes[1, 2], "policy change per update KL(π_old‖π_new)", [("KL", [r["KL"] for r in rows], it)], True),
    ]
    for ax, title, series, log in panels:
        ax.clear()
        for label, ys, xs in series:
            ax.plot(xs, ys, label=label, lw=1.5)
        ax.set_title(title, fontsize=10)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=8)
    axes[1, 0].set_ylim(-1, 1)
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
