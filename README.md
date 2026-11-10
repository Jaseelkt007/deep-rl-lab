# Deep RL playground

A small playground for understanding how actor-critic learns, by watching it. It currently focuses on **CartPole-v1**: **batch actor-critic** (on-policy, Monte Carlo returns), then **PPO**, built from it one change at a time. MountainCar-v0 is next. You train in the terminal and watch the run in the web app.

![Run page](docs/img/cartpole-page.png)

## Quick start

You need Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/), and Node ≥ 20 with npm. Everything runs locally on the CPU.

**Train (terminal):**
```bash
cd backend && uv sync
uv run python scripts/train_cartpole.py --algo all --render-every 0   # all 5 algorithms, one after another
uv run python scripts/train_cartpole.py --algo ppo                    # or one: batch_ac | epochs | ratio | clip | gae | ppo
```
- Every algorithm trains for the same 200,000 environment steps (`--budget`).
- One log line per iteration in the terminal.
- A window showing the current policy every 25 iterations (`--render-every 0`: none).
- One plot at the end; then the final policy is tested for 10,000 steps (the scorecard).
- Everything is saved in `runs/cli/<algo>/`, replaced by the next run of that algorithm.

**See it (web page)**, in two more terminals:
```bash
cd backend && uv run uvicorn server:app --port 8000
cd frontend && npm install && npm run dev
```
Then open **http://localhost:3000** (on WSL, in the Windows browser). The page refreshes every 3 s, also while a run is training.

Useful options: `--budget 100000`, `--seed 1`, `--batch 10`, `--actor-lr 0.005`, `--critic-lr 0.05`, `--hidden 16`. Options you leave out use each algorithm's own values. `--help` lists them all.

## The web app

- **Run** (`/`): the latest terminal run of each algorithm.
  - The steps `Batch AC → + epochs → + ratio → + clip → + GAE → PPO`, with the status of each, and a button that copies the train command.
  - **Compare the algorithms**, beyond the 500 cap, all against environment steps (iterations differ between algorithms):
    - a table: gradient steps per batch, **speed** (steps until the policy first scores 500), **stability** (does it stay there), **quality** (the final policy tested for up to 10,000 steps from 20 starts: survival, centering, calmness, action switching), next to the hand-written good policy;
    - episode length vs environment steps, one line per algorithm;
    - update size: how far each batch moves the policy (KL);
    - survival curves of the final policies;
    - inside one batch, epoch by epoch: KL from π_old, the ratio r of the steps pushed furthest, the share of steps past the clip limit.
  - Side by side: two policies (or one and the good policy) playing the same start, at the same iteration.
  - The selected algorithm (click a step of the ladder or a table row):
    - recorded episodes: the policy every 10 iterations, the final one, and the good policy, replayed with the state values;
    - its update rule (batch, advantage, critic and actor loss, gradient steps) and the environment;
    - inside one iteration (pick any): r, KL and the share past the clip limit after each of its epochs;
    - six charts: episode length; P(good action), agreement, entropy; KL and the share of steps moved > 20 %; the raw critic loss; the critic's scale-free error against the true value; corr(V, target) and explained variance;
    - the log.
- **Learn** (`/learn`): one update, step by step: collect, critic, returns, advantage, the critic step and the actor step weight by weight.
- **Compare** (`/compare`): overlay saved runs of the experiments.

## The algorithm (batch on-policy actor-critic)

Two networks: actor π(a|s) (4 → 8 tanh → 2 → softmax) and critic V(s) (4 → 8 tanh → 1). Each iteration:

```
1. the frozen actor plays N complete episodes (default 5)  → the batch
2. returns, backwards per episode:  G_t = r_t + γ·G_(t+1)
     ending with 0 if the pole fell (terminated), V(s_T) at the 500-step limit (truncated)
3. advantage:  A_t = G_t − V(s_t)
4. one critic step on   L_V = mean ½ (G_t − V(s_t))²
5. one actor step on    L_π = −mean A_t · log π(a_t|s_t)
6. throw the batch away (on-policy) and repeat
```

The code is `backend/rl/explain.py`. The training script is `backend/scripts/train_cartpole.py`.

## From batch actor-critic to PPO

`backend/rl/ppo.py` adds PPO's pieces to the algorithm above **one at a time**, so each one's effect is visible:

| `--algo` | adds | gradient steps per batch |
|---|---|---|
| `batch_ac` | the start: 5 episodes, 1 gradient step, throw away | 1 |
| `epochs` | use every batch 10 times (10 epochs) with the same loss −A·log π, as if the current policy had played the batch | 10 |
| `ratio` | importance ratio r = π(a\|s)/π_old(a\|s), loss −r·A: corrects for the batch coming from π_old (r = 1 in epoch 1), no limit | 10 |
| `clip` | the brake: loss −min(r·A, clip(r, 0.8, 1.2)·A); a step past the limit stops pushing | 10 |
| `gae` | advantage from the critic's one-step errors: δ = r + γV(s′) − V(s), A = Σ (γλ)ᵏ δ, λ = 0.95 | 10 |
| `ppo` | 2,048-step batches (cut episodes continue next batch), shuffled minibatches of 64, advantages normalised per minibatch, learning rates 1e-3 / 3e-3 | 10 × 32 = 320 |

Still on-policy: each batch is played by the current policy, used for its epochs and thrown away. No action is taken during the update; the ratio only re-scores the stored (state, action) pairs under the new policy. The advantages and critic targets are computed once per batch, with the critic before the update.

After every epoch the script measures how far the policy has moved from π_old on the batch (`per_epoch.csv`): KL, the ratio r of the steps pushed furthest, and the share of steps past the clip limit. The Run page shows this averaged over the learning iterations (to compare the steps) and for any single iteration.

Things worth knowing, all visible on the page:
- **The critic loss grows** as episodes get longer, because the returns it predicts get bigger. Judge the critic by its scale-free error instead.
- **The actor loss ≈ |A| × entropy.** Its value is not a quality measure. Judge the actor by P(good action) and the episode length.
- **The reward is +1 per step and nothing else.** Every policy that survives 500 steps scores the same: staying centered is not rewarded. Once almost every episode reaches 500 there is little signal left, and the policy can drift while still scoring 500.

## Other scripts

```bash
cd backend
uv run python scripts/play_cartpole.py          # watch the hand-written good policy in a window
uv run pytest                                   # tests (TD targets, n-step windows, exact known values, …)
uv run python scripts/check_cartpole.py         # known-answer learning check (one-step actor-critic)
uv run python scripts/experiments.py analyse 03-cartpole-no-signal   # pre-registered experiments
```

## Experiments

Hypotheses, predictions and decision rules are committed before the runs; results are added afterwards. These used the one-step (per-step update) actor-critic in `backend/rl/actor_critic.py`.

- [01 · Why does CartPole collapse?](docs/experiments/01-cartpole-collapse.md)
- [02 · MountainCar: how the formulation changes what can be learned](docs/experiments/02-mountaincar-formulation.md)
- [03 · Is the CartPole collapse Adam amplifying a no-signal δ?](docs/experiments/03-cartpole-no-signal-at-success.md)
- [03b · confirmatory (drafted, not run)](docs/experiments/03b-confirm-actor-eps.md)

## Project structure

```
backend/
  rl/explain.py         batch actor-critic, every number of an iteration recorded
  rl/actor_critic.py    one-step actor-critic (used by the experiments)
  rl/trainer.py, envs.py, networks.py, config.py, diagnostics.py, logger.py
  rl/ppo.py             the ladder from batch actor-critic to PPO (epochs, clip, GAE, minibatches)
  server.py             FastAPI: terminal runs (/api/cli/algos, /api/cli/run/{algo}), Learn page API, older run API
  scripts/              train_cartpole.py, play_cartpole.py, check_cartpole.py, experiments.py, run_batch.py
  tests/                pytest
frontend/               Next.js: / (Run), /learn (one update, step by step), /compare
docs/experiments/       pre-registered experiments and their results
runs/                   saved runs (git-ignored); runs/cli/<algo>/ = the latest terminal run of each algorithm
```
