# 02 · MountainCar: how the RL formulation changes what can be learned

**Status:** predictions written down before the runs (this commit). Results are added below afterwards.

## Background

Plain actor-critic on MountainCar-v0 does not fail because of a bug or a broken critic:

- The reward is −1 per step. With γ = 0.99, bootstrapping at the time limit, and a policy that never reaches the flag, the true value of every state is V* = −1/(1−γ) = **−100**.
- The critic learns exactly that, so δ = −1 + 0.99·(−100) − (−100) = 0 everywhere.
- The actor gets no advantage signal at all: the critic is right, and that is why nothing learns.
- This differs from CartPole, where a flat V is a pathology (experiment 01).
- Adam still takes lr-sized steps on this δ ≈ 1e-5 (‖Δw‖ ≈ 1e-4), the same noise amplification as on CartPole.

## Design

- Base: MountainCar-v0, plain settings (actor 1e-3, critic 5e-3, γ = 0.99, 200-step limit), 500 episodes.
- Seeds 10–14. One change at a time.

## Metrics (code in `backend/scripts/experiments.py`)

- First successful episode, number of successes, success rate over the last 100 episodes.
- Mean of the highest position per episode, over the last 100 episodes.
- Mean |δ| and mean V over the last 100 episodes; entropy over the last 100 episodes.

## Predictions

| Setup | Prediction | Why |
|---|---|---|
| plain | no success; V → −100 everywhere; δ → 0; no learning | the no-signal fixed point |
| bootstrap off (bug) | no success; V depends on the hidden time left (about −87 at the start, about −1 near the limit), so δ is nonzero, but its sign is noise with respect to the actions | the time limit becomes a fake terminal state that the observation cannot see |
| time input + bootstrap off | no success; V becomes the correct finite-horizon value (a function of the time left), δ shrinks again; still no success signal | the problem is Markov again, but the reward still says nothing about the goal |
| shaping (energy) | **learns**: successes appear, and the success rate rises | F = γΦ(s′) − Φ(s) rewards building energy, so δ carries a signal before the goal is ever reached |
| n-step 10, GAE 0.95 | no help until the goal is reached once, then faster credit assignment; most likely no success within 500 episodes | multi-step returns propagate a signal that does not exist yet |
| entropy 0.01 | more random wandering; maybe an occasional success by chance, but rarely enough to learn within 200 steps | exploration by noise is very unlikely to climb the hill |

## Results

*(pending)*
