# 02 · MountainCar: how the RL formulation changes what can be learned

**Status:** done. Predictions committed before the runs (6e34c23), amended before the runs started (ad7c835); results below.

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

## Amendment (before the runs of this experiment started)

The "bootstrap off (bug)" prediction above is wrong as written (pointed out in review). A
time-blind critic cannot learn "−87 at the start, −1 near the limit": it sees roughly the
same states at every t, so V ≈ a constant c. Its TD(0) fixed point can be computed exactly:

- 199 ordinary steps: δ = −1 + γc − c = −1 − 0.01c. The fake-terminal step 200: δ = −1 − c.
- Σδ = −200 − 2.99c = 0  ⇒  **c ≈ −66.9**
- ⇒ δ ≈ **−0.33** on ordinary steps and δ ≈ **+65.9** on the last step of every episode.

So the bug does not create noise. It gives a large positive advantage to whatever action
happens at step 200, a systematic artefact. −66.9 is the TD fixed point, which is not the
average Monte Carlo return (≈ −57.1). Under function approximation, TD and MC converge to
different answers.

Known answers to check (code: `scripts/experiments.py`, from the last logged episode):

| Setup | Known answer |
|---|---|
| plain | V ≈ −100 for all t; δ ≈ 0 |
| bootstrap off (bug) | V ≈ −66.9 for all t; δ ≈ −0.33 for t < 199; δ ≈ +65.9 at t = 199 |
| time input + bootstrap off | V(t) = −(1 − 0.99^(200−t))/0.01: −86.6 at t = 0, −1.0 at t = 199 |

Corrected prediction for "bootstrap off (bug)": no success. V converges near −66.9, with
δ ≈ −0.33 everywhere except a large positive spike at the time limit.

## Results

Computed by `uv run python scripts/experiments.py analyse 02-mountaincar-formulation` (500 episodes, seeds 10–14). The known answers are read from the last logged episode (episode 476).

| Setup | Seeds with any success | Success rate, last 100 (per seed) | Mean \|δ\|, last 100 | Entropy, last 100 |
|---|---|---|---|---|
| plain | 0/5 | 0 | 7e-5 | **≈ 1e-5** |
| bootstrap off (bug) | 1/5 (s10: 9 successes, first at ep 442) | 0.09, 0, 0, 0, 0 | 0.67 | 1e-8 … 0.03 |
| time input + bootstrap off | 2/5 (s12: 13, s14: 1) | 0, 0, 0.10, 0, 0 | 0.67–0.86 | 0.01–0.05 |
| shaping (energy) | **5/5** | 0.0, **1.0, 0.99, 0.88**, 0.17 | 0.4–1.6 | 0.015–0.024 |
| n-step 10, GAE 0.95 | 0/5 | 0 | 6e-5 | ≈ 4e-5 |
| entropy 0.01 | 0/5 | 0 | 7e-5 | **1.10 (= ln 3, uniform)** |

### Known answers

| Setup | Known answer | Observed (5 seeds) |
|---|---|---|
| plain | V ≈ −100 for all t; δ ≈ 0 | ✓ V = −100.0 at t = 0 and t = 199; δ ≈ 1e-7 (t < 199) and ≤ 1.4e-4 (t = 199) |
| bootstrap off (bug) | V ≈ −66.9; δ ≈ −0.33 (t < 199); δ ≈ +65.9 (t = 199) | ✓ V = −68.8 to −69.2 on the last episode (mean V over the last 100 episodes −66.9 to −67.2); δ = −0.328; **δ = +68.2 at the time limit** (= −1 − V(s₁₉₉)) |
| time input + bootstrap off | V(t) = −(1 − 0.99^(200−t))/0.01: −86.6 at t = 0, −1.0 at t = 199 | **✗ only partly.** V(t=0) −68.7 to −83.9, V(t=199) −50.7 to −69.2. On s11 and s13 the critic ignored the time input entirely and sat at the bug's constant (−68.7/−69.1, δ = +68 at the limit). On the other three it learned only part of the slope. |

### Predicted vs observed

| Setup | Predicted | Observed |
|---|---|---|
| plain | no success; V → −100; δ → 0 | ✓ exactly. **Also (observation, not pre-registered): the policy became deterministic** (entropy ≈ 1e-5) with zero signal: Adam turned δ ≈ 1e-7 into full-size steps. |
| bootstrap off (bug) | no success; V ≈ −66.9; δ ≈ −0.33, with a +65.9 spike at the time limit | ✓ the numbers match. One seed reached the flag 9 times late in training: the spike rewards whatever happens at t = 199, a systematic artefact, and here it occasionally pushed the car up. |
| time input + bootstrap off | correct finite-horizon V(t); still no success signal | **✗ the critic mostly did not learn V(t)** (see the known answers); 2/5 seeds reached the flag a few times |
| shaping (energy) | learns: successes appear, the success rate rises | ✓ on 3/5 seeds (success ≥ 0.88 over the last 100). s14 reached 0.17. **s10 succeeded once at episode 11, then fell back to V = −100, δ ≈ 0 and a deterministic policy**: shaping gave a signal, but the policy collapsed before it was used. |
| n-step 10, GAE 0.95 | no help until the goal is reached once; most likely no success | ✓ 0/5; identical to plain (V = −100, δ ≈ 0) |
| entropy 0.01 | more random wandering; rarely enough to learn | ✓ 0/5. With δ ≈ 0 only the entropy gradient acts, so the policy stays exactly uniform (H = ln 3) |

### What this means

1. **No signal means no learning, however correct the critic is.** plain, n-step and entropy all sit at V = −100, δ ≈ 0. Multi-step returns cannot spread a reward that never arrives, and the entropy bonus can only make the random walk uniform.
2. **With Adam, "no signal" does not mean "no change".** Plain MountainCar ends with a deterministic policy (entropy ≈ 1e-5) that was built entirely from noise. This is the same mechanism experiment 03 tests on CartPole, where success, not failure, removes the signal.
3. **The time-limit bug is an artefact you can compute exactly.** V ≈ −67 and δ ≈ +68 on the last step of every episode, which the dashboard's δ chart shows as a spike at t = 199.
4. **Giving the time to the critic did not produce V(t) in these runs. The cause is not established.**
   - The code path is right: `test_time_input_learns_the_exact_finite_horizon_value` learns V(t) = (1 − γ^(T−t))/(1 − γ) within 5% on a deterministic loop (SGD; it gets 8.78 → 1.0 exactly in 1000 episodes).
   - A post-hoc check of "plasticity loss" did not separate the seeds: the critic's last hidden layer was about 100% saturated from episode 1 in **all 5** seeds, both the two that ignored time (s11, s13) and the three that learned part of the slope. Early saturation is universal here, so it cannot explain the difference.
5. **A formulation change (potential shaping) is what works.** It was the only setup that learned, on 3/5 seeds within 500 episodes. Even shaping is not robust with plain one-step actor-critic: one seed collapsed to a deterministic no-signal policy after its first success.
6. **Shaping only helps while the car keeps moving** (post-hoc, last logged episode of the shaping seeds).
   - s10 ended choosing "do nothing" on 200/200 steps, rocking near the bottom (x −0.62…−0.43, mean |v| 0.005).
   - There Φ barely changes, so the shaping bonus is ≈ 0.0009 per step and |δ| ≈ 1e-4.
   - It is the same no-signal trap again, now *with* shaping.
   - The three seeds that learned push mostly right with some left, swing over −1.0…0.49 and reach the flag in ~120 steps with |δ| ≈ 1.
7. **The entropy bonus opposes the Adam drift.** With δ ≈ 0 the entropy gradient is the only force left, so the policy stays exactly uniform (H = ln 3). Without it, plain drifts to entropy ≈ 1e-5. This fits experiment 01, where entropy was the only arm that let CartPole recover; whether that drift is the CartPole mechanism is what 03 tests.
