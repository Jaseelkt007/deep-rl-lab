# 03 · Is the CartPole collapse Adam amplifying a no-signal δ?

**Status:** predictions written down before any run of this experiment (this commit).

## Background (from 01)

- Removing critic saturation (value_norm, reward ×0.01) did not stop the collapse.
- In 3 of 5 control runs, the returns were *constant* just before entropy fell below 0.1. Those were 500-step episodes, truncated and bootstrapped, so every visited state has a true value of about 1/(1−γ) = +100.
- That is MountainCar's no-signal fixed point (V* = −100) with the sign flipped. The critic is right, δ ≈ 0, and the actor receives only noise.
- Adam divides by a running gradient RMS, so that noise still produces full-size steps (`test_adam_step_size_does_not_shrink_with_the_gradient`).

## Hypothesis: no signal at success

When the policy succeeds every time, the reward carries no information about the actions, so δ ≈ 0. Adam turns the remaining noise into a random drift of the policy until it becomes deterministic (entropy → 0, p → 1, ∇log π → 0) and can no longer recover. One lesson covers both tasks: whether the agent **never** succeeds (MountainCar) or **always** succeeds (CartPole), δ ≈ 0, and Adam turns noise into drift.

## Design

- Base: CartPole-v1, plain, check LRs (actor 5e-4, critic 2.5e-3), 1000 episodes.
- **Held-out seeds 10–19** (10 per arm). With 5 seeds, 01 could not reach a verdict. With 10, a split of 8/10 vs 2/10 gives one-sided Fisher p ≈ 0.01.
- One switch per arm:

| Arm | Change | Why |
|---|---|---|
| control | none | |
| adam ε 1e-5 | Adam ε = 1e-5 (default 1e-8) | late in training the actor gradient is ~1e-5, so √v̂ ~ 1e-5: this roughly halves the noise steps |
| adam ε 1e-3 | Adam ε = 1e-3 | nearly removes steps on gradients ≪ 1e-3 |
| actor SGD | SGD for the actor only, actor LR chosen as below; the critic stays Adam | SGD's step ∝ δ: no signal, no step |
| actor lr ×0.3 | actor LR 1.5e-4 | a smaller step, but still Adam-normalised |
| entropy 0.01 + value_norm | both (value_norm with the output-preserving fix) | does the critic matter for the collapse? |

**Choosing the actor-SGD LR (done before the held-out runs, on tuning seeds 0–2 only):**
- Grid: actor LR ∈ {1e-3, 3e-3, 1e-2, 3e-2}, 1000 episodes.
- Pick the **largest** LR that diverges on no seed and learns (best MA100 > 400) on ≥ 2 of 3 tuning seeds.
- If none qualifies, the arm is dropped and reported as such.
- The chosen LR is committed as an amendment before the held-out actor-SGD runs start.

## Metrics (code in `backend/scripts/experiments.py`)

As in 01:
- **learned**: best MA100 > 400.
- **collapsed**: after learning, MA100 < 50.
- **stays solved**: share of the last 200 episodes with return > 400.
- **diverged**: as before.

New, for the hypothesis, over the 50 episodes before MA10 entropy first drops below 0.1 (the "pre-collapse window"):
- **truncated share**: fraction of those episodes that ran to the 500-step limit.
- **|δ|**: median of the episode mean |δ|.
- **amplification**: median of the episode mean ‖Δw_actor‖ / (lr·‖∇_actor‖) (1 for SGD, ≫ 1 when Adam amplifies).

## Decision rules

- Collapse is counted among learned seeds.
- **fixes** = collapses in ≤ 20% of its learned seeds, **and** the control collapses in ≥ 60% of its learned seeds.
- **inconclusive** = fewer than 5 learned seeds (or the arm was dropped).
- Otherwise: **does not fix**.

**The hypothesis is supported** if all three hold:
1. In the control, the pre-collapse window is mostly a success streak (median truncated share ≥ 0.5 over the control runs that collapse) with |δ| < 0.05 and amplification > 10.
2. adam ε 1e-3 **or** actor SGD fixes it.
3. The collapse share falls monotonically: control ≥ ε 1e-5 ≥ ε 1e-3.

## Predictions

| Arm | Prediction |
|---|---|
| control | 6–8/10 learn; most collapse; the pre-collapse window is a success streak (truncated share ≥ 0.5) with tiny \|δ\| and amplification ≫ 10 |
| adam ε 1e-5 | fewer collapses than the control, or later ones |
| adam ε 1e-3 | **fixes** the collapse; learning may be a little slower |
| actor SGD | **fixes** the collapse, but learns more slowly |
| actor lr ×0.3 | only **delays** the collapse (as the LR sweep did), does not fix it |
| entropy 0.01 + value_norm | like entropy alone in 01: returns dip but recover; the critic does not matter for the collapse |

If ε and actor SGD do **not** reduce the collapse, the leading alternative is that the policy *forgets* the failure states: once it is good, the online data contains only balanced states. That would need a different test (a small replay, or resets to failure states), which is outside 03.

## Amendment 1 (after review, before any run of 03)

1. **The ε arms are actor-only.** `adam_eps` changes both networks, and at success the critic's gradient is tiny too, so ε = 1e-3 would also freeze the critic. A fix could then not be traced to the actor. A new switch, `actor_adam_eps`, sets ε for the actor only. The arms are now **actor adam ε 1e-5** and **actor adam ε 1e-3**; the critic keeps ε = 1e-8. This also makes them comparable with the actor-only SGD arm.
2. **The support criterion is relative to each run's own learning phase.** For Adam, ‖Δw‖/(lr·‖∇‖) ≈ 1/|typical gradient component|, which is already 10–100 during healthy learning, so "> 10" does not discriminate. Criterion 1 becomes:
   - median truncated share ≥ 0.5 in the pre-collapse window, over the control runs that collapse;
   - and median amplification in that window ≥ **10×** its median over episodes 50–100 (still learning);
   - and median |δ| in that window ≤ **0.2×** its median over episodes 50–100.

   `experiments.py` reports `amp_ratio` and `delta_ratio` per run.

## Amendment 2: actor-SGD LR (selected on tuning seeds 0–2, before any held-out run)

Grid run as `03a-select-actor-sgd-lr` (seeds 0–2, 1000 episodes; no run diverged):

| actor SGD LR | learned (of 3) | collapsed | stays solved |
|---|---|---|---|
| 1e-3 | **3/3** (episodes 197, 197, 218) | 0/3 | 0.995, 1.0, 1.0 |
| 3e-3 | 1/3 | 0/1 | 0, 0.01, 1.0 |
| 1e-2 | 1/3 | 0/1 | 1.0, 0, 0 |
| 3e-2 | 0/3 | – | 0, 0, 0 |

The rule ("largest LR that diverges on no seed and learns on ≥ 2/3") selects **actor SGD lr = 1e-3**. Larger SGD LRs mostly fail to learn at all.

On the tuning seeds, lr 1e-3 never collapsed. These are selection data, not the test; the predictions above are unchanged and the held-out seeds 10–19 decide.

## Results

*(pending)*
