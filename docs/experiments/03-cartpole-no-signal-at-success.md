# 03 · Is the CartPole collapse Adam amplifying a no-signal δ?

**Status:** done. Predictions f39ab08, amendments 80afb53 and cad7836 (all before the held-out runs); results below.

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

Computed by `uv run python scripts/experiments.py analyse 03-cartpole-no-signal` (seeds 10–19, 1000 episodes).

**Correction made before writing this section.** The first analysis printed "FIXES" for actor ε 1e-3. The verdict code was still applying 01's absolute rule (control collapsed ≥ 4) instead of 03's registered fractional rule. Fixed in `a423945`; the 01 verdicts are unchanged by the fix.

| Arm | Learned | Collapsed (of learned) | Median stays solved (learned) | Fisher p vs control (one-sided) | Verdict (registered rule) |
|---|---|---|---|---|---|
| control | 8/10 | 4/8 (50%) | 0.30 | – | – |
| actor adam ε 1e-5 | 8/10 | 3/8 | 0.99 | 0.50 | does not fix |
| actor adam ε 1e-3 | 8/10 | **0/8** | **1.00** | **0.038** | **not decidable**: meets the fix condition, but the control precondition (≥ 60%) fails at 50% |
| actor SGD (lr 1e-3) | 8/10 | 2/8 | 1.00 | 0.30 | does not fix (25% > 20%) |
| actor lr ×0.3 | 10/10 | 7/10 | 0.36 | 0.91 | does not fix |
| entropy 0.01 + value_norm | **1/10** | 0/1 | – | – | inconclusive (learned on 1 seed) |

Further observations:
- **actor ε 1e-3**: entropy never fell below 0.1 in any of its 8 learned runs.
- **actor SGD**: of its two "collapses", s13 recovered (stays solved 1.0). The one lasting collapse (s12) happened with amplification ≈ 1 (plain SGD) and |δ| at 0.017 of its learning-phase value: that is not Adam amplification.

### Support criteria (from the amendment)

Computed by `support_03()` in `scripts/experiments.py`, printed as a separate block after the per-arm verdicts. The numbers below are copied from that output.

1. **The pre-collapse window is a no-signal success streak. ✓ in 3 of the 4 control runs that collapse.**
   - Truncated share: 1.0, 0.28, 1.0, 1.0 (median 1.0).
   - Amplification vs the learning phase: 1.9e7×, 8.8e17×, 3.9e7×, 4.9e11× (all ≥ 10×).
   - |δ| vs the learning phase: 0.00095, 0.48, 0.00069, 0.00006 (median ≈ 0.0008 ≤ 0.2).
   - The exception is s12: its collapse did not start inside a success streak (truncated share 0.28, δ ratio 0.48).
   - The same window pattern also appears in the four control runs where entropy fell below 0.1 but the policy *stayed* good (s14–s17): a deterministic policy that balances is not a failure by itself.
2. **adam ε 1e-3 or actor SGD fixes it. ✗ formally (not decidable).** The control collapsed in only 50% of its learned seeds, below the 60% the rule requires. Descriptively, ε 1e-3: 0/8 collapsed vs 4/8 (p = 0.038).
3. **The collapse share falls monotonically: control ≥ ε 1e-5 ≥ ε 1e-3. ✓** 4/8 ≥ 3/8 ≥ 0/8.

**Conclusion.** The hypothesis is **not formally confirmed**, because criterion 2 could not be decided under the rule as registered. The evidence points its way:
- the collapse starts in a no-signal success streak where Adam's step is 10⁷× or more larger relative to the gradient than during learning;
- actor ε 1e-3, which removes exactly that amplification and only for the actor, prevented every collapse and kept every learned run solved;
- the collapse share falls with ε.

Two things do not fit the pure story:
- actor SGD still had one lasting collapse (with no amplification);
- the control collapses less often than the earlier seeds suggested (50%, not "most").

### Predicted vs observed

| Arm | Predicted | Observed |
|---|---|---|
| control | 6–8/10 learn; **most** collapse; the window is a success streak with tiny \|δ\| and amplification ≫ 10 | 8/10 learn ✓; **only half (4/8) collapse** ✗; the window ✓ in 3/4 |
| actor adam ε 1e-5 | fewer or later collapses than the control | 3/8 vs 4/8 ✓ (weak, p = 0.5) |
| actor adam ε 1e-3 | fixes the collapse; learning may be a little slower | 0/8 collapse ✓ (not decidable by rule); **learning not slower** (first pass at episodes 140–308 vs 166–461) |
| actor SGD | fixes the collapse, but learns more slowly | **2/8 collapse** (one recovered) ✗ by the 20% rule; median stays solved 1.0; learning speed similar |
| actor lr ×0.3 | only delays the collapse | ✓ learns 10/10, then **7/10 collapse later** (episodes 301–955) |
| entropy 0.01 + value_norm | like entropy alone in 01: returns dip but recover | **✗ it barely learns: 1/10 seeds reach MA > 400** (best MA 240–466). Cause not checked. |

### Next

- **03b (confirmatory):** control vs actor ε 1e-3 only, on 20 new held-out seeds (20–39), with the same rule, written down before running. With 20 seeds per arm the precondition question also gets a better estimate of the control's collapse rate.
- Why entropy + value_norm (with the output-preserving fix) prevents learning: not explained yet.
