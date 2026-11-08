# 01 · Why does plain actor-critic collapse on CartPole?

**Status:** predictions written down before the runs (this commit). Results are added below afterwards, without editing the predictions.

## What we saw

Plain one-step actor-critic learns CartPole, then collapses to a return of about 9: a deterministic policy that always pushes the same way.
- **Default LRs** (actor 1e-3, critic 5e-3): MA100 reaches ~300–360 by episode 200, then collapses.
- **Check LRs** (actor 5e-4, critic 2.5e-3): first pass of MA100 > 400 at episodes 166–240 on 5 of 6 seeds, then every seed collapses (final MA100 9.4–12).

Lower LRs only delay it. The seed-0 trace shows this order:

1. The critic's last hidden layer saturates: more than 90% of its tanh units sit beyond ±0.95.
2. While the pole stays up, V(s) is about 100 everywhere and |δ| ≈ 0.
3. The actor's gradient is ≈ 0, but its weight change per step *grows*. Adam divides by its running gradient RMS, so a δ near 1e-5 still produces lr-sized steps.
4. Entropy falls to about 0.05 and π(a|s) reaches 0.99. The pole falls after about 12 steps, and at p ≈ 1, ∇log π ≈ 0, so the policy cannot recover.

## Hypothesis: the scale of the value target

CartPole returns grow from about 10 to about 100 (= 1/(1−γ)) as the agent improves. The critic's output layer is linear, so outputting 100 does not by itself require saturated hidden units. But while V chases a large and fast-moving target, the hidden weights grow until the tanh units saturate. Then V is nearly flat across states, δ is noise, and Adam turns that noise into full-size steps until the policy is deterministic. The value_norm arm separates "large target" from "saturation".

## Design

- Base: CartPole-v1, plain, check LRs (actor 5e-4, critic 2.5e-3), 1000 episodes.
- **Seeds 10–14 are held out**: none of them was used to choose the LRs.
- One switch at a time; the unchanged setup is the control on the same seeds.

| Setup | Change |
|---|---|
| control | none |
| value_norm | critic regresses target/σ, σ = running RMS of the targets |
| reward ×0.01 | the training reward is multiplied by 0.01, so returns stay ≤ 1 |
| sgd (lr ×20) | SGD with actor 1e-2 and critic 5e-2. SGD's step is lr·gradient, so Adam's LRs would barely move it. This also changes the LR, which is unavoidable. |
| entropy 0.01 | actor loss −= 0.01·H(π) |

## Metrics (fixed before running; code in `backend/scripts/experiments.py`)

- **learned**: best MA100 > 400.
- **collapsed**: after learning, MA100 < 50 at some later episode.
- **stays solved**: fraction of the last 200 episodes with return > 400.
- **order**: first episode where MA10 of critic last-layer saturation is > 0.9, first episode where MA10 of entropy is < 0.1, and the gap between them.

## Predictions

If the hypothesis is right:

| Setup | Prediction | Why |
|---|---|---|
| control | most seeds learn, then collapse; saturation > 90% comes before entropy < 0.1 | as seen on seeds 0–2 and 10–12 |
| value_norm | **fixes it**: learns, does not collapse, stays solved ≥ 0.8 on most learned seeds; critic saturation stays low | the network only ever regresses numbers near 1 |
| reward ×0.01 | **fixes the critic** (no saturation). Less sure about the actor: Adam makes the actor's step independent of δ's scale, so ×0.01 does not shrink it. It helps only if δ's sign carries signal again. | target scale near 1 |
| sgd (lr ×20) | **slows the drift but does not fix the critic**: the critic still saturates; steps ∝ δ, so δ ≈ 0 no longer moves the policy; collapse is late or absent, but learning is slower and possibly unstable | removes the noise amplification, not the cause |
| entropy 0.01 | **delays the policy collapse** without fixing the critic: entropy stays up, the critic still saturates, returns degrade but do not crash to 9 | treats the symptom (p → 1) |

If value_norm and reward ×0.01 do **not** stop the collapse, the target-scale hypothesis is wrong, which is the more interesting outcome.

## Amendment: decision rules (written after the runs started, before any result was seen)

With 5 seeds, only a near-complete split means anything. One-sided Fisher exact test, collapsed vs not:
5/5 vs 0/5 gives p ≈ 0.004; 5/5 vs 1/5 gives p ≈ 0.024; 4/5 vs 1/5 gives p ≈ 0.10, which is not significant.

- Collapse is counted **among the learned seeds only**; the number of learned seeds is reported separately.
- **fixes** = collapsed in ≤ 1 of its learned seeds **and** median stays_solved ≥ 0.8, **and** the control collapses in ≥ 4 of its learned seeds.
- **inconclusive (for collapse)** = the setup learns on fewer than 3 seeds (a run that never learns cannot collapse). This may well apply to "sgd (lr ×20)".
- Otherwise: **does not fix**.

## Results

*(pending)*
