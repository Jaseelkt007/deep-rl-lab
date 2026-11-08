# 01 · Why does plain actor-critic collapse on CartPole?

**Status:** done. Predictions were committed before the runs (6e34c23), decision rules before any result was seen (ad7c835); results are added below without editing either.

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

Computed by `uv run python scripts/experiments.py analyse 01-cartpole-collapse` with the metrics and decision rules above, unchanged.

| Setup | Learned | Collapsed (of learned) | Median stays solved (learned) | Critic sat > 90% | Entropy < 0.1 | Verdict |
|---|---|---|---|---|---|---|
| control | 4/5 | 3/4 | 0.00 | 5/5 (ep 11–30) | 5/5 (ep 172–473) | (reference) |
| value_norm | 4/5 | 3/4 | 0.33 | **0/5** | 4/5 | does not fix |
| reward ×0.01 | 3/5 | 2/3 | 0.16 | **0/5** | 5/5 | does not fix |
| sgd (lr ×20) | 0/5 | – | – | – | – | **diverged on 5/5** (episodes 83–93) |
| entropy 0.01 | 3/5 | 2/3 | 0.43 | 5/5 | 3/5 | does not fix |

Per seed (best and final 100-episode moving average):

- control: s10 never learns (28.8 / 9.4); s11 500 / 11.9; s12 500 / 9.9; s13 500 / 27.2; s14 500 / **500**, stays solved (0.97).
- value_norm: s10 199 / 9.3; s11 443 / 443 (learned only at episode 979); s12 499 / 129; s13 500 / 61; s14 465 / 160.
- reward ×0.01: s10 403 / 97; s11 500 / 176; s12 145 / 32; s13 500 / 49; s14 382 / 10.
- entropy 0.01: s10 22 / 9.4; s11 435 / **365**; s12 347 / 217; s13 444 / 330; s14 476 / 306. The two "collapsed" seeds dipped below MA 50 and then **recovered**.

**The decision rule's precondition was not met.** The control collapsed in 3 of its 4 learned seeds, not ≥ 4. So with 5 seeds no setup could have been declared a fix. The verdicts above are "does not fix" in that limited sense.

### Predicted vs observed

| Setup | Predicted | Observed |
|---|---|---|
| control | most seeds learn, then collapse; saturation before entropy collapse | ✓ 4/5 learn, 3/4 collapse; saturation precedes entropy < 0.1 in all 5 runs (gap +156 to +443 episodes) |
| value_norm | **fixes it**, critic saturation stays low | critic saturation stays low ✓ (never > 90%), but **the policy still collapses (3/4)** ✗ |
| reward ×0.01 | fixes the critic (no saturation); unsure about the actor | ✓ no saturation; the actor still collapses (2/3); learns less often (3/5) |
| sgd (lr ×20) | slows the drift, the critic still saturates, learning slower / possibly unstable | **diverges on every seed before learning** (V → ~1e19, then NaN) ✗ |
| entropy 0.01 | delays the policy collapse without fixing the critic; returns degrade but do not crash to 9 | ✓ the critic still saturates; returns dip but end at 217–365 on 4 seeds instead of ~10 |

### What this means

1. **The target-scale hypothesis is half right.** The large value target does cause the critic saturation: both arms that keep targets near 1 (value_norm, reward ×0.01) remove it completely. But **removing the saturation does not stop the collapse**. In the control, saturation reliably comes first, yet it is a correlated symptom, not the cause. Order is not causation, and the value_norm arm is what showed it.
2. **Unsaturated is not the same as informative** (post-hoc check below). Entropy still runs to < 0.1 in 4/5 value_norm runs and 5/5 reward ×0.01 runs, but the critic was often still useless just before that. The evidence is mixed:
   - sometimes the critic stays uninformative even without saturation;
   - sometimes the actor collapses despite an informative critic.

   So "the collapse is actor-side" would be too strong. The only arm that clearly changes the outcome is the entropy bonus, which acts on the actor: returns degrade but recover.
3. **SGD diverging supports the target-scale story for the critic.** With SGD the critic's step is lr·δ·∇V, so it grows with the target. Critic lr 5e-2 is stable on the 5-state chain (targets ≤ 4.1, `tests/test_known_values.py`) and explodes on CartPole, where targets head towards 100.
4. **Procedural note.** The SGD arm was first run with an older trainer that crashed on NaN (in an SVD at the end of an episode) instead of recording divergence. The arm was rerun with the **identical pre-registered config**; only the recording code changed. Every shared episode record matched the crashed runs exactly (training is deterministic, one thread per process). The new code stops 3–4 episodes earlier, at the first non-finite update instead of the later crash. The crashed copies were deleted after the check.

### Post-hoc check (not pre-registered): was the critic informative just before the collapse?

Median critic EV and corr(V, G) over the 50 episodes before MA10 entropy first drops below 0.1
(`uv run python scripts/experiments.py posthoc 01-cartpole-collapse`):

| Setup | Critic in the 50 episodes before entropy < 0.1 |
|---|---|
| control | corr ≈ 0.01 in 3/5 runs, where EV is undefined because the returns were constant (500-step episodes); s12 informative (EV 0.84, corr 0.90); s10 EV −0.01, corr −0.42 |
| value_norm | **uninformative in 3 of the 4 runs that collapse**: EV −137 / −74 / −97, corr ≈ 0.0–0.04; s10 corr 0.89 but EV 0.04 |
| reward ×0.01 | informative in **2/5** (EV 0.91 and 0.84, corr 0.98 and 0.97), and those runs collapsed anyway; 3/5 uninformative (EV ≈ −50, corr 0.01–0.17) |
| entropy 0.01 | mixed: s11 EV 0.23, corr 0.80; s10 and s13 negative |

So removing saturation did not make the critic informative in most runs. value_norm's large negative EV comes partly from σ moving under a critic that is not output-preserving. And in at least two runs the actor collapsed while the critic was clearly informative. Both mechanisms seem to be at work. Experiment 03 tests the actor side directly.

### Next (exploratory, not pre-registered here)

- Actor-side tests: entropy bonus combined with value_norm; a smaller actor LR only; Adam with a larger ε (so tiny gradients stop producing full-size steps); SGD for the actor only.
- 01b (if wanted): an SGD critic LR chosen on tuning seeds 0–2 only, then one run on held-out seeds 10–14. Prediction written first: SGD + value_norm tolerates a much larger LR than plain SGD.
