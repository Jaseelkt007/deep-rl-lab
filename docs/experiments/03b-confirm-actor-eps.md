# 03b · Confirmatory: does an actor-only Adam ε of 1e-3 prevent the CartPole collapse?

**Status:** pre-registration draft, committed before any run. **Not started:** it needs the user's go-ahead (40 long CartPole runs).

## Why

In 03, actor Adam ε 1e-3 had 0/8 learned runs collapse against 4/8 for the control (one-sided Fisher p = 0.038). Two problems left it undecided:

- It was one of 5 arm comparisons, so it would not survive a Bonferroni correction (threshold 0.01).
- The registered precondition, control collapse ≥ 60%, failed at 50%.

03 now estimates the control's collapse rate at about 50%. Keeping the ≥ 60% precondition would make a repeat undecidable by construction, whatever ε does. The precondition was a proxy for "the control collapses often enough to detect a fix", and a direct significance test does that job better. So 03b is **one** comparison with a direct test.

## Design

- CartPole-v1, plain, check LRs (actor 5e-4, critic 2.5e-3), 1000 episodes.
- **New held-out seeds 20–39** (20 per arm), never used before.
- Two arms only: **control** vs **actor adam ε 1e-3** (`actor_adam_eps=1e-3`, critic ε stays 1e-8).
- Metrics as in 01/03: learned (best MA100 > 400); collapsed (after learning, MA100 < 50); stays solved (share of the last 200 episodes > 400).

## Decision rule (one comparison, no correction)

- **fixes** = collapses in ≤ 20% of its learned seeds **and** one-sided Fisher exact p < 0.05 for fewer collapses than the control (collapse counted among learned seeds).
- **inconclusive** = fewer than 10 learned seeds in either arm.
- Otherwise: **does not fix**.

**Power check** (03 suggests ~80% learn, so ~16 learned per arm):
- control 8/16 vs arm 0/16: p ≈ 0.001;
- control 8/16 vs arm 1/16: p ≈ 0.008;
- control 6/16 vs arm 1/16: p ≈ 0.04.

So a real effect of the size seen in 03 should be detectable; a control rate near 30% would make it marginal.

## Prediction

Control: about 80% learn and about half of those collapse. Actor ε 1e-3: about 80% learn, at most 1–2 collapse, and entropy stays above 0.1 in nearly every learned run. Verdict: **fixes**.

## Results

*(not run yet)*
