import type { EpisodeRec, StepRec } from "./types";

/** Trailing moving average; nulls are skipped. */
export function movingAverage(xs: (number | null)[], w: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0;
  let n = 0;
  const q: (number | null)[] = [];
  for (const x of xs) {
    q.push(x);
    if (x != null) {
      sum += x;
      n++;
    }
    if (q.length > w) {
      const old = q.shift()!;
      if (old != null) {
        sum -= old;
        n--;
      }
    }
    out.push(n > 0 ? sum / n : null);
  }
  return out;
}

/** G_t = r_t + γ G_{t+1} for one episode, same convention as the backend's EV:
 *  after truncation the tail is V(s_T) (the last step's v_next), after termination 0. */
export function discountedReturns(steps: StepRec[], gamma: number): number[] {
  if (!steps.length) return [];
  const last = steps[steps.length - 1];
  let g = last.trunc && !last.term ? last.v_next : 0;
  const out = new Array(steps.length);
  for (let i = steps.length - 1; i >= 0; i--) {
    g = steps[i].r_train + gamma * g;
    out[i] = g;
  }
  return out;
}

/** First episode where a (smoothed) metric crosses a threshold. */
export function firstCrossing(
  eps: EpisodeRec[],
  value: (e: EpisodeRec) => number | null | undefined,
  test: (v: number) => boolean,
  smooth = 10,
): number | null {
  const ma = movingAverage(eps.map((e) => value(e) ?? null), smooth);
  for (let i = smooth - 1; i < eps.length; i++) {
    const v = ma[i];
    if (v != null && test(v)) return eps[i].ep;
  }
  return null;
}

export function fmt(x: number | null | undefined, digits = 3): string {
  if (x == null || Number.isNaN(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(2);
  return x.toFixed(a >= 100 ? 1 : digits);
}
