"use client";

import { useMemo } from "react";
import { C } from "@/lib/colors";
import { firstCrossing, movingAverage } from "@/lib/math";
import type { EpisodeRec } from "@/lib/types";
import { ChartTitle } from "./Panel";
import UChart, { Marker, SeriesSpec } from "./UChart";

const MA = 20;
const faint = (hex: string) => hex + "40";

/** Where the causal chain of a collapse shows up on the episode axis. */
export function collapseMarkers(eps: EpisodeRec[]): Marker[] {
  const out: Marker[] = [];
  const sat = firstCrossing(eps, (e) => e.hidden?.critic.at(-1)?.saturated, (v) => v > 0.9);
  if (sat != null) out.push({ x: sat, label: "critic sat > 90%", color: C.critic });
  const ent = firstCrossing(eps, (e) => e.mean_entropy, (v) => v < 0.1);
  if (ent != null) out.push({ x: ent, label: "entropy < 0.1", color: C.actor });
  const succ = eps.find((e) => e.success);
  if (succ) out.push({ x: succ.ep, label: "1st success", color: C.pos });
  return out;
}

interface ChartDef {
  title: string;
  help: string;
  series: { label: string; color: string; get: (e: EpisodeRec) => number | null | undefined; raw?: boolean; dash?: number[] }[];
  log?: boolean;
  range?: [number | null, number | null];
}

function EpisodeChart({ eps, def, markers }: { eps: EpisodeRec[]; def: ChartDef; markers: Marker[] }) {
  const { x, ys, specs } = useMemo(() => {
    const x = eps.map((e) => e.ep);
    const ys: (number | null)[][] = [];
    const specs: SeriesSpec[] = [];
    for (const s of def.series) {
      const raw = eps.map((e) => s.get(e) ?? null);
      if (s.raw !== false) {
        ys.push(raw);
        specs.push({ label: `${s.label} (raw)`, color: faint(s.color), width: 1, dash: s.dash, noLegend: true });
        ys.push(movingAverage(raw, MA));
        specs.push({ label: s.label, color: s.color, width: 2, dash: s.dash });
      } else {
        ys.push(raw);
        specs.push({ label: s.label, color: s.color, width: 1.75, dash: s.dash });
      }
    }
    return { x, ys, specs };
  }, [eps, eps.length, def]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="min-w-0">
      <ChartTitle help={def.help}>{def.title}</ChartTitle>
      <UChart x={x} ys={ys} series={specs} yLog={def.log} yRange={def.range} markers={markers} sync="episodes" height={140} />
    </div>
  );
}

export const CURVES: ChartDef[] = [
  {
    title: "Return (original reward)",
    help: "Sum of the environment's own reward per episode. MountainCar: −200 = never reached the flag. CartPole: 500 = balanced until the time limit. Shaping never changes this number.",
    series: [{ label: "return", color: C.ink, get: (e) => e.return }],
  },
  {
    title: "Success rate",
    help: "Share of the last 20 episodes that succeeded: reached the flag (MountainCar) or stayed up until the time limit (CartPole).",
    series: [{ label: "success", color: C.pos, get: (e) => (e.success ? 1 : 0), raw: false }],
    range: [0, 1],
  },
  {
    title: "Episode length",
    help: "Steps per episode. It ends at the goal / when the pole falls (terminated) or at the time limit (truncated).",
    series: [{ label: "length", color: C.ink2, get: (e) => e.length }],
  },
  {
    title: "Policy entropy H(π)",
    help: "How random the actor still is, averaged over the episode. ln 3 ≈ 1.10 (MountainCar) or ln 2 ≈ 0.69 (CartPole) is uniform; 0 is deterministic. A fast drop to 0 = the policy collapsed and can no longer explore or learn (∇log π ≈ 0).",
    series: [{ label: "entropy", color: C.actor, get: (e) => e.mean_entropy }],
  },
  {
    title: "Critic loss ½δ²",
    help: "How wrong the critic's one-step prediction was, averaged over the episode (log scale).",
    series: [{ label: "critic loss", color: C.critic, get: (e) => e.mean_critic_loss }],
    log: true,
  },
  {
    title: "Actor loss −A·log π",
    help: "Averaged over the episode. Its value is not a quality measure: only its gradient matters. Its sign follows the sign of δ.",
    series: [{ label: "actor loss", color: C.actor, get: (e) => e.mean_actor_loss }],
  },
  {
    title: "Gradient norm (pre / post clip)",
    help: "Size of each network's gradient before clipping (solid) and after (dashed). If the critic's is 10–100× the actor's, a joint clip mostly shrinks the actor's step.",
    series: [
      { label: "actor", color: C.actor, get: (e) => e.mean_gn_actor },
      { label: "actor clipped", color: C.actor, get: (e) => e.mean_gn_actor_clipped, dash: [4, 3], raw: false },
      { label: "critic", color: C.critic, get: (e) => e.mean_gn_critic },
      { label: "critic clipped", color: C.critic, get: (e) => e.mean_gn_critic_clipped, dash: [4, 3], raw: false },
    ],
    log: true,
  },
  {
    title: "Critic share of the gradient",
    help: "|∇critic|² / (|∇actor|² + |∇critic|²). Near 1 = the critic dominates any joint gradient norm (and a joint clip). Dashed: the factor the actor's gradient was scaled by (1 = not clipped).",
    series: [
      { label: "critic share", color: C.critic, get: (e) => e.mean_critic_share },
      { label: "actor clip factor", color: C.actor, get: (e) => e.mean_clip_actor, dash: [4, 3], raw: false },
    ],
    range: [0, 1],
  },
  {
    title: "Weight change per update ‖Δw‖",
    help: "How far each network's weights moved per step (log scale). With Adam this stays ~learning-rate sized even when δ ≈ 0, so noise still moves the policy. A decaying LR schedule makes it shrink.",
    series: [
      { label: "actor", color: C.actor, get: (e) => e.mean_dw_actor },
      { label: "critic", color: C.critic, get: (e) => e.mean_dw_critic },
    ],
    log: true,
  },
  {
    title: "KL(π_old ‖ π_new) per update",
    help: "How much one update changed the policy at the updated state (log scale). PPO limits exactly this; plain actor-critic does not.",
    series: [{ label: "KL", color: C.actor, get: (e) => e.mean_kl }],
    log: true,
  },
  {
    title: "Probability of the chosen action",
    help: "Mean π(a_t|s_t). Rising towards 1 = the policy becomes deterministic.",
    series: [{ label: "π(a|s)", color: C.actor, get: (e) => e.mean_p_a }],
    range: [0, 1],
  },
  {
    title: "|δ| and advantage spread",
    help: "Mean |δ| and the standard deviation of the advantage over the episode (reward units). n-step and GAE trade bias for variance: the spread grows with n and λ.",
    series: [
      { label: "|δ|", color: C.ink, get: (e) => e.mean_abs_delta },
      { label: "std A", color: C.ink2, get: (e) => e.adv_std, dash: [4, 3] },
    ],
  },
];

export default function Curves({ eps }: { eps: EpisodeRec[] }) {
  const markers = useMemo(() => collapseMarkers(eps), [eps, eps.length]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!eps.length) return <div className="py-8 text-center text-[12px] text-neutral-400">No episodes yet.</div>;
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
      {CURVES.map((def) => (
        <EpisodeChart key={def.title} eps={eps} def={def} markers={markers} />
      ))}
    </div>
  );
}

export { EpisodeChart };
export type { ChartDef };
