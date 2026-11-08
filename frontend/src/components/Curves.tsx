"use client";

import { useMemo } from "react";
import { C } from "@/lib/colors";
import { firstCrossing, movingAverage } from "@/lib/math";
import type { EpisodeRec, RunConfig } from "@/lib/types";
import { ChartTitle } from "./Panel";
import UChart, { Marker, SeriesSpec } from "./UChart";

const MA = 20;
const faint = (hex: string) => hex + "40";

/** V* = r/(1−γ): the value of every state when the reward is a constant r per step forever
 *  (MountainCar, never reaching the goal, bootstrapping at the time limit). null if not applicable. */
export function noSignalValue(cfg: RunConfig | undefined): number | null {
  if (!cfg || cfg.env_id !== "MountainCar-v0" || cfg.shaping !== "none" || !cfg.bootstrap_on_truncation || cfg.gamma >= 1) return null;
  return -cfg.reward_scale / (1 - cfg.gamma);
}

/** Where the important events show up on the episode axis.
 *  CartPole: critic saturation > 90% (the start of the collapse chain), entropy < 0.1.
 *  MountainCar: "no signal" = on the states the episode visited, mean V within 1% of V* and
 *  mean |δ| < 0.05 reward units (the critic is RIGHT; the reward just carries no information).
 *  Tolerances are relative to the reward scale, and the backend logs V in raw (denormalised)
 *  units even with value_norm, so the marker means the same for any reward_scale. Both: first success. */
export function collapseMarkers(eps: EpisodeRec[], cfg?: RunConfig): Marker[] {
  const out: Marker[] = [];
  const vStar = noSignalValue(cfg);
  if (cfg?.env_id === "MountainCar-v0") {
    if (vStar != null) {
      const scale = cfg?.reward_scale ?? 1;
      const e = eps.find((e) => Math.abs(e.mean_v - vStar) < 0.01 * Math.abs(vStar) && e.mean_abs_delta < 0.05 * scale);
      if (e) out.push({ x: e.ep, label: "no signal: V≈V*, δ≈0", color: C.ink2 });
    }
  } else {
    const sat = firstCrossing(eps, (e) => e.hidden?.critic.at(-1)?.saturated, (v) => v > 0.9);
    if (sat != null) out.push({ x: sat, label: "critic sat > 90%", color: C.critic });
  }
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

/** One row above a chart grid that names the dashed marker lines (the charts draw lines only). */
export function MarkerLegend({ markers }: { markers: Marker[] }) {
  if (!markers.length) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-neutral-600">
      {markers.map((m) => (
        <span key={m.label} className="flex items-center gap-1.5">
          <svg width="14" height="10">
            <line x1="7" y1="0" x2="7" y2="10" stroke={m.color} strokeWidth="1.5" strokeDasharray="3 2" />
          </svg>
          {m.label} <span className="tabular-nums text-neutral-400">ep {m.x}</span>
        </span>
      ))}
    </div>
  );
}

export const unlabelled = (ms: Marker[]) => ms.map((m) => ({ ...m, label: "" }));

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

export default function Curves({ eps, cfg }: { eps: EpisodeRec[]; cfg?: RunConfig }) {
  const markers = useMemo(() => collapseMarkers(eps, cfg), [eps, eps.length, cfg]); // eslint-disable-line react-hooks/exhaustive-deps
  const lines = useMemo(() => unlabelled(markers), [markers]);
  if (!eps.length) return <div className="py-8 text-center text-[12px] text-neutral-400">No episodes yet.</div>;
  return (
    <div>
      <MarkerLegend markers={markers} />
      <div className="grid grid-cols-1 gap-x-4 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
        {CURVES.map((def) => (
          <EpisodeChart key={def.title} eps={eps} def={def} markers={lines} />
        ))}
      </div>
    </div>
  );
}

export { EpisodeChart };
export type { ChartDef };
