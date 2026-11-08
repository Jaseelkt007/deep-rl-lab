"use client";

import { ReactNode } from "react";
import { ACTION_COLORS, ACTION_GLYPH, C } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { StepRec } from "@/lib/types";

const sign = (x: number) => (x > 0 ? "+" : x < 0 ? "−" : "");
const deltaColor = (x: number) => (x > 0 ? C.pos : x < 0 ? C.neg : C.muted);
const updown = (x: number) => (x > 0 ? "↑" : x < 0 ? "↓" : "·");

function Stage({ i, title, color, children, animate }: { i: number; title: string; color?: string; children: ReactNode; animate: boolean }) {
  return (
    <div
      className={`min-w-0 rounded-md border bg-white px-2 py-1.5 ${animate ? "stage-in" : ""}`}
      style={{ borderColor: color ?? "#e5e5e5", animationDelay: animate ? `${i * 110}ms` : undefined }}
    >
      <div className="mb-1 text-[10px] font-medium uppercase tracking-wide" style={{ color: color ?? C.muted }}>
        {title}
      </div>
      <div className="text-[12px] leading-snug text-neutral-800">{children}</div>
    </div>
  );
}

const Arrow = () => <div className="self-center text-neutral-300">→</div>;

function ProbBars({ before, after, names, chosen }: { before: number[]; after?: number[]; names: string[]; chosen: number }) {
  return (
    <div className="space-y-0.5">
      {names.map((n, i) => (
        <div key={n} className="flex items-center gap-1 text-[10px]">
          <span className="w-3 text-neutral-500">{ACTION_GLYPH[n]}</span>
          <div className="relative h-2 flex-1 rounded-sm bg-neutral-100">
            <div className="absolute h-2 rounded-sm opacity-40" style={{ width: `${before[i] * 100}%`, background: ACTION_COLORS[n] }} />
            {after && <div className="absolute top-0 h-2 border-r-2" style={{ width: `${after[i] * 100}%`, borderColor: C.ink }} />}
          </div>
          <span className={`w-8 text-right tabular-nums ${i === chosen ? "font-semibold" : "text-neutral-500"}`}>{(after ?? before)[i].toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

/** One update, left to right, with the actual numbers of the selected step. */
export default function UpdateView({
  step,
  gamma,
  actionNames,
  obsNames,
  animate,
}: {
  step: StepRec | null;
  gamma: number;
  actionNames: string[];
  obsNames: string[];
  animate: boolean;
}) {
  if (!step) return <div className="py-10 text-center text-[12px] text-neutral-400">Start a run to see one update.</div>;
  const s = step;
  const name = actionNames[s.a];
  const nstep = s.window > 1 || Math.abs(s.adv_raw - s.delta) > 1e-6;
  const boot = s.window === 1 ? (s.bootstrap ? 1 : 0) : null;
  const dlogp = s.logp_after - s.logp;
  const key = `${s.ep}-${s.t}`;

  return (
    <div key={animate ? key : "static"} className="grid grid-cols-[1fr_auto_1fr_auto_1fr_auto_1.2fr_auto_1.4fr] gap-1.5">
      <Stage i={0} title="state s" animate={animate}>
        {s.s.slice(0, 4).map((v, i) => (
          <div key={i} className="flex justify-between gap-2">
            <span className="truncate text-neutral-500">{obsNames[i]?.replace("pole ", "")}</span>
            <span className="tabular-nums">{fmt(v, 3)}</span>
          </div>
        ))}
      </Stage>
      <Arrow />
      <div className="flex min-w-0 flex-col gap-1.5">
        <Stage i={1} title="actor π(·|s)" color={C.actor} animate={animate}>
          <ProbBars before={s.p} names={actionNames} chosen={s.a} />
        </Stage>
        <Stage i={1} title="critic V(s)" color={C.critic} animate={animate}>
          <span className="tabular-nums">{fmt(s.v)}</span>
        </Stage>
      </div>
      <Arrow />
      <Stage i={2} title="act, then env" animate={animate}>
        <div className="font-medium" style={{ color: ACTION_COLORS[name] }}>
          a = {ACTION_GLYPH[name]} {name}
        </div>
        <div className="tabular-nums">r = {fmt(s.r_train)}</div>
        {s.r_train !== s.r_env && <div className="text-[10px] text-neutral-500 tabular-nums">score r = {fmt(s.r_env)}</div>}
        <div className="mt-0.5 text-neutral-500">
          V(s′) = <span className="tabular-nums" style={{ color: C.critic }}>{fmt(s.v_next)}</span>
        </div>
        {(s.term || s.trunc) && (
          <div className="mt-0.5 text-[10px] font-medium" style={{ color: s.term ? C.neg : C.ink2 }}>
            {s.term ? "terminated: no bootstrap" : s.bootstrap ? "time limit: bootstrap" : "time limit: NO bootstrap (bug)"}
          </div>
        )}
      </Stage>
      <Arrow />
      <Stage i={3} title={nstep ? `advantage (${s.window}-step)` : "TD error δ"} color={deltaColor(s.adv_raw)} animate={animate}>
        {nstep ? (
          <div className="text-[11px] text-neutral-600">A = Σ (γλ)ᵏ δₖ over {s.window} steps, δ₀ = {fmt(s.delta)}</div>
        ) : (
          <div className="text-[11px] tabular-nums text-neutral-600">
            {fmt(s.r_train)} + {gamma}×{fmt(s.v_next)}
            {boot === 0 ? "×0" : ""} − ({fmt(s.v)})
          </div>
        )}
        <div className="mt-0.5 text-lg font-semibold tabular-nums" style={{ color: deltaColor(s.adv_raw) }}>
          {sign(s.adv_raw)}
          {fmt(Math.abs(s.adv_raw))} {updown(s.adv_raw)}
        </div>
        <div className="text-[10px] text-neutral-500">
          {s.adv_raw > 0 ? "better than expected" : s.adv_raw < 0 ? "worse than expected" : "as expected"}
        </div>
      </Stage>
      <Arrow />
      <div className="flex min-w-0 flex-col gap-1.5">
        <Stage i={4} title="actor step  −A·∇log π" color={C.actor} animate={animate}>
          <div className="tabular-nums">
            log π(a|s): {fmt(s.logp)} → {fmt(s.logp_after)}{" "}
            <span style={{ color: deltaColor(dlogp) }}>
              {updown(dlogp)} {sign(dlogp)}
              {fmt(Math.abs(dlogp), 4)}
            </span>
          </div>
          <ProbBars before={s.p} after={s.p_after} names={actionNames} chosen={s.a} />
          <div className="text-[10px] text-neutral-500 tabular-nums">
            KL {fmt(s.kl, 5)} · |∇| {fmt(s.gn_actor)}
            {s.clip_actor < 1 ? ` → ${fmt(s.gn_actor_clipped)} (clipped)` : ""}
          </div>
        </Stage>
        <Stage i={5} title="critic step  −δ·∇V" color={C.critic} animate={animate}>
          <div className="tabular-nums">
            V(s): {fmt(s.v)} → {fmt(s.v_after)}
          </div>
          <div className="text-[10px] text-neutral-500 tabular-nums">
            target {fmt(s.target)} · |∇| {fmt(s.gn_critic)}
            {s.clip_critic < 1 ? ` → ${fmt(s.gn_critic_clipped)} (clipped)` : ""}
          </div>
        </Stage>
      </div>
    </div>
  );
}
