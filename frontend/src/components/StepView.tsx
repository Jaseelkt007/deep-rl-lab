"use client";

import { useMemo } from "react";
import { C } from "@/lib/colors";
import { discountedReturns, fmt } from "@/lib/math";
import type { StepRec } from "@/lib/types";
import { ChartTitle } from "./Panel";
import UChart from "./UChart";

function Num({ label, value, color, digits = 3 }: { label: string; value: number | null | undefined; color?: string; digits?: number }) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-neutral-100 py-0.5">
      <span className="text-[11px] text-neutral-500">{label}</span>
      <span className="text-[12px] tabular-nums" style={{ color: color ?? C.ink }}>
        {fmt(value, digits)}
      </span>
    </div>
  );
}

/** The selected step's numbers and δ over the steps of its episode. */
export function StepNumbers({ step }: { step: StepRec | null }) {
  const s = step;
  const dColor = s ? (s.delta > 0 ? C.pos : s.delta < 0 ? C.neg : C.ink) : C.ink;
  return (
    <div className="grid grid-cols-2 gap-x-4">
      <Num label="r (score)" value={s?.r_env} />
      <Num label="r (trains on)" value={s?.r_train} />
      <Num label="V(s)" value={s?.v} color={C.critic} />
      <Num label="V(s′)" value={s?.v_next} color={C.critic} />
      <Num label="target" value={s?.target} />
      <Num label={s && s.delta >= 0 ? "δ (+)" : "δ (−)"} value={s?.delta} color={dColor} />
      <Num label="advantage A" value={s?.adv} />
      <Num label="entropy H" value={s?.entropy} color={C.actor} />
      <Num label="actor loss" value={s?.actor_loss} color={C.actor} />
      <Num label="critic loss" value={s?.critic_loss} color={C.critic} />
      <Num label="|∇| actor" value={s?.gn_actor} color={C.actor} />
      <Num label="|∇| critic" value={s?.gn_critic} color={C.critic} />
      <Num label="‖Δw‖ actor" value={s?.dw_actor} color={C.actor} />
      <Num label="‖Δw‖ critic" value={s?.dw_critic} color={C.critic} />
    </div>
  );
}

export function DeltaChart({ steps, cursor }: { steps: StepRec[]; cursor: number | null }) {
  const { x, ys } = useMemo(() => {
    const x = steps.map((s) => s.t);
    return { x, ys: [steps.map((s) => s.delta), steps.map((s) => (s.delta > 0 ? s.delta : null)), steps.map((s) => (s.delta < 0 ? s.delta : null))] };
  }, [steps, steps.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const markers = cursor != null && steps[cursor] ? [{ x: steps[cursor].t, label: "", color: C.ink2 }] : undefined;
  return (
    <div>
      <ChartTitle help="The teaching signal at every step of this episode. Above 0: the action was reinforced; below 0: discouraged. A critic that predicts well makes δ small and centred on 0.">
        δ over this episode
      </ChartTitle>
      <UChart
        x={x}
        ys={ys}
        height={130}
        xLabel="step t"
        markers={markers}
        series={[
          { label: "δ", color: C.axis, width: 1 },
          { label: "δ > 0 (+)", color: C.pos, width: 1.5 },
          { label: "δ < 0 (−)", color: C.neg, width: 1.5 },
        ]}
      />
    </div>
  );
}

/** Critic quality on one finished episode: V(s_t) against the return that actually followed. */
export function ValueVsReturn({ steps, gamma, vStar }: { steps: StepRec[]; gamma: number; vStar: number | null }) {
  const { x, ys } = useMemo(() => {
    const g = discountedReturns(steps, gamma);
    const ys: (number | null)[][] = [steps.map((s) => s.v), g];
    if (vStar != null) ys.push(steps.map(() => vStar));
    return { x: steps.map((s) => s.t), ys };
  }, [steps, gamma, vStar]);
  return (
    <div>
      <ChartTitle help="Orange: what the critic predicted at each step. Black: the discounted return that actually followed, of the reward the critic trains on (after a time limit the unseen tail is filled with V(s_T)). If orange is flat while black moves, the critic is nearly constant and δ is mostly noise.">
        V(s) vs return that followed (last finished episode)
      </ChartTitle>
      {vStar != null && (
        <div className="mb-1 text-[10px] text-neutral-500">
          Dotted: V* = {vStar.toFixed(0)}, the true value when the flag is never reached. A critic sitting there is right, and δ ≈ 0 gives the actor nothing.
        </div>
      )}
      <UChart
        x={x}
        ys={ys}
        height={150}
        xLabel="step t"
        series={[
          { label: "V(s_t)", color: C.critic, width: 2 },
          { label: "return G_t", color: C.ink, width: 1.25 },
          ...(vStar != null ? [{ label: `V* = −1/(1−γ) = ${vStar.toFixed(1)}`, color: C.muted, width: 1, dash: [2, 3] }] : []),
        ]}
      />
    </div>
  );
}
