"use client";

import { useMemo } from "react";
import { C } from "@/lib/colors";
import { movingAverage } from "@/lib/math";
import type { EpisodeRec } from "@/lib/types";
import { ChartTitle } from "./Panel";
import UChart from "./UChart";

/** Explained variance of V against the return that followed, per episode. */
export function EvChart({ eps }: { eps: EpisodeRec[] }) {
  const { x, ys } = useMemo(() => {
    const clip = (v: number | null) => (v == null ? null : Math.max(-1, v));
    const ev = eps.map((e) => clip(e.ev));
    return { x: eps.map((e) => e.ep), ys: [ev, movingAverage(ev, 20), eps.map((e) => e.corr)] };
  }, [eps, eps.length]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div>
      <ChartTitle help="1 − Var(G − V)/Var(G) per episode, against the return of the reward the critic trains on. 1 = V explains the returns, 0 = no better than a constant, below 0 (clipped at −1) = worse. Gaps: the returns did not vary, nothing to explain. Dashed: correlation of V with G.">
        Critic explained variance
      </ChartTitle>
      <UChart
        x={x}
        ys={ys}
        height={150}
        yRange={[-1, 1]}
        sync="episodes"
        series={[
          { label: "EV (raw)", color: C.critic + "40", width: 1, noLegend: true },
          { label: "EV (avg 20)", color: C.critic, width: 2 },
          { label: "corr(V, G)", color: C.ink2, width: 1, dash: [4, 3] },
        ]}
      />
    </div>
  );
}
