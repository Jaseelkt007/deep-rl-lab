"use client";

import { useMemo } from "react";
import { C } from "@/lib/colors";
import type { EpisodeRec, ProbeRec, RunConfig } from "@/lib/types";
import { MarkerLegend, collapseMarkers, unlabelled } from "./Curves";
import { ChartTitle } from "./Panel";
import UChart, { Marker, SeriesSpec } from "./UChart";

const DASH: (number[] | undefined)[] = [undefined, [5, 3], [1.5, 3], [8, 3, 2, 3]];
const NETS = [
  { key: "actor" as const, color: C.actor },
  { key: "critic" as const, color: C.critic },
];

function LayerChart({
  eps,
  title,
  help,
  pick,
  log,
  range,
  hiddenOnly,
  markers,
}: {
  eps: EpisodeRec[];
  title: string;
  help: string;
  pick: (e: EpisodeRec, net: "actor" | "critic", i: number) => number | null | undefined;
  log?: boolean;
  range?: [number | null, number | null];
  hiddenOnly?: boolean;
  markers: Marker[];
}) {
  const { x, ys, specs } = useMemo(() => {
    const withL = eps.filter((e) => e.layers && e.hidden);
    const first = withL[0];
    const x = withL.map((e) => e.ep);
    const ys: (number | null)[][] = [];
    const specs: SeriesSpec[] = [];
    if (first) {
      for (const { key, color } of NETS) {
        const names = hiddenOnly ? first.hidden![key].map((_, i) => `h${i + 1}`) : first.layers![key].map((l) => l.name);
        names.forEach((name, i) => {
          ys.push(withL.map((e) => pick(e, key, i) ?? null));
          specs.push({ label: `${key} ${name}`, color, width: 1.5, dash: DASH[i % DASH.length] });
        });
      }
    }
    return { x, ys, specs };
  }, [eps, eps.length, pick, hiddenOnly]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!x.length) return null;
  return (
    <div className="min-w-0">
      <ChartTitle help={help}>{title}</ChartTitle>
      <UChart x={x} ys={ys} series={specs} yLog={log} yRange={range} markers={markers} sync="episodes" height={150} />
    </div>
  );
}

// picks are module-level so their identity is stable across renders
const pickRatio = (e: EpisodeRec, n: "actor" | "critic", i: number) => e.layers?.[n][i]?.ratio;
const pickGrad = (e: EpisodeRec, n: "actor" | "critic", i: number) => e.layers?.[n][i]?.grad;
const pickSat = (e: EpisodeRec, n: "actor" | "critic", i: number) => e.hidden?.[n][i]?.saturated;
const pickDead = (e: EpisodeRec, n: "actor" | "critic", i: number) => e.hidden?.[n][i]?.dead;
const pickConst = (e: EpisodeRec, n: "actor" | "critic", i: number) => e.hidden?.[n][i]?.constant;
const pickRank = (e: EpisodeRec, n: "actor" | "critic", i: number) => e.hidden?.[n][i]?.erank;

function ProbeCharts({ probes, markers }: { probes: ProbeRec[]; markers: Marker[] }) {
  const data = useMemo(() => {
    const x = probes.map((p) => p.ep);
    return {
      x,
      returns: [
        probes.map((p) => p.returns.policy),
        probes.map((p) => p.returns.greedy),
        probes.map((p) => p.returns.fixed),
        probes.map((p) => p.returns.fixed_greedy),
      ],
      dep: [probes.map((p) => p.state_dependence)],
    };
  }, [probes, probes.length]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!probes.length) return null;
  return (
    <>
      <div className="min-w-0">
        <ChartTitle help="Every K episodes the real policy is replaced by ONE fixed, state-independent ranking: the actor's mean logits over recently visited states (gray), and scored on the same start states. If gray matches blue, the actor's choices do not depend on the state.">
          Fixed-ranking probe: score
        </ChartTitle>
        <UChart
          x={data.x}
          ys={data.returns}
          markers={markers}
          sync="episodes"
          height={150}
          series={[
            { label: "policy π(·|s)", color: C.actor, width: 2 },
            { label: "greedy argmax π", color: C.actor, width: 1.25, dash: [5, 3] },
            { label: "fixed ranking (sampled)", color: C.ink2, width: 2 },
            { label: "fixed best action", color: C.ink2, width: 1.25, dash: [5, 3] },
          ]}
        />
      </div>
      <div className="min-w-0">
        <ChartTitle help="Share of visited states where the actor's top action differs from the fixed ranking's top action. 0 = the same action everywhere (state ignored).">
          State dependence of the actor
        </ChartTitle>
        <UChart x={data.x} ys={data.dep} markers={markers} sync="episodes" height={150} yRange={[0, 1]} series={[{ label: "state dependence", color: C.actor, width: 2 }]} />
      </div>
    </>
  );
}

export default function Diagnostics({ eps, probes, cfg }: { eps: EpisodeRec[]; probes: ProbeRec[]; cfg?: RunConfig }) {
  const labelled = useMemo(() => collapseMarkers(eps, cfg), [eps, eps.length, cfg]); // eslint-disable-line react-hooks/exhaustive-deps
  const markers = useMemo(() => unlabelled(labelled), [labelled]);
  if (!eps.length) return <div className="py-8 text-center text-[12px] text-neutral-400">No episodes yet.</div>;
  const relu = cfg?.activation === "relu";
  return (
    <div>
      <MarkerLegend markers={labelled} />
      <div className="grid grid-cols-1 gap-x-4 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
      <LayerChart
        eps={eps}
        title="Update / weight ratio per layer"
        help="‖Δw‖ / ‖w‖ per update, per layer (log). Blue = actor, orange = critic; solid / dashed / dotted = first hidden / second hidden / output layer. ~1e-3 is a healthy rule of thumb; 1e-6 means a layer has stopped learning (e.g. LR decayed to ~0), 1e-1 means it is being thrown around."
        pick={pickRatio}
        log
        markers={markers}
      />
      <LayerChart
        eps={eps}
        title="Gradient norm per layer (pre-clip)"
        help="Mean gradient norm of each layer's weights and biases over the episode (log). Shows which network, and which layer, the learning signal reaches."
        pick={pickGrad}
        log
        markers={markers}
      />
      <LayerChart
        eps={eps}
        title={relu ? "Dead units (ReLU never active)" : "Saturated units |h| > 0.95"}
        help={
          relu
            ? "Share of each hidden layer's ReLU units that output 0 for every visited state: they pass no gradient."
            : "Share of tanh activations beyond ±0.95 on this episode's states. There the slope is below 0.1, so gradients barely pass. A saturated critic layer makes V nearly flat across states."
        }
        pick={relu ? pickDead : pickSat}
        hiddenOnly
        range={[0, 1]}
        markers={markers}
      />
      <LayerChart
        eps={eps}
        title="Effective rank of hidden features"
        help="exp(entropy of the singular values) of each hidden layer's (centred) activations over this episode's states. Low = every state maps to nearly the same features, so the network cannot tell states apart."
        pick={pickRank}
        hiddenOnly
        markers={markers}
      />
      <LayerChart
        eps={eps}
        title="Constant units"
        help="Share of hidden units whose output barely changes across this episode's states (std < 0.01): they carry no information about the state."
        pick={pickConst}
        hiddenOnly
        range={[0, 1]}
        markers={markers}
      />
      <ProbeCharts probes={probes} markers={markers} />
      </div>
    </div>
  );
}
