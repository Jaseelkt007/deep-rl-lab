"use client";

import { C, divDelta, rgbCss } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { NetWeights } from "@/lib/learn";
import { maxAbs } from "@/lib/learn";

export type EdgeMode = "weights" | "grads" | "delta";

interface Props {
  title: string;
  color: string; // actor blue / critic orange
  edges: NetWeights; // what the edges show: weights, gradients or weight changes
  mode: EdgeMode;
  inputs: string[];
  outputs: string[];
  activations?: { x: number[]; h: number[]; out: number[]; outText?: string[] };
  focusLayer?: "W1" | "W2" | null; // dim the other layer (backprop animation)
}

const W = 480;
const H = 230;
const COLS = [118, 262, 376]; // room for input labels on the left and output labels on the right

function ys(n: number) {
  const top = 22;
  const bottom = H - 18;
  if (n === 1) return [(top + bottom) / 2];
  return Array.from({ length: n }, (_, i) => top + ((bottom - top) * i) / (n - 1));
}

/** Every weight drawn as an edge: colour = sign (aqua +, red −), thickness = size. */
export default function NetDiagram({ title, color, edges, mode, inputs, outputs, activations, focusLayer }: Props) {
  const nIn = edges.W1[0].length;
  const nH = edges.W1.length;
  const nOut = edges.W2.length;
  const [yIn, yH, yOut] = [ys(nIn), ys(nH), ys(nOut)];
  // one scale for the whole network, so W1 and W2 edges are comparable
  const m = Math.max(maxAbs(edges.W1), maxAbs(edges.W2));
  const edge = (v: number) => ({
    stroke: v >= 0 ? C.pos : C.neg,
    strokeWidth: 0.3 + 4 * (Math.abs(v) / m),
    strokeOpacity: 0.15 + 0.85 * (Math.abs(v) / m),
  });
  const dim = (layer: "W1" | "W2") => (focusLayer && focusLayer !== layer ? 0.12 : 1);
  const node = (v: number | undefined) => (v == null ? "#fff" : rgbCss(divDelta(0.5 + 0.5 * Math.max(-1, Math.min(1, v)))));
  const label = { weights: "weights", grads: "gradient ∂L/∂w", delta: "change Δw" }[mode];

  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-baseline gap-2">
        <span className="text-[12px] font-semibold" style={{ color }}>
          {title}
        </span>
        <span className="text-[11px] text-neutral-500">edges = {label}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-md border border-neutral-200 bg-white">
        <text x={COLS[0]} y={12} textAnchor="middle" className="fill-neutral-400 text-[9px]">
          input (state)
        </text>
        <text x={COLS[1]} y={12} textAnchor="middle" className="fill-neutral-400 text-[9px]">
          hidden (tanh)
        </text>
        <text x={COLS[2]} y={12} textAnchor="middle" className="fill-neutral-400 text-[9px]">
          output
        </text>
        <g opacity={dim("W1")} style={{ transition: "opacity 400ms" }}>
          {edges.W1.map((row, j) =>
            row.map((v, i) => (
              <line key={`a${j}-${i}`} x1={COLS[0]} y1={yIn[i]} x2={COLS[1]} y2={yH[j]} {...edge(v)} style={{ transition: "all 500ms" }}>
                <title>{`W1[hidden ${j + 1}][${inputs[i]}] = ${fmt(v, 4)}`}</title>
              </line>
            )),
          )}
        </g>
        <g opacity={dim("W2")} style={{ transition: "opacity 400ms" }}>
          {edges.W2.map((row, k) =>
            row.map((v, j) => (
              <line key={`b${k}-${j}`} x1={COLS[1]} y1={yH[j]} x2={COLS[2]} y2={yOut[k]} {...edge(v)} style={{ transition: "all 500ms" }}>
                <title>{`W2[${outputs[k]}][hidden ${j + 1}] = ${fmt(v, 4)}`}</title>
              </line>
            )),
          )}
        </g>
        {yIn.map((y, i) => (
          <g key={`in${i}`}>
            <circle cx={COLS[0]} cy={y} r={9} fill={node(activations?.x[i])} stroke={C.axis} />
            <text x={COLS[0] - 13} y={y + 3} textAnchor="end" className="fill-neutral-600 text-[9px]">
              {inputs[i]}
            </text>
            {activations && (
              <text x={COLS[0]} y={y + 3} textAnchor="middle" className="fill-neutral-900 text-[7px]">
                {activations.x[i].toFixed(2)}
              </text>
            )}
          </g>
        ))}
        {yH.map((y, j) => (
          <g key={`h${j}`}>
            <circle cx={COLS[1]} cy={y} r={nH > 12 ? 5 : 8} fill={node(activations?.h[j])} stroke={C.axis} />
            {activations && nH <= 12 && (
              <text x={COLS[1]} y={y + 2.5} textAnchor="middle" className="fill-neutral-900 text-[6.5px]">
                {activations.h[j].toFixed(2)}
              </text>
            )}
          </g>
        ))}
        {yOut.map((y, k) => (
          <g key={`o${k}`}>
            <circle cx={COLS[2]} cy={y} r={10} fill="#fff" stroke={color} strokeWidth={1.5} />
            <text x={COLS[2] + 14} y={y + 3} className="fill-neutral-700 text-[9px]">
              {outputs[k]}
              {activations?.outText ? ` ${activations.outText[k]}` : ""}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

/** A weight matrix as a grid: rows = to, columns = from. Numbers printed when it is small. */
export function MatrixGrid({
  title,
  m,
  scale,
  rowLabel,
  colLabels,
}: {
  title: string;
  m: number[][];
  scale?: number;
  rowLabel: (i: number) => string;
  colLabels: string[];
}) {
  const s = scale ?? maxAbs(m);
  const small = m[0].length <= 8 && m.length <= 16;
  return (
    <div className="min-w-0">
      <div className="mb-0.5 text-[10px] font-medium text-neutral-600">{title}</div>
      <div className="inline-grid gap-px rounded-sm bg-neutral-200 p-px" style={{ gridTemplateColumns: `auto repeat(${m[0].length}, minmax(0, ${small ? "34px" : "6px"}))` }}>
        <div className="bg-white" />
        {colLabels.map((c, i) => (
          <div key={i} className="truncate bg-white px-0.5 text-center text-[8px] text-neutral-500" title={c}>
            {small ? c.slice(0, 5) : ""}
          </div>
        ))}
        {m.map((row, r) => (
          <div key={r} className="contents">
            <div className="bg-white pr-1 text-right text-[8px] text-neutral-500">{small ? rowLabel(r) : ""}</div>
            {row.map((v, c) => (
              <div
                key={c}
                title={`${rowLabel(r)} ← ${colLabels[c]}: ${fmt(v, 4)}`}
                className="text-center text-[8px] tabular-nums leading-[14px]"
                style={{ background: rgbCss(divDelta(0.5 + 0.5 * Math.max(-1, Math.min(1, v / s)))), height: small ? 14 : 6 }}
              >
                {small ? (Math.abs(v) >= 0.01 || v === 0 ? v.toFixed(2) : v.toExponential(0)) : ""}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
