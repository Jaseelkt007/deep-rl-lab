"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ACTION_COLORS, ACTION_GLYPH, C, divAction, divDelta, rgbCss, seqOrange } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { RunStart, SnapshotRec } from "@/lib/types";
import { ChartTitle } from "./Panel";

type Color = (v: number) => string;

interface HeatProps {
  nx: number;
  ny: number;
  values: (number | null)[]; // row-major, row = y index from low to high
  color: Color;
  xRange: number[];
  yRange: number[];
  traj?: number[][];
  names: string[];
  tip: (v: number | null) => string;
}

/** A G×G map over two state dims, y (velocity) pointing up, with the trajectory on top. */
function Heatmap({ nx, ny, values, color, xRange, yRange, traj, names, tip }: HeatProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number; px: number; py: number; v: number | null } | null>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth;
    const H = cv.clientHeight;
    cv.width = W * dpr;
    cv.height = H * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);
    const cw = W / nx;
    const ch = H / ny;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = values[j * nx + i];
        if (v == null) continue;
        ctx.fillStyle = color(v);
        ctx.fillRect(i * cw, H - (j + 1) * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    if (traj && traj.length > 1) {
      const px = (x: number) => ((x - xRange[0]) / (xRange[1] - xRange[0])) * W;
      const py = (y: number) => H - ((y - yRange[0]) / (yRange[1] - yRange[0])) * H;
      ctx.strokeStyle = "rgba(11,11,11,0.85)";
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      traj.forEach(([x, y], k) => (k ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
      ctx.stroke();
      ctx.fillStyle = C.ink;
      ctx.beginPath();
      ctx.arc(px(traj[0][0]), py(traj[0][1]), 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = C.ink;
      ctx.beginPath();
      const last = traj[traj.length - 1];
      ctx.arc(px(last[0]), py(last[1]), 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [nx, ny, values, color, traj, xRange, yRange]);

  return (
    <div className="relative">
      <canvas
        ref={ref}
        className="aspect-square w-full cursor-crosshair rounded-sm border border-neutral-200"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const fx = (e.clientX - r.left) / r.width;
          const fy = 1 - (e.clientY - r.top) / r.height;
          const i = Math.min(nx - 1, Math.max(0, Math.floor(fx * nx)));
          const j = Math.min(ny - 1, Math.max(0, Math.floor(fy * ny)));
          setHover({
            x: xRange[0] + fx * (xRange[1] - xRange[0]),
            y: yRange[0] + fy * (yRange[1] - yRange[0]),
            px: e.clientX - r.left,
            py: e.clientY - r.top,
            v: values[j * nx + i],
          });
        }}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded border border-neutral-200 bg-white px-1.5 py-1 text-[10px] leading-tight text-neutral-700 shadow"
          style={{ left: Math.min(hover.px + 10, 9999), top: hover.py + 10 }}
        >
          <div>
            {names[0]} {fmt(hover.x, 3)}
          </div>
          <div>
            {names[1]} {fmt(hover.y, 4)}
          </div>
          <div className="font-medium text-neutral-900">{tip(hover.v)}</div>
        </div>
      )}
      <div className="mt-0.5 flex justify-between text-[9px] text-neutral-400">
        <span>
          {names[0]} {fmt(xRange[0], 2)}
        </span>
        <span>{fmt(xRange[1], 2)}</span>
      </div>
    </div>
  );
}

function GradientLegend({ color, lo, hi, loLabel, hiLabel }: { color: Color; lo: number; hi: number; loLabel?: string; hiLabel?: string }) {
  const stops = Array.from({ length: 11 }, (_, k) => color(lo + ((hi - lo) * k) / 10));
  return (
    <div className="mt-1">
      <div className="h-1.5 rounded-sm" style={{ background: `linear-gradient(90deg, ${stops.join(",")})` }} />
      <div className="flex justify-between text-[9px] tabular-nums text-neutral-500">
        <span>{loLabel ?? fmt(lo, 2)}</span>
        <span>{hiLabel ?? fmt(hi, 2)}</span>
      </div>
    </div>
  );
}

export default function PhaseMaps({ snapshots, runStart, vStar }: { snapshots: SnapshotRec[]; runStart: RunStart | null; vStar: number | null }) {
  const [sel, setSel] = useState<number | null>(null); // null = follow the newest snapshot
  // "run": one V colour scale over all snapshots, so scrubbing shows V change, not the scale
  const [vScale, setVScale] = useState<"run" | "snapshot">("run");
  const n = snapshots.length;
  const idx = sel == null ? n - 1 : Math.min(sel, n - 1);
  const snap = idx >= 0 ? snapshots[idx] : null;
  const actionNames = useMemo(() => runStart?.action_names ?? ["left", "none", "right"], [runStart]);
  const names = runStart?.phase.names ?? ["position", "velocity"];
  const ranges = runStart?.phase.ranges ?? [
    [-1.2, 0.6],
    [-0.07, 0.07],
  ];

  const rewardScale = runStart?.config.reward_scale ?? 1;
  const runRange = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of snapshots) {
      for (const v of s.value) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
    return [lo, hi];
  }, [snapshots, snapshots.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const maps = useMemo(() => {
    if (!snap) return null;
    const sLo = Math.min(...snap.value);
    const sHi = Math.max(...snap.value);
    const meanV = snap.value.reduce((a, b) => a + b, 0) / snap.value.length;
    // flat = spread below 1% of the value's size: -100.01 … -99.99 is NOT structure
    const flat = sHi - sLo < 0.01 * Math.max(1, Math.abs(meanV));
    const [vLo, vHi] = vScale === "run" ? runRange : [sLo, sHi];
    const vColor: Color = (v) => rgbCss(seqOrange(flat && vScale === "snapshot" ? 0.6 : vHi - vLo < 1e-12 ? 0.6 : (v - vLo) / (vHi - vLo)));
    const ds = snap.delta_mean.filter((d): d is number => d != null).map(Math.abs).sort((a, b) => a - b);
    // floor tied to the reward scale, so δ ≈ 1e-5 noise renders as gray, not as vivid structure
    const p95 = ds.length ? ds[Math.floor(0.95 * (ds.length - 1))] : 0;
    const dMax = Math.max(p95, 0.05 * rewardScale);
    const dColor: Color = (v) => rgbCss(divDelta(0.5 + (0.5 * Math.max(-dMax, Math.min(dMax, v))) / dMax));
    const aColor: Color = (v) => ACTION_COLORS[actionNames[v]] ?? C.axis;
    const pColor: Color = (v) => rgbCss(divAction(v));
    return { vLo, vHi, sLo, sHi, flat, vColor, dMax, dColor, aColor, pColor };
  }, [snap, actionNames, vScale, runRange, rewardScale]);

  if (!snap || !maps) return <div className="py-8 text-center text-[12px] text-neutral-400">The first map appears when training starts.</div>;
  const G = snap.grid;
  const B = snap.delta_bins;
  const rightName = actionNames[actionNames.length - 1];

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-[11px] text-neutral-500">
        <span>snapshot</span>
        <input type="range" min={0} max={Math.max(0, n - 1)} value={Math.max(0, idx)} onChange={(e) => setSel(Number(e.target.value))} className="w-64" />
        <span className="w-24 tabular-nums text-neutral-800">after episode {snap.ep}</span>
        <button
          type="button"
          onClick={() => setSel(null)}
          className={`rounded border px-1.5 py-0.5 ${sel == null ? "border-blue-300 bg-blue-50 text-blue-700" : "border-neutral-300"}`}
        >
          latest
        </button>
        <span className="ml-2">line: that episode&apos;s trajectory, ● start, ○ end</span>
      </div>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <div>
          <ChartTitle help="The critic's value of each state (other state dims at 0 for CartPole). A successful MountainCar agent shows a spiral: high value along the orbit that swings up to the flag. A flat map at V* = −1/(1−γ) means: no reward ever seen, so nothing to tell states apart.">
            Critic V(s)
          </ChartTitle>
          <Heatmap nx={G} ny={G} values={snap.value} color={maps.vColor} xRange={ranges[0]} yRange={ranges[1]} traj={snap.traj} names={names} tip={(v) => `V = ${fmt(v)}`} />
          <GradientLegend color={maps.vColor} lo={maps.vLo} hi={maps.vHi} />
          <div className="flex items-center gap-1 text-[10px] text-neutral-500">
            colour scale:
            {(["run", "snapshot"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setVScale(m)}
                className={`rounded px-1 ${vScale === m ? "bg-neutral-200 text-neutral-800" : "hover:bg-neutral-100"}`}
              >
                {m === "run" ? "fixed over run" : "this snapshot"}
              </button>
            ))}
          </div>
          {maps.flat && (
            <div className="text-[10px] text-neutral-500">
              flat: all states ≈ {fmt((maps.sLo + maps.sHi) / 2)} (spread {fmt(maps.sHi - maps.sLo, 4)}){vStar != null ? `, V* = ${vStar.toFixed(0)}` : ""}
            </div>
          )}
        </div>
        <div>
          <ChartTitle help="The actor's most likely action in each state. The learned MountainCar policy pushes in the direction of the velocity (left when moving left, right when moving right): a split along velocity = 0.">
            Most likely action
          </ChartTitle>
          <Heatmap
            nx={G}
            ny={G}
            values={snap.best_action}
            color={maps.aColor}
            xRange={ranges[0]}
            yRange={ranges[1]}
            traj={snap.traj}
            names={names}
            tip={(v) => (v == null ? "" : `${ACTION_GLYPH[actionNames[v]]} ${actionNames[v]}`)}
          />
          <div className="mt-1 flex gap-2 text-[10px] text-neutral-600">
            {actionNames.map((a) => (
              <span key={a} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: ACTION_COLORS[a] }} />
                {ACTION_GLYPH[a]} {a}
              </span>
            ))}
          </div>
        </div>
        <div>
          <ChartTitle help={`π(${rightName}|s): how sure the actor is about pushing ${rightName}. Gray = 0.5, unsure; strong colour = nearly deterministic. A map that is one colour everywhere = the actor ignores the state.`}>
            P(push {rightName})
          </ChartTitle>
          <Heatmap nx={G} ny={G} values={snap.p_right} color={maps.pColor} xRange={ranges[0]} yRange={ranges[1]} traj={snap.traj} names={names} tip={(v) => `P = ${fmt(v)}`} />
          <GradientLegend color={maps.pColor} lo={0} hi={1} loLabel={`0 (${ACTION_GLYPH.left})`} hiLabel={`1 (${ACTION_GLYPH[rightName]})`} />
        </div>
        <div>
          <ChartTitle help="Mean TD error δ the actor received in each region since the previous snapshot (empty = not visited). + (aqua): outcomes better than the critic predicted, so actions there were reinforced; − (red): worse, so they were discouraged. Hover shows the signed number.">
            Mean δ received (±)
          </ChartTitle>
          <Heatmap
            nx={B}
            ny={B}
            values={snap.delta_mean}
            color={maps.dColor}
            xRange={ranges[0]}
            yRange={ranges[1]}
            traj={snap.traj}
            names={names}
            tip={(v) => (v == null ? "not visited" : `δ = ${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v), 4)}`)}
          />
          <GradientLegend color={maps.dColor} lo={-maps.dMax} hi={maps.dMax} loLabel={`− ${fmt(maps.dMax, 2)}`} hiLabel={`+ ${fmt(maps.dMax, 2)}`} />
        </div>
      </div>
    </div>
  );
}
