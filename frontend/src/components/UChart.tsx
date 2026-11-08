"use client";

import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { C } from "@/lib/colors";

export interface SeriesSpec {
  label: string;
  color: string;
  width?: number;
  dash?: number[];
  fill?: string; // area fill (for bands)
  hidden?: boolean; // band edges: no line, no legend entry
  noLegend?: boolean; // drawn, but not listed (e.g. the faint raw series under a moving average)
}

function decades(lo: number, hi: number): number[] {
  if (!(lo > 0) || !(hi > 0)) return [];
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) out.push(10 ** e);
  return out.length >= 2 ? out : [lo, Math.sqrt(lo * hi), hi];
}
const tickFmt = (v: number) => {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-2) return v.toExponential(0).replace("e+", "e");
  return String(+v.toPrecision(3));
};

export interface Marker {
  x: number;
  label: string;
  color: string;
}

interface Props {
  x: number[];
  ys: (number | null)[][];
  series: SeriesSpec[];
  height?: number;
  yLog?: boolean;
  yRange?: [number | null, number | null];
  markers?: Marker[];
  bands?: { hi: number; lo: number; fill: string }[]; // indices into `series`
  sync?: string; // charts with the same key share the hover crosshair
  xLabel?: string;
}

/** Thin wrapper around uPlot: recreated when the series spec changes, otherwise setData. */
export default function UChart({ x, ys, series, height = 150, yLog, yRange, markers, bands, sync, xLabel }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const markersRef = useRef<Marker[] | undefined>(markers);

  const specKey = JSON.stringify({ series, yLog, yRange, bands, height, sync, xLabel });

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const opts: uPlot.Options = {
      width: Math.max(el.clientWidth, 120),
      height,
      pxAlign: false,
      cursor: { sync: sync ? { key: sync } : undefined, points: { size: 6 } },
      legend: { show: true, live: true },
      scales: {
        x: { time: false },
        y: yLog
          ? { distr: 3 }
          : { range: yRange ? (_u, lo, hi) => [yRange[0] ?? lo, yRange[1] ?? hi] : undefined },
      },
      axes: [
        {
          stroke: C.muted,
          grid: { stroke: C.grid, width: 1 },
          ticks: { show: false },
          font: "10px var(--font-geist-sans), sans-serif",
          label: xLabel,
          labelFont: "10px var(--font-geist-sans), sans-serif",
          labelSize: xLabel ? 14 : 0,
          size: 22,
        },
        {
          stroke: C.muted,
          grid: { stroke: C.grid, width: 1 },
          ticks: { show: false },
          font: "10px var(--font-geist-sans), sans-serif",
          size: 44,
          // log axes: one tick (and grid line) per decade
          splits: yLog ? (_u: uPlot, _i: number, lo: number, hi: number) => decades(lo, hi) : undefined,
          values: yLog ? (_u: uPlot, splits: number[]) => splits.map((v) => (v == null ? "" : tickFmt(v))) : undefined,
        },
      ],
      series: [
        { label: xLabel ?? "episode" },
        ...series.map((s) => ({
          label: s.label,
          stroke: s.hidden ? "transparent" : s.color,
          width: s.hidden ? 0 : (s.width ?? 1.5),
          dash: s.dash,
          fill: s.fill,
          points: { show: false },
          spanGaps: true,
          show: true,
          value: (_u: uPlot, v: number | null) => (v == null ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(3)),
        })),
      ],
      bands: bands?.map((b) => ({ series: [b.hi + 1, b.lo + 1] as [number, number], fill: b.fill })),
      hooks: {
        draw: [
          (u) => {
            const ms = markersRef.current;
            if (!ms?.length) return;
            const ctx = u.ctx;
            const { top, height: h } = u.bbox;
            ctx.save();
            ctx.font = `${10 * devicePixelRatio}px sans-serif`;
            ctx.textBaseline = "top";
            ms.forEach((m, k) => {
              const px = u.valToPos(m.x, "x", true);
              if (px < u.bbox.left || px > u.bbox.left + u.bbox.width) return;
              ctx.strokeStyle = m.color;
              ctx.lineWidth = 1.5 * devicePixelRatio;
              ctx.setLineDash([4 * devicePixelRatio, 3 * devicePixelRatio]);
              ctx.beginPath();
              ctx.moveTo(px, top);
              ctx.lineTo(px, top + h);
              ctx.stroke();
              if (m.label) {
                // stack labels so nearby markers stay readable
                const ly = top + (2 + 13 * k) * devicePixelRatio;
                const tw = ctx.measureText(m.label).width;
                ctx.fillStyle = "rgba(255,255,255,0.9)";
                ctx.fillRect(px + 2 * devicePixelRatio, ly - devicePixelRatio, tw + 4 * devicePixelRatio, 12 * devicePixelRatio);
                ctx.fillStyle = m.color;
                ctx.fillText(m.label, px + 4 * devicePixelRatio, ly);
              }
            });
            ctx.restore();
          },
        ],
      },
    };
    const u = new uPlot(opts, [x, ...ys] as uPlot.AlignedData, el);
    plot.current = u;
    // hide band-edge series from the legend
    series.forEach((s, i) => {
      if (s.hidden || s.noLegend) {
        const row = u.root.querySelectorAll(".u-legend .u-series")[i + 1] as HTMLElement | undefined;
        if (row) row.style.display = "none";
      }
    });
    const ro = new ResizeObserver(() => u.setSize({ width: Math.max(el.clientWidth, 120), height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      u.destroy();
      plot.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specKey]);

  useEffect(() => {
    markersRef.current = markers;
    const u = plot.current;
    if (!u) return;
    const data = yLog ? ys.map((col) => col.map((v) => (v != null && v > 0 ? v : null))) : ys;
    u.setData([x, ...data] as uPlot.AlignedData);
  }, [x, ys, yLog, markers]);

  return <div ref={box} className="uchart w-full" />;
}
