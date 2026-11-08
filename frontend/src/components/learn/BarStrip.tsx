"use client";

import { useEffect, useRef } from "react";
import { C } from "@/lib/colors";

/** One bar per step of the batch (aqua +, red −), episode boundaries as gray lines.
 *  Optional line overlays share the same y axis. Click a bar to select that step. */
export default function BarStrip({
  values,
  boundaries,
  selected,
  onSelect,
  lines = [],
  height = 140,
  zeroLabel = "0",
}: {
  values: (number | null)[];
  boundaries: number[]; // indices where a new episode starts
  selected?: number | null;
  onSelect?: (i: number) => void;
  lines?: { values: (number | null)[]; color: string; width?: number; dash?: number[] }[];
  height?: number;
  zeroLabel?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const n = values.length;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const Wd = cv.clientWidth;
    const Hd = cv.clientHeight;
    cv.width = Wd * dpr;
    cv.height = Hd * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, Wd, Hd);
    const all = [...values, ...lines.flatMap((l) => l.values)].filter((v): v is number => v != null && Number.isFinite(v));
    let lo = Math.min(0, ...all);
    let hi = Math.max(0, ...all);
    if (hi - lo < 1e-9) {
      hi += 1;
      lo -= 1;
    }
    const padL = 36;
    const bw = (Wd - padL) / Math.max(1, n);
    const y = (v: number) => 6 + ((hi - v) / (hi - lo)) * (Hd - 12);
    // axis labels
    ctx.fillStyle = C.muted;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(hi.toFixed(Math.abs(hi) < 10 ? 2 : 0), padL - 4, 12);
    ctx.fillText(lo.toFixed(Math.abs(lo) < 10 ? 2 : 0), padL - 4, Hd - 4);
    ctx.fillText(zeroLabel, padL - 4, y(0) + 3);
    ctx.strokeStyle = C.axis;
    ctx.beginPath();
    ctx.moveTo(padL, y(0));
    ctx.lineTo(Wd, y(0));
    ctx.stroke();
    // episode boundaries
    ctx.strokeStyle = "#d4d4d4";
    for (const b of boundaries) {
      if (b === 0) continue;
      const x = padL + b * bw;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, Hd);
      ctx.stroke();
    }
    // bars
    values.forEach((v, i) => {
      if (v == null) return;
      ctx.fillStyle = v >= 0 ? C.pos : C.neg;
      const y0 = y(0);
      const y1 = y(v);
      ctx.fillRect(padL + i * bw, Math.min(y0, y1), Math.max(bw - (bw > 3 ? 1 : 0), 0.6), Math.max(1, Math.abs(y1 - y0)));
    });
    // lines
    for (const l of lines) {
      ctx.strokeStyle = l.color;
      ctx.lineWidth = l.width ?? 1.5;
      ctx.setLineDash(l.dash ?? []);
      ctx.beginPath();
      let pen = false;
      l.values.forEach((v, i) => {
        if (v == null || boundaries.includes(i)) pen = false;
        if (v == null) return;
        const x = padL + (i + 0.5) * bw;
        if (pen) ctx.lineTo(x, y(v));
        else ctx.moveTo(x, y(v));
        pen = true;
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // selection
    if (selected != null && selected >= 0 && selected < n) {
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(padL + selected * bw - 1, 1, bw + 2, Hd - 2);
    }
  }, [values, boundaries, selected, lines, n, zeroLabel]);

  return (
    <canvas
      ref={ref}
      className={`w-full ${onSelect ? "cursor-pointer" : ""}`}
      style={{ height }}
      onClick={(e) => {
        if (!onSelect) return;
        const r = e.currentTarget.getBoundingClientRect();
        const padL = 36;
        const i = Math.floor(((e.clientX - r.left - padL) / (r.width - padL)) * n);
        if (i >= 0 && i < n) onSelect(i);
      }}
    />
  );
}
