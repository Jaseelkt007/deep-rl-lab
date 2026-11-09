"use client";

import { useEffect, useRef, useState } from "react";
import { C } from "@/lib/colors";
import { fmt } from "@/lib/math";

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
export function Heatmap({ nx, ny, values, color, xRange, yRange, traj, names, tip }: HeatProps) {
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
