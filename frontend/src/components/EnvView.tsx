"use client";

import { useLayoutEffect, useRef } from "react";
import { ACTION_COLORS, ACTION_GLYPH, C } from "@/lib/colors";
import type { EnvId, StepRec } from "@/lib/types";

const hill = (x: number) => 0.45 * Math.sin(3 * x) + 0.55;

function arrow(ctx: CanvasRenderingContext2D, x: number, y: number, dir: number, color: string) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  if (dir === 0) {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x - 14 * dir, y);
    ctx.lineTo(x + 8 * dir, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + 16 * dir, y);
    ctx.lineTo(x + 6 * dir, y - 7);
    ctx.lineTo(x + 6 * dir, y + 7);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

function drawMountainCar(ctx: CanvasRenderingContext2D, W: number, H: number, step: StepRec | null, trail: number[], actionName: string | null) {
  const m = 14;
  const px = (x: number) => m + ((x + 1.2) / 1.8) * (W - 2 * m);
  const py = (y: number) => H - m - (y / 1.1) * (H - 2 * m - 10);
  // hill
  ctx.beginPath();
  ctx.moveTo(px(-1.2), H);
  for (let i = 0; i <= 200; i++) {
    const x = -1.2 + (1.8 * i) / 200;
    ctx.lineTo(px(x), py(hill(x)));
  }
  ctx.lineTo(px(0.6), H);
  ctx.closePath();
  ctx.fillStyle = "#f3f2ef";
  ctx.fill();
  ctx.strokeStyle = C.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = -1.2 + (1.8 * i) / 200;
    if (i === 0) ctx.moveTo(px(x), py(hill(x)));
    else ctx.lineTo(px(x), py(hill(x)));
  }
  ctx.stroke();
  // flag at the goal x = 0.5
  const fx = px(0.5);
  const fy = py(hill(0.5));
  ctx.strokeStyle = C.ink2;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(fx, fy);
  ctx.lineTo(fx, fy - 30);
  ctx.stroke();
  ctx.fillStyle = C.pos;
  ctx.beginPath();
  ctx.moveTo(fx, fy - 30);
  ctx.lineTo(fx + 16, fy - 25);
  ctx.lineTo(fx, fy - 20);
  ctx.closePath();
  ctx.fill();
  // where this episode has been
  ctx.fillStyle = "rgba(42,120,214,0.18)";
  for (const x of trail) ctx.fillRect(px(x) - 1, py(hill(x)) - 1, 2, 2);
  if (!step) return;
  // car
  const x = step.s[0];
  const cx = px(x);
  const cy = py(hill(x));
  const dydx = (py(hill(x + 0.01)) - py(hill(x - 0.01))) / (px(x + 0.01) - px(x - 0.01));
  const ang = Math.atan(dydx);
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(ang);
  ctx.fillStyle = C.ink;
  ctx.fillRect(-14, -14, 28, 10);
  ctx.fillStyle = C.ink2;
  for (const wx of [-8, 8]) {
    ctx.beginPath();
    ctx.arc(wx, -3, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  if (actionName) {
    const dir = actionName === "left" ? -1 : actionName === "right" ? 1 : 0;
    arrow(ctx, cx, cy - 32, dir, ACTION_COLORS[actionName]);
  }
}

function drawCartPole(ctx: CanvasRenderingContext2D, W: number, H: number, step: StepRec | null, actionName: string | null) {
  const m = 14;
  const px = (x: number) => m + ((x + 2.4) / 4.8) * (W - 2 * m);
  const ground = H - 22;
  ctx.strokeStyle = C.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(m, ground);
  ctx.lineTo(W - m, ground);
  ctx.stroke();
  for (const lim of [-2.4, 2.4]) {
    ctx.strokeStyle = C.neg;
    ctx.beginPath();
    ctx.moveTo(px(lim), ground - 6);
    ctx.lineTo(px(lim), ground + 6);
    ctx.stroke();
  }
  if (!step) return;
  const [x, , theta] = step.s;
  const cx = px(x);
  const cartW = 44;
  const cartH = 16;
  ctx.fillStyle = C.ink;
  ctx.fillRect(cx - cartW / 2, ground - cartH, cartW, cartH);
  const L = H * 0.62;
  const top = ground - cartH;
  ctx.strokeStyle = "#c08a50";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cx, top);
  ctx.lineTo(cx + L * Math.sin(theta), top - L * Math.cos(theta));
  ctx.stroke();
  ctx.lineCap = "butt";
  if (actionName) {
    const dir = actionName === "left" ? -1 : 1;
    arrow(ctx, cx + dir * (cartW / 2 + 16), ground - cartH / 2, dir, ACTION_COLORS[actionName]);
  }
}

export default function EnvView({
  envId,
  step,
  trail,
  actionNames,
}: {
  envId: EnvId;
  step: StepRec | null;
  trail: number[];
  actionNames: string[];
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const actionName = step ? actionNames[step.a] : null;

  useLayoutEffect(() => {
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
    if (envId === "MountainCar-v0") drawMountainCar(ctx, W, H, step, trail, actionName);
    else drawCartPole(ctx, W, H, step, actionName);
  }, [envId, step, trail, actionName]);

  return (
    <div>
      <canvas ref={ref} className="h-44 w-full" />
      <div className="mt-2 space-y-1">
        {actionNames.map((name, i) => {
          const p = step?.p[i] ?? 0;
          const chosen = step?.a === i;
          return (
            <div key={name} className="flex items-center gap-2 text-[11px]">
              <span className="w-12 text-neutral-600">
                {ACTION_GLYPH[name]} {name}
              </span>
              <div className="relative h-3 flex-1 rounded-sm bg-neutral-100">
                <div className="h-3 rounded-sm" style={{ width: `${p * 100}%`, background: ACTION_COLORS[name] }} />
              </div>
              <span className={`w-10 text-right tabular-nums ${chosen ? "font-semibold text-neutral-900" : "text-neutral-500"}`}>
                {p.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
