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

/** Two sticks on a free hinge. state = (x, ẋ, θ1, θ̇1, φ, φ̇); the top stick's angle is θ1 + φ. */
function drawDoubleCartPole(ctx: CanvasRenderingContext2D, W: number, H: number, step: StepRec | null, actionName: string | null) {
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
  const [x, , t1, , phi] = step.s;
  const t2 = t1 + phi;
  const cx = px(x);
  const cartW = 44;
  const cartH = 16;
  ctx.fillStyle = C.ink;
  ctx.fillRect(cx - cartW / 2, ground - cartH, cartW, cartH);
  const L = H * 0.31; // each stick: half the height of the single pole
  const base = [cx, ground - cartH];
  const joint = [base[0] + L * Math.sin(t1), base[1] - L * Math.cos(t1)];
  const tip = [joint[0] + L * Math.sin(t2), joint[1] - L * Math.cos(t2)];
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#c08a50";
  ctx.beginPath();
  ctx.moveTo(base[0], base[1]);
  ctx.lineTo(joint[0], joint[1]);
  ctx.stroke();
  ctx.strokeStyle = "#6250d6";
  ctx.beginPath();
  ctx.moveTo(joint[0], joint[1]);
  ctx.lineTo(tip[0], tip[1]);
  ctx.stroke();
  ctx.lineCap = "butt";
  ctx.fillStyle = C.ink2;
  ctx.beginPath();
  ctx.arc(joint[0], joint[1], 4, 0, Math.PI * 2);
  ctx.fill();
  if (actionName) {
    const dir = actionName === "left" ? -1 : 1;
    arrow(ctx, cx + dir * (cartW / 2 + 16), ground - cartH / 2, dir, ACTION_COLORS[actionName]);
  }
}

/** LunarLander from its 8 numbers: x (±1 = edge), y (0 = pad height), angle, leg contacts; the flame shows the
 *  engine that fires. The real terrain is not in the state, so the ground is drawn flat with the pad's flags. */
function drawLunarLander(ctx: CanvasRenderingContext2D, W: number, H: number, step: StepRec | null, actionName: string | null) {
  const m = 10;
  const px = (x: number) => m + ((x + 1.05) / 2.1) * (W - 2 * m);
  const ground = H - 18;
  const py = (y: number) => ground - (y / 1.65) * (ground - 12);
  ctx.strokeStyle = C.axis;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(m, ground);
  ctx.lineTo(W - m, ground);
  ctx.stroke();
  for (const fx of [-0.2, 0.2]) {
    const x = px(fx);
    ctx.strokeStyle = C.ink2;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, ground);
    ctx.lineTo(x, ground - 22);
    ctx.stroke();
    ctx.fillStyle = "#e8b100";
    ctx.beginPath();
    ctx.moveTo(x, ground - 22);
    ctx.lineTo(x + 9, ground - 18);
    ctx.lineTo(x, ground - 14);
    ctx.closePath();
    ctx.fill();
  }
  for (const ex of [-1, 1]) {
    ctx.strokeStyle = C.neg;
    ctx.beginPath();
    ctx.moveTo(px(ex), ground - 6);
    ctx.lineTo(px(ex), ground + 6);
    ctx.stroke();
  }
  if (!step) return;
  const [x, y, , , angle, , legL, legR] = step.s;
  ctx.save();
  ctx.translate(px(x), py(y) - 14); // the body sits above its legs; y = 0 means the legs touch the pad
  ctx.rotate(-angle); // Box2D: positive angle = counter-clockwise; the canvas y axis points down
  // flames (drawn first, under the body)
  const flame = (fx: number, fy: number, dx: number, dy: number) => {
    ctx.fillStyle = "#f19463";
    ctx.beginPath();
    ctx.moveTo(fx - dy * 4, fy + dx * 4);
    ctx.lineTo(fx + dx * 14, fy + dy * 14);
    ctx.lineTo(fx + dy * 4, fy - dx * 4);
    ctx.closePath();
    ctx.fill();
  };
  if (actionName === "main engine") flame(0, 9, 0, 1);
  // Gymnasium names the side engines by the way they PUSH: "left engine" (action 1) has its nozzle on the
  // right side and blows its gas to the right, so the lander goes left (and turns counter-clockwise).
  if (actionName === "left engine") flame(13, -4, 1, 0);
  if (actionName === "right engine") flame(-13, -4, -1, 0);
  // legs
  ctx.lineWidth = 2.5;
  for (const [side, down] of [
    [-1, legL],
    [1, legR],
  ] as const) {
    ctx.strokeStyle = down > 0.5 ? C.pos : C.ink2;
    ctx.beginPath();
    ctx.moveTo(side * 9, 6);
    ctx.lineTo(side * 15, 15);
    ctx.stroke();
  }
  // body
  ctx.fillStyle = "#6250d6";
  ctx.beginPath();
  ctx.moveTo(-9, -9);
  ctx.lineTo(9, -9);
  ctx.lineTo(12, 0);
  ctx.lineTo(12, 7);
  ctx.lineTo(-12, 7);
  ctx.lineTo(-12, 0);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
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
    else if (envId === "LunarLander-v3") drawLunarLander(ctx, W, H, step, actionName);
    else if (envId === "DoubleCartPole-v0" || envId === "DoubleCartPoleJoint-v0") drawDoubleCartPole(ctx, W, H, step, actionName);
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
              <span className="shrink-0 whitespace-nowrap text-neutral-600" style={{ minWidth: 48 }}>
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
