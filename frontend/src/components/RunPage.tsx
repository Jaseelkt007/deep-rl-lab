"use client";

import Link from "next/link";
import { Fragment, ReactNode, createContext, useContext, useEffect, useMemo, useState } from "react";
import { STATIC, cliUrl } from "@/lib/api";
import { ACTION_COLORS, ACTION_GLYPH, C } from "@/lib/colors";
import type { EnvId, StepRec } from "@/lib/types";
import EnvView from "./EnvView";
import NavBar from "./NavBar";
import Panel from "./Panel";
import UChart, { LegendToggles, SeriesSpec } from "./UChart";

// ------------------------------------------------------------------ data
type Row = Record<string, number | null>;
type AlgoKey = "batch_ac" | "epochs" | "ratio" | "clip" | "gae" | "ppo";

interface AlgoConfig {
  batch_episodes?: number | null;
  batch_steps?: number | null;
  epochs?: number;
  minibatch?: number | null;
  ratio?: boolean;
  clip?: number | null;
  gae_lambda?: number;
  adv_norm?: boolean;
  hidden: number;
  actor_lr: number;
  critic_lr: number;
  gamma: number;
  optimizer?: string;
  seed: number;
  env_kwargs?: Record<string, number | boolean>; // gym.make options, e.g. LunarLander's wind and gravity
}
interface Settings {
  algo: AlgoKey;
  algo_name: string;
  budget: number | null;
  seed: number;
  config: AlgoConfig;
}
interface Status {
  state?: string;
  iter?: number;
  env_steps?: number;
  budget?: number | null;
  started?: string;
}
interface Quality {
  starts: number;
  max_steps: number;
  lengths: number[];
  survived: number;
  fell_cart: number;
  fell_pole: number;
  median_len: number;
  mean_abs_x: number;
  mean_abs_theta: number;
  switch_rate: number;
  survival_x: number[];
  survival: number[];
  scores?: number[]; // the final test's score per start
  median_score?: number;
  outcomes?: Record<string, number>; // LunarLander: how the episodes ended
}
interface Scorecard {
  speed_stability: {
    total_env_steps: number;
    iters: number;
    eval500_iter: number | null;
    eval500_steps: number | null;
    batch475_iter: number | null;
    batch475_steps: number | null;
    stay_475: number | null;
    drops_below_250: number;
    evals_500_after: number | null;
  };
  quality: { learned: Quality; good: Quality };
}
interface AlgoSummary {
  algo: AlgoKey;
  settings: Settings;
  status: Status;
  scorecard: Scorecard | null;
  speed: {
    iter: number;
    env_steps: number;
    ret: number;
    eval: number | null;
    KL?: number | null;
    clip_frac?: number | null;
    score?: number | null;
    eval_score?: number | null;
  }[];
  inner: Inner | null;
}
interface EpisodeInfo {
  name: string;
  iter: number;
  steps: number;
  terminated: boolean;
  kind?: "greedy" | "training" | "good"; // greedy: most likely action from the fixed start; training: a real batch episode
  score?: number | null; // its return with the official reward
  outcome?: string | null; // LunarLander: landed / crashed / flew away / time limit
}
type Mode = "greedy" | "training";
/** greedy recordings and training episodes; runs from before training episodes were recorded have no `kind` */
const ofKind = (e: EpisodeInfo, mode: Mode) => (e.kind ?? (e.name === "good" ? "good" : "greedy")) === mode;

/** The switch between the most likely action and the actions as sampled in training. */
function ModeSwitch({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  return (
    <span className="flex rounded bg-neutral-100 p-0.5 text-[11px]">
      {(
        [
          ["greedy", "most likely action"],
          ["training", "as in training (sampled)"],
        ] as const
      ).map(([k, label]) => (
        <button key={k} onClick={() => setMode(k)} className={`rounded px-1.5 ${mode === k ? "bg-white shadow-sm" : "text-neutral-500"}`}>
          {label}
        </button>
      ))}
    </span>
  );
}
interface RunData {
  algo: AlgoKey;
  settings: Settings;
  status: Status;
  rows: Row[];
  episodes: EpisodeInfo[];
  per_epoch: EpochRow[];
  scorecard: Scorecard | null;
}
/** After epoch k of iteration i: how far the policy is from π_old (the policy that played the batch). */
interface EpochRow {
  iter: number;
  epoch: number;
  kl: number;
  past: number; // share of steps past the clip limit: A > 0 and r > 1.2, or A < 0 and r < 0.8
  r_pos: number | null; // mean r = π/π_old on steps with A > 0
  r_neg: number | null; // ... and with A < 0
  r_hi: number | null; // 90th percentile of r on steps with A > 0 (the 10 % pushed up the most)
  r_lo: number | null; // 10th percentile of r on steps with A < 0 (the 10 % pushed down the most)
  r_max: number;
  r_min: number;
}
/** EpochRow averaged over the iterations where the batch was still learning (mean length < 475). */
interface Inner {
  n_iters: number;
  epoch: number[];
  kl: (number | null)[];
  past: (number | null)[];
  r_pos: (number | null)[];
  r_neg: (number | null)[];
  r_hi?: (number | null)[];
  r_lo?: (number | null)[];
}
interface Episode extends EpisodeInfo {
  states: number[][];
  actions: number[];
  probs: number[][];
  good_actions?: number[]; // the good policy's action at every step
}

/** From our batch actor-critic to PPO, one switch at a time (backend/rl/ppo.py). */
const LADDER: { key: AlgoKey; tab: string; name: string; adds: string; why: string; color: string }[] = [
  {
    key: "batch_ac",
    tab: "Batch AC",
    name: "Batch Actor-Critic",
    adds: "the start: 5 episodes → 1 gradient step → throw the batch away",
    why: "The starting point. The policy (the actor) plays 5 episodes; on every step the advantage A = G − V(s) says whether the action did better or worse than the critic expected. One gradient step on −A·log π makes the better actions more likely, and the critic moves V(s) towards the returns G. Then the batch is thrown away, because after the update it no longer comes from the current policy.",
    color: C.muted,
  },
  {
    key: "epochs",
    tab: "+ epochs",
    name: "Batch Actor-Critic + 10 epochs",
    adds: "use every batch 10 times: 10 gradient steps instead of 1, with the same loss −A·log π, as if the current policy had played the batch",
    why: "Batch AC with one change: each batch is used for 10 gradient steps instead of 1. The motivation is data efficiency: playing episodes is the expensive part, so learning more from every batch should need fewer environment steps. The catch: after the first step the policy is no longer the one that played the batch, but the loss still treats it as if it were, so the updates can overshoot.",
    color: "#6250d6",
  },
  {
    key: "ratio",
    tab: "+ ratio",
    name: "+ importance ratio (no clip)",
    adds: "loss −r·A with r = π(a|s)/π_old(a|s): corrects for the batch coming from π_old (r = 1 in epoch 1, so epoch 1 is unchanged), but nothing limits how far r goes",
    why: "Fixes that mismatch with importance sampling: each step's push is weighted by r = π(a|s) / π_old(a|s), how much more or less likely the action has become since the batch was played. In the first epoch r = 1, so nothing changes; in later epochs the gradient is correct for data from π_old. But nothing limits r, so one batch can still move the policy very far.",
    color: "#e34948",
  },
  {
    key: "clip",
    tab: "+ clip",
    name: "+ clip (ε = 0.2)",
    adds: "the brake: once r has moved past 1.2 (A > 0) or below 0.8 (A < 0), that step stops pushing",
    why: "Adds a brake (ε = 0.2): the loss becomes −min(r·A, clip(r, 0.8, 1.2)·A). Once an action has become 20% more likely (A > 0) or 20% less likely (A < 0), that step stops pushing, so one batch can move the policy only a limited amount. This keeps the extra epochs while preventing the large jumps the ratio alone allows.",
    color: "#eda100",
  },
  {
    key: "gae",
    tab: "+ GAE",
    name: "+ GAE (λ = 0.95)",
    adds: "advantage from the critic's one-step errors δ, weighted by (γλ)ᵏ, instead of the full return G − V",
    why: "Changes how the advantage is measured. Instead of the full return G − V(s), which carries the randomness of every later step, A is built from the critic's one-step errors δ = r + γV(s′) − V(s), weighted by (γλ)ᵏ with λ = 0.95. A little bias for much less noise, so the actor gets a cleaner signal; the critic's target becomes A + V_old.",
    color: "#e87ba4",
  },
  {
    key: "ppo",
    tab: "PPO",
    name: "PPO",
    adds: "fixed 2,048-step batches, minibatches of 64, advantages normalised per minibatch, smaller learning rates",
    why: "Proximal Policy Optimization: everything above (epochs, ratio, clip, GAE) plus the settings it is normally run with: fixed batches of 2,048 steps, 10 epochs over minibatches of 64 (320 small gradient steps per batch), advantages normalised per minibatch, and smaller learning rates. Many small, controlled steps per batch instead of a few large ones.",
    color: C.ink,
  },
];
const LADDER_BY = Object.fromEntries(LADDER.map((l) => [l.key, l])) as Record<AlgoKey, (typeof LADDER)[number]>;
const GOOD_COLOR = C.pos;

/** Everything that differs between environments. MountainCar plugs in here next. */
interface StateRow {
  name: string;
  unit: string;
  range: string;
  scale: number; // network input = (value − offset) / scale  (≈ ±1)
  offset?: number;
  deg?: boolean; // also show degrees
  derived?: (s: number[]) => number; // computed from the state, not one of its numbers
}
interface Trace {
  label: string;
  f: (s: number[]) => number; // scaled so that ±1 = the episode ends
  color: string;
  dash?: number[];
}
export interface EnvCfg {
  key: "cartpole" | "double_cartpole" | "double_cartpole_joint" | "mountaincar" | "mountaincar_energy" | "lunarlander" | "lunarlander_wind";
  envId: EnvId;
  title: string;
  tab: string;
  href: string;
  command: (algo: string) => string;
  actions: string[];
  state: StateRow[];
  reward: string;
  ends: string;
  goodPolicy: string;
  goodAction: (s: number[]) => number; // only for episodes recorded before good_actions were saved
  traces: Trace[];
  traceNote: string;
  endText: (s: number[], steps: number) => string;
  status: (s: number[]) => string; // one line under the cart in the side-by-side replay
  angleColumn: string; // the quality test's angle measure
  /** "limit": success = still going at the time limit, longer is better (the cart-poles).
   *  "flag": success = the episode ends at the goal, shorter is better (MountainCar). */
  success: "limit" | "flag" | "score"; // score: success = a score of at least 200 (LunarLander)
  lengthMax: number; // the time limit (y-axis of the length charts)
  label: {
    length: string; // what an episode's length means
    speed: string; // scorecard column: the greedy policy succeeds on all 10 eval starts
    firstGood: string; // the first solved batch
    stays: string;
    quality: (steps: number) => string;
    median: string;
    curve: string; // the final-test curve
  };
}
const CARTPOLE_LABELS = {
  length: "steps balanced",
  speed: "speed: steps to 500",
  firstGood: "first batch ≥ 475",
  stays: "stability: stays ≥ 475",
  quality: (n: number) => `survives ${n.toLocaleString()} steps`,
  median: "median survival",
  curve: "Quality: share of the 20 starts still balancing (final policy, long test)",
};
const DEG = 180 / Math.PI;
const D15 = 15 / DEG;
const D20 = 20 / DEG;
const D30 = 30 / DEG;
export const ENVS: Record<EnvCfg["key"], EnvCfg> = {
  cartpole: {
    key: "cartpole",
    envId: "CartPole-v1",
    title: "CartPole",
    tab: "CartPole",
    href: "/",
    command: (algo) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --algo ${algo}`,
    actions: ["left", "right"],
    state: [
      { name: "cart position", unit: "m", range: "±2.4 (end)", scale: 2.4 },
      { name: "cart velocity", unit: "m/s", range: "÷3.0", scale: 3.0 },
      { name: "pole angle", unit: "rad", range: "±0.21 = ±12° (end)", scale: 0.2095, deg: true },
      { name: "pole angular velocity", unit: "rad/s", range: "÷3.5", scale: 3.5 },
    ],
    reward: "+1 per step alive",
    ends: "terminated: pole > 12° or cart off track · truncated: 500 steps",
    goodPolicy: "angle + 0.5·angular velocity > 0 → right",
    goodAction: (s) => (s[2] + 0.5 * s[3] > 0 ? 1 : 0),
    traces: [
      { label: "cart position", f: (s) => s[0] / 2.4, color: C.ink },
      { label: "pole angle", f: (s) => s[2] / 0.2095, color: C.critic },
    ],
    traceNote: "cart position / 2.4 and pole angle / 12° (±1 = episode ends)",
    endText: (s, n) => (Math.abs(s[0]) / 2.4 > Math.abs(s[2]) / 0.2095 ? `cart left the track after ${n} steps` : `pole fell after ${n} steps`),
    status: (s) => `cart ${s[0].toFixed(2)} m · pole ${(s[2] * DEG).toFixed(1)}°`,
    angleColumn: "mean |pole angle|",
    success: "limit",
    lengthMax: 500,
    label: CARTPOLE_LABELS,
  },
  double_cartpole: {
    key: "double_cartpole",
    envId: "DoubleCartPole-v0",
    title: "Double CartPole",
    tab: "Double CartPole",
    href: "/double",
    command: (algo) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --env double_cartpole --algo ${algo}`,
    actions: ["left", "right"],
    state: [
      { name: "cart position", unit: "m", range: "±2.4 (end)", scale: 2.4 },
      { name: "cart velocity", unit: "m/s", range: "÷3.0", scale: 3.0 },
      { name: "lower stick angle θ1", unit: "rad", range: "±0.26 = ±15° (end)", scale: D15, deg: true },
      { name: "lower stick ang. velocity", unit: "rad/s", range: "÷3.5", scale: 3.5 },
      { name: "angle between sticks φ", unit: "rad", range: "÷0.52 = 30°", scale: 2 * D15, deg: true },
      { name: "angle between sticks, rate", unit: "rad/s", range: "÷5.0", scale: 5.0 },
      { name: "top stick angle θ1 + φ", unit: "rad", range: "±0.26 = ±15° (end)", scale: D15, deg: true, derived: (s) => s[2] + s[4] },
    ],
    reward: "+1 per step alive",
    ends: "terminated: either stick > 15° from vertical or cart off track · truncated: 500 steps",
    goodPolicy: "the LQR controller (textbook linear control) on ±10 N",
    goodAction: () => 0,
    traces: [
      { label: "cart position", f: (s) => s[0] / 2.4, color: C.ink },
      { label: "lower stick", f: (s) => s[2] / D15, color: "#c08a50" },
      { label: "top stick", f: (s) => (s[2] + s[4]) / D15, color: "#6250d6" },
    ],
    traceNote: "cart position / 2.4 and each stick's angle from vertical / 15° (±1 = episode ends)",
    endText: (s, n) => {
      const r = [Math.abs(s[0]) / 2.4, Math.abs(s[2]) / D15, Math.abs(s[2] + s[4]) / D15];
      const k = r.indexOf(Math.max(...r));
      return [`cart left the track`, `lower stick fell`, `top stick fell`][k] + ` after ${n} steps`;
    },
    status: (s) => `cart ${s[0].toFixed(2)} m · lower ${(s[2] * DEG).toFixed(1)}° · top ${((s[2] + s[4]) * DEG).toFixed(1)}°`,
    angleColumn: "mean |stick angle|",
    success: "limit",
    lengthMax: 500,
    label: CARTPOLE_LABELS,
  },
  double_cartpole_joint: {
    key: "double_cartpole_joint",
    envId: "DoubleCartPoleJoint-v0",
    title: "Double CartPole (joint rule)",
    tab: "Double · joint rule",
    href: "/double-joint",
    command: (algo) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --env double_cartpole_joint --algo ${algo}`,
    actions: ["left", "right"],
    state: [
      { name: "cart position", unit: "m", range: "±2.4 (end)", scale: 2.4 },
      { name: "cart velocity", unit: "m/s", range: "÷3.0", scale: 3.0 },
      { name: "lower stick angle θ1", unit: "rad", range: "±0.52 = ±30° (end)", scale: D30, deg: true },
      { name: "lower stick ang. velocity", unit: "rad/s", range: "÷3.5", scale: 3.5 },
      { name: "angle between sticks φ", unit: "rad", range: "±0.35 = ±20° (end)", scale: D20, deg: true },
      { name: "angle between sticks, rate", unit: "rad/s", range: "÷5.0", scale: 5.0 },
      { name: "top stick angle θ1 + φ", unit: "rad", range: "no limit of its own", scale: 1, deg: true, derived: (s) => s[2] + s[4] },
    ],
    reward: "+1 per step alive",
    ends: "terminated: lower stick > 30° from vertical, top stick bent > 20° away from the lower stick, or cart off track · truncated: 500 steps",
    goodPolicy: "the LQR controller (textbook linear control) on ±10 N",
    goodAction: () => 0,
    traces: [
      { label: "cart position", f: (s) => s[0] / 2.4, color: C.ink },
      { label: "lower stick", f: (s) => s[2] / D30, color: "#c08a50" },
      { label: "bend between sticks", f: (s) => s[4] / D20, color: "#6250d6" },
    ],
    traceNote: "cart position / 2.4, lower stick / 30° and the bend between the sticks / 20° (±1 = episode ends)",
    endText: (s, n) => {
      const r = [Math.abs(s[0]) / 2.4, Math.abs(s[2]) / D30, Math.abs(s[4]) / D20];
      const k = r.indexOf(Math.max(...r));
      return [`cart left the track`, `lower stick fell`, `top stick bent too far`][k] + ` after ${n} steps`;
    },
    status: (s) => `cart ${s[0].toFixed(2)} m · lower ${(s[2] * DEG).toFixed(1)}° · bend ${(s[4] * DEG).toFixed(1)}°`,
    angleColumn: "mean |stick angle|",
    success: "limit",
    lengthMax: 500,
    label: CARTPOLE_LABELS,
  },
  mountaincar: {
    key: "mountaincar",
    envId: "MountainCar-v0",
    title: "MountainCar",
    tab: "MountainCar",
    href: "/mountaincar",
    command: (algo) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --env mountaincar --algo ${algo}`,
    actions: ["left", "none", "right"],
    state: [
      { name: "position x", unit: "", range: "−1.2 … 0.6 (flag at 0.5)", scale: 0.9, offset: -0.3 },
      { name: "velocity v", unit: "", range: "±0.07", scale: 0.07 },
      { name: "height on the hill", unit: "", range: "0 … 1", scale: 1, derived: (s) => 0.45 * Math.sin(3 * s[0]) + 0.55 },
    ],
    reward: "−1 per step until the flag",
    ends: "terminated: position ≥ 0.5 (the flag) · truncated: 200 steps",
    goodPolicy: "push in the direction you are moving (v ≥ 0 → right, else left)",
    goodAction: (s) => (s[1] >= 0 ? 2 : 0),
    traces: [
      { label: "position", f: (s) => (s[0] + 0.3) / 0.9, color: C.ink },
      { label: "velocity", f: (s) => s[1] / 0.07, color: C.critic },
    ],
    traceNote: "position as the network sees it (the flag is at 0.89) and velocity / 0.07",
    endText: (_s, n) => `reached the flag after ${n} steps`,
    status: (s) => `x ${s[0].toFixed(3)} · v ${s[1] >= 0 ? "+" : ""}${s[1].toFixed(4)}`,
    angleColumn: "",
    success: "flag",
    lengthMax: 200,
    label: {
      length: "steps to the flag (200 = never)",
      speed: "speed: steps until every eval start reaches the flag",
      firstGood: "first batch where all reach the flag",
      stays: "stability: stays solved",
      quality: (n: number) => `reaches the flag within ${n} steps`,
      median: "median steps to the flag",
      curve: "Quality: share of the 20 starts that have reached the flag by step t (final policy)",
    },
  },
  mountaincar_energy: {
    key: "mountaincar_energy",
    envId: "MountainCar-v0",
    title: "MountainCar + energy bonus",
    tab: "MountainCar · energy",
    href: "/mountaincar-energy",
    command: (algo) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --env mountaincar_energy --algo ${algo}`,
    actions: ["left", "none", "right"],
    state: [
      { name: "position x", unit: "", range: "−1.2 … 0.6 (flag at 0.5)", scale: 0.9, offset: -0.3 },
      { name: "velocity v", unit: "", range: "±0.07", scale: 0.07 },
      { name: "height on the hill", unit: "", range: "0 … 1", scale: 1, derived: (s) => 0.45 * Math.sin(3 * s[0]) + 0.55 },
    ],
    reward: "−1 per step + 10·(γΦ(s′) − Φ(s)), Φ = energy (height + speed²) scaled to 0…1; training only, the scorecard uses the official −1",
    ends: "terminated: position ≥ 0.5 (the flag) · truncated: 200 steps",
    goodPolicy: "push in the direction you are moving (v ≥ 0 → right, else left)",
    goodAction: (s) => (s[1] >= 0 ? 2 : 0),
    traces: [
      { label: "position", f: (s) => (s[0] + 0.3) / 0.9, color: C.ink },
      { label: "velocity", f: (s) => s[1] / 0.07, color: C.critic },
    ],
    traceNote: "position as the network sees it (the flag is at 0.89) and velocity / 0.07",
    endText: (_s, n) => `reached the flag after ${n} steps`,
    status: (s) => `x ${s[0].toFixed(3)} · v ${s[1] >= 0 ? "+" : ""}${s[1].toFixed(4)}`,
    angleColumn: "",
    success: "flag",
    lengthMax: 200,
    label: {
      length: "steps to the flag (200 = never)",
      speed: "speed: steps until every eval start reaches the flag",
      firstGood: "first batch where all reach the flag",
      stays: "stability: stays solved",
      quality: (n: number) => `reaches the flag within ${n} steps`,
      median: "median steps to the flag",
      curve: "Quality: share of the 20 starts that have reached the flag by step t (final policy)",
    },
  },
  lunarlander: {
    key: "lunarlander",
    envId: "LunarLander-v3",
    title: "LunarLander",
    tab: "LunarLander",
    href: "/lunarlander",
    command: (algo) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --env lunarlander --algo ${algo}`,
    actions: ["nothing", "left engine", "main engine", "right engine"],
    state: [
      { name: "x (0 = pad centre)", unit: "", range: "±1 = edge (end)", scale: 1 },
      { name: "y (0 = pad height)", unit: "", range: "−0.2 … 1.6", scale: 0.9, offset: 0.7 },
      { name: "x velocity", unit: "", range: "÷2", scale: 2 },
      { name: "y velocity", unit: "", range: "÷2", scale: 2 },
      { name: "angle", unit: "rad", range: "÷2", scale: 2, deg: true },
      { name: "angular velocity", unit: "rad/s", range: "÷4", scale: 4 },
      { name: "left leg down", unit: "", range: "0 or 1", scale: 0.5, offset: 0.5 },
      { name: "right leg down", unit: "", range: "0 or 1", scale: 0.5, offset: 0.5 },
    ],
    reward: "shaped: closer to the pad, slower and level is better; each leg touching +10; main engine −0.3 and side engines −0.03 per step; +100 for landing, −100 for crashing. The side engines are named by the way they push: 'left engine' blows its gas to the right, so the lander moves left and turns counter-clockwise",
    ends: "terminated: landed (comes to rest), crashed (the body touches the ground) or flew away (|x| ≥ 1) · truncated: 1,000 steps",
    goodPolicy: "Gymnasium's hand-written heuristic",
    goodAction: () => 0,
    traces: [
      { label: "x", f: (s) => s[0], color: C.ink },
      { label: "height y / 1.6", f: (s) => s[1] / 1.6, color: "#6250d6" },
      { label: "angle / 1 rad", f: (s) => s[4], color: C.critic },
    ],
    traceNote: "x (±1 = the edges), height y / 1.6 and the angle in rad",
    endText: (_s, n) => `ended after ${n} steps`,
    status: (s) => `x ${s[0].toFixed(2)} · y ${s[1].toFixed(2)} · angle ${(s[4] * DEG).toFixed(1)}°${s[6] > 0.5 || s[7] > 0.5 ? " · leg down" : ""}`,
    angleColumn: "mean |angle|",
    success: "score",
    lengthMax: 1000,
    label: {
      length: "score (200 = solved)",
      speed: "speed: steps until the greedy policy averages ≥ 200",
      firstGood: "first batch averaging ≥ 200",
      stays: "stability: stays ≥ 200",
      quality: () => "score ≥ 200 (final test)",
      median: "median score",
      curve: "Quality: share of the 20 starts scoring at least s (final policy)",
    },
  },
  lunarlander_wind: {
    key: "lunarlander_wind",
    envId: "LunarLander-v3",
    title: "LunarLander + wind",
    tab: "LunarLander · wind",
    href: "/lunarlander-wind",
    command: (algo) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --env lunarlander_wind --algo ${algo}`,
    actions: ["nothing", "left engine", "main engine", "right engine"],
    state: [
      { name: "x (0 = pad centre)", unit: "", range: "±1 = edge (end)", scale: 1 },
      { name: "y (0 = pad height)", unit: "", range: "−0.2 … 1.6", scale: 0.9, offset: 0.7 },
      { name: "x velocity", unit: "", range: "÷2", scale: 2 },
      { name: "y velocity", unit: "", range: "÷2", scale: 2 },
      { name: "angle", unit: "rad", range: "÷2", scale: 2, deg: true },
      { name: "angular velocity", unit: "rad/s", range: "÷4", scale: 4 },
      { name: "left leg down", unit: "", range: "0 or 1", scale: 0.5, offset: 0.5 },
      { name: "right leg down", unit: "", range: "0 or 1", scale: 0.5, offset: 0.5 },
    ],
    reward: "shaped: closer to the pad, slower and level is better; each leg touching +10; main engine −0.3 and side engines −0.03 per step; +100 for landing, −100 for crashing. The side engines are named by the way they push: 'left engine' blows its gas to the right, so the lander moves left and turns counter-clockwise",
    ends: "with wind (strength 15) and random gusts (1.5) while in the air; the agent does not see them, only their effect on its speed · terminated: landed (comes to rest), crashed (the body touches the ground) or flew away (|x| ≥ 1) · truncated: 1,000 steps",
    goodPolicy: "Gymnasium's hand-written heuristic",
    goodAction: () => 0,
    traces: [
      { label: "x", f: (s) => s[0], color: C.ink },
      { label: "height y / 1.6", f: (s) => s[1] / 1.6, color: "#6250d6" },
      { label: "angle / 1 rad", f: (s) => s[4], color: C.critic },
    ],
    traceNote: "x (±1 = the edges), height y / 1.6 and the angle in rad",
    endText: (_s, n) => `ended after ${n} steps`,
    status: (s) => `x ${s[0].toFixed(2)} · y ${s[1].toFixed(2)} · angle ${(s[4] * DEG).toFixed(1)}°${s[6] > 0.5 || s[7] > 0.5 ? " · leg down" : ""}`,
    angleColumn: "mean |angle|",
    success: "score",
    lengthMax: 1000,
    label: {
      length: "score (200 = solved)",
      speed: "speed: steps until the greedy policy averages ≥ 200",
      firstGood: "first batch averaging ≥ 200",
      stays: "stability: stays ≥ 200",
      quality: () => "score ≥ 200 (final test)",
      median: "median score",
      curve: "Quality: share of the 20 starts scoring at least s (final policy)",
    },
  },
};
const EnvCtx = createContext<EnvCfg>(ENVS.cartpole);
const useEnv = () => useContext(EnvCtx);

// ------------------------------------------------------------------ helpers
const num = (v: number | null | undefined) => (v == null ? "never" : Math.round(v).toLocaleString());

/** How many gradient steps one batch gives, and how they are split. */
function updates(c: AlgoConfig): { n: number | null; text: string } {
  const e = c.epochs ?? 1;
  if (!c.minibatch) return { n: e, text: e === 1 ? "1 gradient step on the whole batch" : `${e} epochs × 1 gradient step on the whole batch` };
  if (c.batch_steps) {
    const m = Math.ceil(c.batch_steps / c.minibatch);
    return { n: e * m, text: `${e} epochs × ${m} minibatches of ${c.minibatch} steps` };
  }
  return { n: null, text: `${e} epochs × minibatches of ${c.minibatch} steps` };
}

function tags(c: AlgoConfig): string[] {
  const lam = c.gae_lambda ?? 1;
  const u = updates(c);
  return [
    "on-policy",
    c.batch_steps ? `${c.batch_steps.toLocaleString()}-step batches` : `${c.batch_episodes ?? 5}-episode batches`,
    u.n != null ? `${u.n} gradient step${u.n === 1 ? "" : "s"} per batch` : u.text,
    lam === 1 ? "Monte Carlo returns" : `GAE λ ${lam}`,
    c.clip ? `clip ε ${c.clip}` : c.ratio ? "ratio r·A, no clip" : (c.epochs ?? 1) > 1 ? "plain −A·log π, no ratio" : "plain −A·log π",
    ...(c.adv_norm ? ["normalised advantages"] : []),
  ];
}

/** Put curves with different x values on one shared, sorted x axis (gaps are bridged). */
function overlay(curves: { x: number[]; y: (number | null)[] }[]) {
  const x = Array.from(new Set(curves.flatMap((c) => c.x))).sort((a, b) => a - b);
  const at = new Map(x.map((v, i) => [v, i]));
  const ys = curves.map((c) => {
    const y: (number | null)[] = new Array(x.length).fill(null);
    c.x.forEach((v, i) => (y[at.get(v)!] = c.y[i]));
    return y;
  });
  return { x, ys };
}

function Kv({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[92px_1fr] gap-2 border-t border-neutral-100 py-1 text-[12px]">
      <span className="text-neutral-500">{k}</span>
      <span className="text-neutral-800">{children}</span>
    </div>
  );
}

const Mono = ({ children }: { children: ReactNode }) => <span className="font-mono text-[11.5px] text-neutral-900">{children}</span>;
const Swatch = ({ color, dashed }: { color: string; dashed?: boolean }) => (
  <span
    className="mr-1.5 inline-block h-[3px] w-3.5 shrink-0 align-middle"
    style={dashed ? { backgroundImage: `linear-gradient(90deg, ${color} 60%, transparent 60%)`, backgroundSize: "5px 3px" } : { background: color }}
  />
);

function Chart({ title, help, rows, series, log, range }: {
  title: string;
  help: string;
  rows: Row[];
  series: { key: string; label: string; color: string; dash?: number[] }[];
  log?: boolean;
  range?: [number | null, number | null];
}) {
  const { x, ys, specs } = useMemo(() => {
    const x = rows.map((r) => r.iter as number);
    // log axes cannot show 0
    const ys = series.map((s) => rows.map((r) => (r[s.key] == null || (log && (r[s.key] as number) <= 0) ? null : r[s.key])));
    const specs: SeriesSpec[] = series.map((s) => ({ label: s.label, color: s.color, width: 1.75, dash: s.dash }));
    return { x, ys, specs };
  }, [rows, series, log]);
  return (
    <Panel title={title} help={help}>
      <UChart x={x} ys={ys} series={specs} yLog={log} yRange={range} height={170} sync="run" xLabel="iteration" />
    </Panel>
  );
}

// ------------------------------------------------------------------ comparison
const B = ({ on, children }: { on: boolean; children: ReactNode }) => (on ? <b className="font-semibold text-neutral-950">{children}</b> : <>{children}</>);

function Compare({ algos, selected, onSelect }: { algos: AlgoSummary[]; selected: AlgoKey; onSelect: (a: AlgoKey) => void }) {
  const env = useEnv();
  // which "what is it and why" notes are open (a click on the name)
  const [open, setOpen] = useState<ReadonlySet<string>>(() => new Set());
  const flip = (k: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (!next.delete(k)) next.add(k);
      return next;
    });
  const [speedY, setSpeedY] = useState<"ret" | "eval">("ret");
  const { have, by } = useMemo(
    () => ({
      have: LADDER.filter((l) => algos.some((a) => a.algo === l.key)),
      by: Object.fromEntries(algos.map((a) => [a.algo, a])) as Partial<Record<AlgoKey, AlgoSummary>>,
    }),
    [algos],
  );

  const speed = useMemo(() => {
    const curves = have.map((l) => {
      // LunarLander is judged by its score, the others by their episode length
      const key = env.success === "score" ? (speedY === "ret" ? "score" : "eval_score") : speedY;
      const pts = by[l.key]!.speed.filter((p) => speedY === "ret" || p.eval != null);
      return { x: pts.map((p) => p.env_steps), y: pts.map((p) => p[key] ?? null) };
    });
    return { ...overlay(curves), series: have.map((l) => ({ label: l.tab, color: l.color, width: l.key === "ppo" ? 2 : 1.75 })) };
  }, [have, by, speedY, env.success]);

  const kl = useMemo(() => {
    const curves = have.map((l) => {
      const pts = by[l.key]!.speed.filter((p) => p.KL != null && p.KL > 0);
      return { x: pts.map((p) => p.env_steps), y: pts.map((p) => p.KL!) };
    });
    return { ...overlay(curves), series: have.map((l) => ({ label: l.tab, color: l.color, width: 1.5 })) };
  }, [have, by]);

  // inside one batch: after each epoch, averaged over the iterations that were still learning
  const inner = useMemo(() => {
    const withInner = have.filter((l) => by[l.key]!.inner);
    const curve = (k: "kl" | "past" | "r_hi" | "r_lo") =>
      withInner.map((l) => {
        const d = by[l.key]!.inner!;
        return { x: d.epoch, y: d[k] ?? d.epoch.map(() => null) };
      });
    const spec = (dash?: number[]) =>
      withInner.map((l) => ({ label: l.tab, color: l.color, width: 1.75, dash, points: by[l.key]!.inner!.epoch.length < 3 }));
    const kl = overlay(curve("kl"));
    const past = overlay(curve("past"));
    const r = overlay([...curve("r_hi"), ...curve("r_lo")]);
    const band = r.x.map(() => 1.2);
    const bandLo = r.x.map(() => 0.8);
    return {
      n: withInner.length,
      note: withInner.map((l) => `${l.tab} ${by[l.key]!.inner!.n_iters}`).join(" · "),
      kl: { ...kl, series: spec() },
      past: { ...past, series: spec() },
      r: {
        x: r.x,
        ys: [...r.ys, band, bandLo],
        series: [
          ...spec().map((sp) => ({ ...sp, label: `${sp.label} A>0`, group: sp.label })),
          ...spec([4, 3]).map((sp) => ({ ...sp, label: `${sp.label} A<0`, noLegend: true, group: sp.label })),
          { label: "clip 1.2", color: GOOD_COLOR, hidden: true },
          { label: "clip 0.8", color: GOOD_COLOR, hidden: true },
        ],
        bands: [{ hi: 2 * withInner.length, lo: 2 * withInner.length + 1, fill: "rgba(27,175,122,0.10)" }],
      },
    };
  }, [have, by]);

  const withCard = useMemo(() => have.filter((l) => by[l.key]!.scorecard), [have, by]);
  const good = withCard.length ? by[withCard[0].key]!.scorecard!.quality.good : null;
  const surv = useMemo(() => {
    const curves = withCard.map((l) => {
      const q = by[l.key]!.scorecard!.quality.learned;
      return { x: q.survival_x, y: q.survival };
    });
    if (good) curves.push({ x: good.survival_x, y: good.survival });
    return {
      ...overlay(curves),
      series: [
        ...withCard.map((l) => ({ label: l.tab, color: l.color, width: 2 })),
        ...(good ? [{ label: "good policy", color: GOOD_COLOR, width: 2, dash: [6, 4] }] : []),
      ],
    };
  }, [withCard, by, good]);

  // best value per column among the finished algorithms (bold in the table)
  const cards = withCard.map((l) => by[l.key]!.scorecard!);
  const best = (vals: (number | null | undefined)[], high: boolean) => {
    const v = vals.filter((x): x is number => x != null);
    return v.length < 2 ? null : high ? Math.max(...v) : Math.min(...v);
  };
  const bestSpeed = best(cards.map((c) => c.speed_stability.eval500_steps), false);
  const bestBatch = best(cards.map((c) => c.speed_stability.batch475_steps), false);
  const bestStay = best(cards.map((c) => c.speed_stability.stay_475), true);
  const bestSurv = best(cards.map((c) => c.quality.learned.survived), true);
  const flag = env.success === "flag";
  const bestMedian = best(
    cards.map((c) => (env.success === "score" ? c.quality.learned.median_score : c.quality.learned.median_len)),
    !flag,
  ); // MountainCar: fewer steps is better; LunarLander: the median score

  const cell = "px-2 py-1.5 text-right tabular-nums";
  const nCols = 8 + (flag ? 0 : 2) + (env.success === "score" ? 1 : 0);
  const nameCell = (k: string) => ({
    className: "whitespace-nowrap px-2 py-1.5 text-left",
    title: open.has(k) ? "hide the explanation" : "what is it, and why?",
    onClick: () => flip(k), // the row's own click (open this algorithm below) still happens
  });
  const chevron = (k: string) => (
    <span aria-hidden className={`mr-1 inline-block w-2.5 text-[9px] text-neutral-400 transition-transform ${open.has(k) ? "rotate-90" : ""}`}>
      ▶
    </span>
  );
  const note = (k: string, children: ReactNode) =>
    open.has(k) ? (
      <tr>
        <td colSpan={nCols} className="px-2 pb-2.5 pt-0 text-left">
          <div className="ml-5 max-w-[880px] rounded-md bg-[#f6f6f4] px-3 py-2 text-[12px] leading-relaxed text-neutral-700">{children}</div>
        </td>
      </tr>
    ) : null;
  const head = "px-2 py-1 text-right font-normal";
  const maxSteps = good?.max_steps ?? 10000;
  const qualityCells = (q: Quality, bold: boolean) => (
    <>
      <td className={cell}>
        <B on={bold && q.survived === bestSurv}>
          {q.survived}/{q.starts}
        </B>
      </td>
      <td className={cell}>
        <B on={bold && (env.success === "score" ? q.median_score : q.median_len) === bestMedian}>
          {Math.round((env.success === "score" ? q.median_score : q.median_len) ?? 0).toLocaleString()}
        </B>
      </td>
      {!flag && (
        <td className={cell}>
          {q.mean_abs_x.toFixed(2)}
          {env.success === "score" ? "" : " m"}
        </td>
      )}
      {!flag && <td className={cell}>{q.mean_abs_theta.toFixed(3)} rad</td>}
      {env.success === "score" && (
        <td className={cell}>
          {Object.entries(q.outcomes ?? {})
            .map(([k, v]) => `${v} ${k}`)
            .join(" · ") || "–"}
        </td>
      )}
      <td className={cell}>{(q.switch_rate * 100).toFixed(0)}%</td>
    </>
  );

  return (
    <Panel
      title="Compare the algorithms"
      help="Every algorithm gets the same budget of environment steps (200,000). Their iterations are NOT comparable (a PPO iteration is 2,048 steps, a batch actor-critic iteration is 5 episodes), so the x-axis here is environment steps. Speed: steps until the greedy policy first succeeds on all 10 eval starts (CartPole: balances 500 steps; MountainCar: reaches the flag). Stability: after the first solved batch, how often later batches stay solved. Quality: the FINAL policy, most likely action, tested from 20 start states (CartPole: up to 10,000 steps; MountainCar: within the 200-step limit, fewer steps is better). Click a name to read what it is and why it was added; the click also opens that algorithm below."
    >
      <div className="space-y-3">
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead className="text-[11px] text-neutral-500">
              <tr>
                <th className="px-2 py-1 text-left font-normal">algorithm</th>
                <th className={head}>gradient steps per batch</th>
                <th className={head}>{env.label.speed}</th>
                <th className={head}>{env.label.firstGood}</th>
                <th className={head}>{env.label.stays}</th>
                <th className={head}>{env.label.quality(maxSteps)}</th>
                <th className={head}>{env.label.median}</th>
                {!flag && <th className={head}>{env.success === "score" ? "mean |x| (0 = pad)" : "mean |cart position|"}</th>}
                {!flag && <th className={head}>{env.angleColumn}</th>}
                {env.success === "score" && <th className={head}>how the 20 ended</th>}
                <th className={head}>action switches</th>
              </tr>
            </thead>
            <tbody>
              {LADDER.map((l) => {
                const a = by[l.key];
                const ss = a?.scorecard?.speed_stability;
                const u = a ? updates(a.settings.config) : null;
                return (
                  <Fragment key={l.key}>
                  <tr
                    key={l.key}
                    onClick={() => a && onSelect(l.key)}
                    className={`border-t border-neutral-100 ${a ? "cursor-pointer hover:bg-neutral-50" : "text-neutral-400"} ${selected === l.key ? "bg-neutral-50" : ""}`}
                  >
                    <td {...nameCell(l.key)}>
                      {chevron(l.key)}
                      <Swatch color={l.color} />
                      <span className={selected === l.key ? "font-medium" : ""}>{l.tab}</span>
                    </td>
                    <td className={cell}>{u ? (u.n ?? "–") : "–"}</td>
                    {!a ? (
                      <td className="px-2 py-1.5 text-left text-[11.5px]" colSpan={8}>
                        not run yet: <span className="font-mono">--algo {l.key}</span>
                      </td>
                    ) : !ss ? (
                      <td className="px-2 py-1.5 text-left text-[11.5px] text-neutral-500" colSpan={8}>
                        {a.status.state === "testing"
                          ? "testing the final policy…"
                          : a.status.state === "running"
                            ? `training… iteration ${a.status.iter} · ${num(a.status.env_steps)} steps`
                            : "no scorecard (stopped early)"}
                      </td>
                    ) : (
                      <>
                        <td className={cell}>
                          <B on={ss.eval500_steps === bestSpeed}>{num(ss.eval500_steps)}</B>
                          {ss.eval500_iter != null && <span className="text-neutral-400"> (iter {ss.eval500_iter})</span>}
                        </td>
                        <td className={cell}>
                          <B on={ss.batch475_steps === bestBatch}>{num(ss.batch475_steps)}</B>
                        </td>
                        <td className={cell}>
                          <B on={ss.stay_475 === bestStay}>{ss.stay_475 == null ? "–" : `${Math.round(ss.stay_475 * 100)}%`}</B>
                          {ss.drops_below_250 > 0 && <span className="text-neutral-400"> · {ss.drops_below_250} drops</span>}
                        </td>
                        {qualityCells(a.scorecard!.quality.learned, true)}
                      </>
                    )}
                  </tr>
                  {note(
                    l.key,
                    <>
                      <b className="font-medium text-neutral-900">{l.name}.</b> {l.why}
                    </>,
                  )}
                  </Fragment>
                );
              })}
              {good && (
                <tr className="cursor-pointer border-t border-neutral-100 hover:bg-neutral-50">
                  <td {...nameCell("good")}>
                    {chevron("good")}
                    <Swatch color={GOOD_COLOR} dashed />
                    good policy (hand-written)
                  </td>
                  <td className={cell}>–</td>
                  <td className={cell}>–</td>
                  <td className={cell}>–</td>
                  <td className={cell}>–</td>
                  {qualityCells(good, false)}
                </tr>
              )}
              {good &&
                note(
                  "good",
                  <>
                    <b className="font-medium text-neutral-900">Good policy.</b> Not learned: a hand-written rule ({env.goodPolicy}). It is tested
                    exactly like the final learned policies, so it shows what a good policy achieves here, a reference for the rows above.
                  </>,
                )}
            </tbody>
          </table>
        </div>

        {have.length > 0 && (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            <div>
              <div className="flex items-center gap-2 text-[11.5px] font-medium text-neutral-700">
                Speed: {env.success === "score" ? "score" : "episode length"} vs environment steps
                <span className="ml-auto flex rounded bg-neutral-100 p-0.5 text-[11px] font-normal">
                  {(
                    [
                      ["ret", "batch"],
                      ["eval", "greedy"],
                    ] as const
                  ).map(([k, label]) => (
                    <button key={k} onClick={() => setSpeedY(k)} className={`rounded px-1.5 ${speedY === k ? "bg-white shadow-sm" : "text-neutral-500"}`}>
                      {label}
                    </button>
                  ))}
                </span>
              </div>
              <UChart
                x={speed.x}
                ys={speed.ys}
                series={speed.series}
                height={180}
                yRange={env.success === "score" ? [-450, 320] : [0, env.lengthMax * 1.04]}
                xLabel="environment steps"
              />
              <div className="text-[11px] text-neutral-400">
                {speedY === "ret"
                  ? `batch: mean ${env.success === "score" ? "score" : "length"} of the episodes played while learning (sampled actions)`
                  : "greedy: most likely action, 10 fixed starts, every 10 iterations"}
              </div>
            </div>
            <div>
              <div className="text-[11.5px] font-medium text-neutral-700">Update size: how much one batch changes the policy</div>
              <UChart x={kl.x} ys={kl.ys} series={kl.series} height={180} yLog xLabel="environment steps" />
              <div className="text-[11px] text-neutral-400">KL(π_old ‖ π_new) on the batch states, log scale. Big = the update moved the policy far from the one that collected the data.</div>
            </div>
            <div>
              <div className="text-[11.5px] font-medium text-neutral-700">{env.label.curve}</div>
              {surv.ys.length ? (
                <UChart x={surv.x} ys={surv.ys} series={surv.series} height={180} yRange={[0, 1]} xLabel={env.success === "score" ? "score" : "step"} />
              ) : (
                <div className="py-10 text-center text-[12px] text-neutral-400">appears when a run finishes</div>
              )}
            </div>
          </div>
        )}

        {inner.n > 0 && (
          <div>
            <div className="mb-1 border-t border-neutral-100 pt-3 text-[12px] font-medium text-neutral-800">
              Inside one batch: how far the policy has moved from π_old after each epoch
              <span className="ml-2 text-[11px] font-normal text-neutral-500">
                average over the iterations that were still learning ({env.success === "flag" ? "not every episode reached the flag" : env.success === "score" ? "batch score < 200" : "batch mean < 475"}; afterwards A ≈ 0 and nothing moves) · iterations: {inner.note}
              </span>
            </div>
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
              <div>
                <div className="text-[11.5px] font-medium text-neutral-700">KL(π_old ‖ π) after each epoch</div>
                <UChart x={inner.kl.x} ys={inner.kl.ys} series={inner.kl.series} height={170} yLog xLabel="epoch" />
                <div className="text-[11px] text-neutral-400">How different the policy is from the one that played the batch. Log scale.</div>
              </div>
              <div>
                <div className="text-[11.5px] font-medium text-neutral-700">Ratio r = π(a|s) / π_old(a|s) of the steps that moved most</div>
                <UChart x={inner.r.x} ys={inner.r.ys} series={inner.r.series} bands={inner.r.bands} height={170} xLabel="epoch" />
                <div className="text-[11px] text-neutral-400">
                  Solid: 90th percentile of r on steps with A &gt; 0 (pushed up). Dashed: 10th percentile on steps with A &lt; 0 (pushed down). Green band = the clip range [0.8, 1.2].
                </div>
              </div>
              <div>
                <div className="text-[11.5px] font-medium text-neutral-700">Share of steps past the clip limit</div>
                <UChart x={inner.past.x} ys={inner.past.ys} series={inner.past.series} height={170} yRange={[0, null]} xLabel="epoch" />
                <div className="text-[11px] text-neutral-400">
                  A &gt; 0 and r &gt; 1.2, or A &lt; 0 and r &lt; 0.8: steps where clip stops the push (without clip: where it would).
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ episode replay
/** Play / pause / scrub through n steps. */
function usePlayback(n: number) {
  const [i, setI] = useState(0);
  const [want, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(50);
  const playing = want && i < n - 1;
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setI((k) => Math.min(n - 1, k + 1)), 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, n]);
  return { i, n, playing, setI, setPlaying, speed, setSpeed };
}

function Controls({ pb, children }: { pb: ReturnType<typeof usePlayback>; children?: ReactNode }) {
  const { i, n, playing, setI, setPlaying, speed, setSpeed } = pb;
  return (
    <div className="mt-2 flex items-center gap-2 text-[12px]">
      <button className="rounded border border-neutral-300 px-2 py-0.5" onClick={() => (i >= n - 1 ? (setI(0), setPlaying(true)) : setPlaying(!playing))}>
        {playing ? "❚❚" : "▶"}
      </button>
      <input type="range" min={0} max={n - 1} value={i} onChange={(e) => (setPlaying(false), setI(+e.target.value))} className="flex-1" />
      <span className="w-20 tabular-nums text-neutral-600">
        step {i + 1}/{n}
      </span>
      <select className="rounded border border-neutral-300 px-1 py-0.5" value={speed} onChange={(e) => setSpeed(+e.target.value)}>
        {[10, 25, 50, 100, 200].map((v) => (
          <option key={v} value={v}>
            {v}/s
          </option>
        ))}
      </select>
      {children}
    </div>
  );
}

const argmax = (p: number[]) => p.indexOf(Math.max(...p));

/** More recorded episodes than this: a dropdown instead of one button each. */
const MANY = 24;

// ------------------------------------------------------------------ side by side
type Side = AlgoKey | "good";
type IterPick = "final" | number;

function endText(ep: Episode, env: EnvCfg) {
  if (!ep.terminated) return `${ep.steps} steps: time limit reached, still balancing`;
  return env.endText(ep.states.at(-1)!, ep.steps);
}

interface SideView {
  label: string;
  sub: string;
  color: string;
  ep: Episode;
}

/** Two recorded episodes played together, one shared play / pause / slider. */
function DualPlayer({ a, b }: { a: SideView; b: SideView }) {
  const env = useEnv();
  const n = Math.max(a.ep.actions.length, b.ep.actions.length);
  const pb = usePlayback(n);
  const actions = env.actions;
  const { ep: epA, label: labelA, color: colorA } = a;
  const { ep: epB, label: labelB, color: colorB } = b;
  const trace = useMemo(() => {
    const x = Array.from({ length: n }, (_, k) => k + 1);
    // per side: the cart dashed, the first angle solid, a second angle dotted
    const dashes: (number[] | undefined)[] = [[5, 3], undefined, [1.5, 2.5]];
    const sides = [
      { ep: epA, label: labelA, color: colorA },
      { ep: epB, label: labelB, color: colorB },
    ];
    const ys: (number | null)[][] = [];
    const series: SeriesSpec[] = [];
    env.traces.forEach((t, ti) =>
      sides.forEach((sd) => {
        ys.push(x.map((_, k) => (k < sd.ep.states.length ? t.f(sd.ep.states[k]) : null)));
        series.push({ label: `${sd.label}: ${t.label}`, color: sd.color, width: ti === 0 ? 1.25 : 1.75, dash: dashes[ti] });
      }),
    );
    return { x, ys, series };
  }, [epA, epB, labelA, labelB, colorA, colorB, n, env]);
  // training episodes: share of steps where the sampled action was NOT the most likely one (exploration)
  const explored = useMemo(
    () => [epA, epB].map((ep) => (ep.kind === "training" ? ep.actions.filter((a, k) => a !== argmax(ep.probs[k])).length / ep.actions.length : null)),
    [epA, epB],
  );
  return (
    <div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {[a, b].map((sd, k) => {
          const len = sd.ep.actions.length;
          const j = Math.min(pb.i, len - 1);
          const s = sd.ep.states[j];
          const step = { s, a: sd.ep.actions[j], p: sd.ep.probs[j] } as unknown as StepRec;
          const atEnd = pb.i >= len - 1;
          // a badge when the episode is over: CartPole-like only for a failure; MountainCar for both outcomes
          const badge = !atEnd ? null : env.success === "score"
            ? { ok: (sd.ep.score ?? -1e9) >= 200, text: `${sd.ep.outcome ?? "ended"} · score ${Math.round(sd.ep.score ?? 0)}` }
            : env.success === "flag"
            ? sd.ep.terminated
              ? { ok: true, text: endText(sd.ep, env) }
              : { ok: false, text: `never reached the flag (${len} steps)` }
            : sd.ep.terminated
              ? { ok: false, text: endText(sd.ep, env) }
              : null;
          return (
            <div key={k}>
              <div className="mb-1 flex flex-wrap items-baseline gap-x-2 text-[12.5px]">
                <Swatch color={sd.color} dashed={sd.color === GOOD_COLOR} />
                <b className="font-medium text-neutral-900">{sd.label}</b>
                <span className="text-[11.5px] text-neutral-500">{sd.sub}</span>
              </div>
              <div className="relative">
                <EnvView envId={env.envId} step={step} trail={[]} actionNames={actions} />
                {badge && (
                  <div
                    className={`absolute left-1/2 top-2 -translate-x-1/2 rounded px-2 py-0.5 text-[11.5px] font-medium text-white ${badge.ok ? "bg-[#1baf7a]" : "bg-[#e34948]"}`}
                  >
                    {badge.text}
                  </div>
                )}
              </div>
              <div className="mt-1 text-[11.5px] tabular-nums text-neutral-600">
                step {j + 1}/{len} · {env.status(s)}{" "}
                {env.success === "limit" && !sd.ep.terminated && atEnd && <span className="text-[#0b7a52]">· reached {env.lengthMax}</span>}
                {sd.ep.kind === "training" && sd.ep.actions[j] !== argmax(sd.ep.probs[j]) && (
                  <span className="ml-1 rounded bg-[#fdf3dc] px-1 text-[10.5px] text-[#8a5a00]">explored</span>
                )}
              </div>
              {explored[k] != null && (
                <div className="text-[11px] text-neutral-500">
                  a real training episode: the sampled action was not the most likely one on {(explored[k]! * 100).toFixed(0)}% of the steps
                </div>
              )}
            </div>
          );
        })}
      </div>
      <Controls pb={pb} />
      <div className="mt-2 text-[11px] text-neutral-500">
        {env.traceNote}; {env.traces[0].label} dashed
      </div>
      <UChart x={trace.x} ys={trace.ys} series={trace.series} height={140} yRange={[-1, 1]} markers={[{ x: pb.i + 1, label: "", color: C.ink2 }]} xLabel="step" />
    </div>
  );
}

function SideBySide({ algos, selected }: { algos: AlgoSummary[]; selected: AlgoKey }) {
  const env = useEnv();
  const have = LADDER.filter((l) => algos.some((a) => a.algo === l.key)).map((l) => l.key);
  const by = Object.fromEntries(algos.map((a) => [a.algo, a])) as Partial<Record<AlgoKey, AlgoSummary>>;
  const [leftPick, setLeft] = useState<Side | null>(null);
  const [rightPick, setRight] = useState<Side | null>(null);
  const [pick, setPick] = useState<IterPick>("final");
  // match the two sides by environment steps (the same experience) or by iteration number
  const [match, setMatch] = useState<"steps" | "iter">("steps");
  const [mode, setMode] = useState<Mode>("greedy");
  const [stepPick, setStepPick] = useState<IterPick>("final");
  const [lists, setLists] = useState<Record<string, EpisodeInfo[]>>({});
  const [eps, setEps] = useState<Record<string, Episode>>({});

  // defaults: batch actor-critic on the left, the algorithm selected below on the right
  const left: Side = leftPick ?? (have.includes("batch_ac") ? "batch_ac" : (have[0] ?? "good"));
  const right: Side = rightPick ?? (selected !== left && have.includes(selected) ? selected : (have.find((k) => k !== left) ?? "good"));
  const source = (s: Side): AlgoKey | undefined => (s === "good" ? have[0] : s); // the good policy is saved with every run
  const ver = (k: AlgoKey | undefined) => (k ? `${k}|${by[k]?.status.started}|${by[k]?.status.iter}` : "");
  const srcL = source(left);
  const srcR = source(right);
  const verL = ver(srcL);
  const verR = ver(srcR);

  // the list of recorded episodes of each side; refetched when that run moves on
  useEffect(() => {
    let alive = true;
    for (const [k, v] of [
      [srcL, verL],
      [srcR, verR],
    ] as const) {
      if (!k || lists[v]) continue;
      fetch(cliUrl(`run/${k}/episodes`, env.key))
        .then((r) => (r.ok ? r.json() : null))
        .then((d: EpisodeInfo[] | null) => alive && d && setLists((prev) => ({ ...prev, [v]: d })))
        .catch(() => {});
    }
    return () => {
      alive = false;
    };
  }, [srcL, srcR, verL, verR, lists, env]);

  const learned = (s: Side) => (s === "good" ? [] : (lists[ver(s)] ?? []).filter((e) => ofKind(e, mode)));
  const stepMaps = useMemo(
    () => Object.fromEntries(algos.map((a) => [a.algo, new Map(a.speed.map((pt) => [pt.iter, pt.env_steps]))])) as Partial<Record<AlgoKey, Map<number, number>>>,
    [algos],
  );
  const stepsAt = (s: AlgoKey, it: number) => stepMaps[s]?.get(it);
  const resolve = (s: Side): EpisodeInfo | null | undefined => {
    const list = lists[ver(source(s))];
    if (!list) return undefined; // still loading
    if (s === "good") return list.find((e) => e.name === "good") ?? null;
    const l = learned(s);
    const which = match === "steps" ? stepPick : pick;
    if (which === "final") return l.at(-1) ?? null;
    if (match === "iter") return l.find((e) => e.iter === pick) ?? null;
    // the recording closest to the same number of environment steps
    let best: EpisodeInfo | null = null;
    for (const e of l) {
      const st = stepsAt(s, e.iter);
      if (st == null) continue;
      if (!best || Math.abs(st - which) < Math.abs((stepsAt(s, best.iter) ?? Infinity) - which)) best = e;
    }
    return best;
  };
  const infoL = resolve(left);
  const infoR = resolve(right);
  const keyL = infoL && srcL ? `${verL}|${infoL.name}` : "";
  const keyR = infoR && srcR ? `${verR}|${infoR.name}` : "";

  // the two episodes themselves (cached)
  useEffect(() => {
    let alive = true;
    for (const [k, src, info] of [
      [keyL, srcL, infoL],
      [keyR, srcR, infoR],
    ] as const) {
      if (!k || !src || !info || eps[k]) continue;
      fetch(cliUrl(`run/${src}/episode/${info.name}`, env.key))
        .then((r) => (r.ok ? r.json() : null))
        .then((e: Episode | null) => alive && e && setEps((prev) => ({ ...prev, [k]: e })))
        .catch(() => {});
    }
    return () => {
      alive = false;
    };
  }, [keyL, keyR, srcL, srcR, infoL, infoR, eps, env]);

  // iterations recorded in either run (a run's own last iteration is under "final")
  const sets = [left, right].filter((s) => s !== "good").map((s) => learned(s).map((e) => e.iter));
  const finals = sets.map((s) => s.at(-1));
  const iters = Array.from(new Set(sets.flat()))
    .sort((x, y) => x - y)
    .filter((it) => sets.every((s) => s.includes(it)) || !finals.includes(it));

  const view = (s: Side, info: EpisodeInfo, ep: Episode): SideView => {
    if (s === "good") return { label: "good policy", sub: `hand-written: ${env.goodPolicy}`, color: GOOD_COLOR, ep };
    const steps = by[s]?.speed.find((p) => p.iter === info.iter)?.env_steps;
    const isFinal = learned(s).at(-1)?.name === info.name && !isLive(by[s]?.status);
    return {
      label: LADDER_BY[s].tab,
      sub: `${mode === "training" ? "training episode (own start), " : isFinal ? "final, " : ""}iteration ${info.iter}${steps != null ? ` · trained on ${num(steps)} env steps` : ""}`,
      color: LADDER_BY[s].color,
      ep,
    };
  };
  const epL = eps[keyL];
  const epR = eps[keyR];
  const missing = (s: Side, info: EpisodeInfo | null | undefined) =>
    info === null && s !== "good"
      ? match === "iter"
        ? `${LADDER_BY[s].tab}: no episode recorded at iteration ${pick} (its run has ${learned(s).at(-1)?.iter ?? 0} iterations)`
        : mode === "training"
          ? `${LADDER_BY[s].tab}: no training episodes recorded in this run (re-run it to get them)`
          : `${LADDER_BY[s].tab}: no recorded episode yet`
      : null;

  // environment-step marks: about 10 even points up to the longer run, the nearest 1/2/5 × 10ⁿ apart
  const lastSteps = [left, right]
    .filter((s): s is AlgoKey => s !== "good")
    .map((s) => {
      const last = learned(s).at(-1);
      return last ? (stepsAt(s, last.iter) ?? 0) : 0;
    });
  const maxSteps = Math.max(0, ...lastSteps);
  const raw = maxSteps / 10;
  const mag = raw > 0 ? 10 ** Math.floor(Math.log10(raw)) : 1;
  const unit = [1, 2, 5, 10].map((m) => m * mag).reduce((u, v) => (Math.abs(Math.log(v / raw)) < Math.abs(Math.log(u / raw)) ? v : u));
  const marks = raw > 0 ? Array.from({ length: Math.floor(maxSteps / unit) }, (_, k) => (k + 1) * unit) : [];
  const kfmt = (v: number) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(v));

  const select = (value: Side, set: (s: Side) => void) => (
    <select className="rounded border border-neutral-300 bg-white px-1.5 py-0.5 text-[12px]" value={value} onChange={(e) => set(e.target.value as Side)}>
      {have.map((k) => (
        <option key={k} value={k}>
          {LADDER_BY[k].tab}
        </option>
      ))}
      <option value="good">good policy (hand-written)</option>
    </select>
  );

  return (
    <Panel
      title={mode === "greedy" ? "Side by side: two policies, same start" : "Side by side: two policies, real training episodes"}
      help="Most likely action: every recording starts from the same state and always takes the most likely action, so two policies can be compared directly. As in training: a real episode from that iteration's batch, with the actions as they were sampled (so sometimes not the most likely one: that is exploration); each starts where that episode started. Pick what goes left and right. Same env steps: each side plays its recording closest to the same amount of experience (fair between algorithms, whose iterations differ: a PPO iteration is 2,048 steps, a collapsed run's iteration can be 40). Same iteration: the same iteration number on both sides. Each side shows its iteration and how many environment steps it had trained on."
    >
      <div className="mb-3 flex flex-wrap items-center gap-2 text-[12px]">
        {select(left, setLeft)}
        <span className="text-neutral-400">vs</span>
        {select(right, setRight)}
        <ModeSwitch mode={mode} setMode={setMode} />
        <span className="ml-3 flex rounded bg-neutral-100 p-0.5 text-[11px]">
          {(
            [
              ["steps", "same env steps"],
              ["iter", "same iteration"],
            ] as const
          ).map(([k, label]) => (
            <button key={k} onClick={() => setMatch(k)} className={`rounded px-1.5 ${match === k ? "bg-white shadow-sm" : "text-neutral-500"}`}>
              {label}
            </button>
          ))}
        </span>
        {match === "steps" &&
          (["final", 0, ...marks] as IterPick[]).map((v) => (
            <button
              key={v}
              onClick={() => setStepPick(v)}
              className={`rounded border px-2 py-0.5 tabular-nums ${stepPick === v ? "border-neutral-900 bg-neutral-900 text-white" : "border-neutral-300 bg-white"}`}
            >
              {v === "final" ? "final" : v === 0 ? "start" : kfmt(v)}
            </button>
          ))}
        {match === "iter" && <span className="text-neutral-500">iteration</span>}
        {(match === "iter" ? (["final", ...(iters.length > MANY ? [] : iters)] as IterPick[]) : []).map((it) => (
          <button
            key={it}
            onClick={() => setPick(it)}
            className={`rounded border px-2 py-0.5 tabular-nums ${pick === it ? "border-neutral-900 bg-neutral-900 text-white" : "border-neutral-300 bg-white"}`}
          >
            {it}
          </button>
        ))}
        {match === "iter" && iters.length > MANY && (
          <select
            className="rounded border border-neutral-300 bg-white px-1.5 py-0.5 tabular-nums"
            value={pick === "final" ? "" : pick}
            onChange={(e) => setPick(e.target.value === "" ? "final" : +e.target.value)}
          >
            <option value="">pick an iteration…</option>
            {iters.map((it) => (
              <option key={it} value={it}>
                iteration {it}
              </option>
            ))}
          </select>
        )}
      </div>
      {missing(left, infoL) || missing(right, infoR) ? (
        <div className="py-8 text-center text-[12px] text-neutral-500">
          {[missing(left, infoL), missing(right, infoR)].filter(Boolean).map((m) => (
            <div key={m}>{m}</div>
          ))}
        </div>
      ) : epL && epR && infoL && infoR ? (
        <DualPlayer key={`${keyL}::${keyR}`} a={view(left, infoL, epL)} b={view(right, infoR, epR)} />
      ) : (
        <div className="py-8 text-center text-[12px] text-neutral-400">loading…</div>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ the selected algorithm
function UpdatePanel({ c }: { c: AlgoConfig }) {
  const env = useEnv();
  const nIn = env.state.filter((d) => !d.derived).length;
  const lam = c.gae_lambda ?? 1;
  const u = updates(c);
  return (
    <Panel
      title="Update per iteration"
      help="One iteration = collect a batch with the frozen policy, compute the advantages ONCE with the critic before any update, take the gradient steps, then throw the batch away (on-policy: no replay buffer)."
    >
      <Kv k="batch">
        {c.batch_steps ? (
          <>
            {c.batch_steps.toLocaleString()} steps with the frozen π_old (an episode cut at the end continues in the next batch; its future is filled in with V)
          </>
        ) : (
          <>{c.batch_episodes ?? 5} full episodes with the frozen π_old</>
        )}
      </Kv>
      <Kv k="advantage">
        {lam === 1 ? (
          <>
            <Mono>A = G − V(s)</Mono>
            <div className="text-[11px] text-neutral-500">G = discounted return that actually followed</div>
          </>
        ) : (
          <>
            <Mono>δ = r + γV(s′) − V(s)</Mono>
            <br />
            <Mono>A = Σ (γλ)ᵏ δ</Mono>
            <div className="text-[11px] text-neutral-500">λ = {lam}: between one-step TD (λ = 0) and G − V (λ = 1)</div>
          </>
        )}
      </Kv>
      <Kv k="critic">
        {lam === 1 ? (
          <Mono>L_V = ½(G − V)²</Mono>
        ) : (
          <>
            <Mono>target = A + V_old(s)</Mono>
            <br />
            <Mono>L_V = ½(target − V)²</Mono>
          </>
        )}
      </Kv>
      <Kv k="actor">
        {c.clip ? (
          <>
            <Mono>r = π(a|s) / π_old(a|s)</Mono>
            <br />
            <Mono>
              L_π = −min(r·A, clip(r,{" "}
              <span className="whitespace-nowrap">
                {1 - c.clip}, {1 + c.clip})·A)
              </span>
            </Mono>
          </>
        ) : c.ratio ? (
          <>
            <Mono>r = π(a|s) / π_old(a|s)</Mono>
            <br />
            <Mono>L_π = −r·A</Mono>
            <div className="text-[11px] text-neutral-500">no limit: for A &gt; 0 it pushes harder the larger r gets</div>
          </>
        ) : (
          <>
            <Mono>L_π = −A·log π(a|s)</Mono>
            {(c.epochs ?? 1) > 1 && <div className="text-[11px] text-neutral-500">every epoch, as if the current policy had played the batch</div>}
          </>
        )}
      </Kv>
      {c.adv_norm && <Kv k="normalise">A → mean 0, std 1 in each minibatch</Kv>}
      <Kv k="steps">
        {u.text}
        {u.n != null && u.n > 1 && <span className="text-neutral-500"> = {u.n} gradient steps</span>} (critic and actor each)
      </Kv>
      <Kv k="networks">
        actor {nIn}→{c.hidden}→{env.actions.length} · critic {nIn}→{c.hidden}→1 · {c.optimizer ?? "adam"} · lr {c.actor_lr} / {c.critic_lr}
      </Kv>
      <div className={`pt-1.5 text-right ${STATIC ? "hidden" : ""}`}>
        <Link href="/learn" className="text-[12px] text-[#2a78d6] hover:underline">
          step through one batch actor-critic update →
        </Link>
      </div>
    </Panel>
  );
}

const TABLE: [string, string, number][] = [
  ["iter", "iter", 0],
  ["env_steps", "env steps", 0],
  ["ret", "length", 1],
  ["eval", "eval", 0],
  ["p_good", "P(good)", 2],
  ["H", "entropy", 2],
  ["KL", "KL", 4],
  ["clip_frac", "moved > 20%", 2],
  ["Lpi", "actor loss", 2],
  ["Lv", "critic loss", 1],
  ["rel", "critic rel. err", 2],
  ["true_err", "critic true err", 2],
  ["noise", "noise floor", 2],
];

function Logs({ rows }: { rows: Row[] }) {
  return (
    <div className="max-h-80 overflow-auto">
      <table className="w-full text-right text-[12px] tabular-nums">
        <thead className="sticky top-0 bg-white text-[11px] text-neutral-500">
          <tr>
            {TABLE.map(([k, label]) => (
              <th key={k} className="px-2 py-1 font-normal">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.iter as number} className="border-t border-neutral-100">
              {TABLE.map(([k, , d]) => (
                <td key={k} className="px-2 py-0.5">
                  {r[k] == null ? "" : (r[k] as number).toFixed(d)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The epochs inside ONE iteration: how the policy moves away from π_old, step by step. */
function InsideIteration({ data }: { data: RunData }) {
  const per = data.per_epoch;
  const rowsByIter = useMemo(() => {
    const m = new Map<number, EpochRow[]>();
    per.forEach((e) => m.set(e.iter, [...(m.get(e.iter) ?? []), e]));
    return m;
  }, [per]);
  const iters = useMemo(() => Array.from(rowsByIter.keys()).sort((a, b) => a - b), [rowsByIter]);
  // the iteration whose update moved the policy furthest (last epoch's KL), a good one to look at first
  const biggest = useMemo(() => {
    let best = iters[0];
    for (const it of iters) if ((rowsByIter.get(it)!.at(-1)!.kl ?? 0) > (rowsByIter.get(best)!.at(-1)!.kl ?? 0)) best = it;
    return best;
  }, [iters, rowsByIter]);
  const [picked, setPicked] = useState<number | null>(null);
  const it = picked != null && rowsByIter.has(picked) ? picked : biggest;
  const rows = useMemo(() => rowsByIter.get(it) ?? [], [rowsByIter, it]);
  const logRow = data.rows.find((r) => r.iter === it);
  const one = rows.length < 3; // batch actor-critic: a single gradient step
  const chart = useMemo(() => {
    const x = rows.map((r) => r.epoch);
    return {
      x,
      r: {
        ys: [
          rows.map((r) => r.r_hi ?? null),
          rows.map((r) => r.r_lo ?? null),
          rows.map((r) => r.r_max),
          rows.map((r) => r.r_min),
          rows.map((r) => r.r_pos),
          rows.map((r) => r.r_neg),
          x.map(() => 1.2),
          x.map(() => 0.8),
        ],
        series: [
          { label: "A>0: 90th pct", color: C.actor, width: 2, points: one },
          { label: "A<0: 10th pct", color: C.neg, width: 2, points: one },
          { label: "max r", color: C.actor, width: 1, dash: [3, 3], points: one },
          { label: "min r", color: C.neg, width: 1, dash: [3, 3], points: one },
          { label: "A>0: mean", color: C.actor, width: 1, dash: [1, 3], points: one },
          { label: "A<0: mean", color: C.neg, width: 1, dash: [1, 3], points: one },
          { label: "clip 1.2", color: C.pos, hidden: true },
          { label: "clip 0.8", color: C.pos, hidden: true },
        ] as SeriesSpec[],
      },
      kl: { ys: [rows.map((r) => (r.kl > 0 ? r.kl : null))], series: [{ label: "KL(π_old ‖ π)", color: C.ink, width: 2, points: one }] as SeriesSpec[] },
      past: { ys: [rows.map((r) => r.past)], series: [{ label: "past clip limit", color: C.marker2, width: 2, points: one }] as SeriesSpec[] },
    };
  }, [rows, one]);
  if (!iters.length) return null;
  const k = iters.indexOf(it);
  return (
    <Panel
      title={`Inside one iteration: the ${one ? "gradient step" : `${rows.length} epochs`} on one batch`}
      help="The batch is played once by π_old. Then every epoch takes gradient steps on that same batch. After each epoch we measure, on all its states, how far the current policy is from π_old: the ratio r = π(a|s)/π_old(a|s) of the action that was taken (1 = unchanged), the KL, and the share of steps past the clip limit (A > 0 and r > 1.2, or A < 0 and r < 0.8). Without clip nothing stops r; with clip, steps past the limit stop pushing."
    >
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[12px]">
        <span className="text-neutral-500">iteration</span>
        <button className="rounded border border-neutral-300 px-1.5" onClick={() => setPicked(iters[Math.max(0, k - 1)])}>
          ‹
        </button>
        <input type="range" min={0} max={iters.length - 1} value={k} onChange={(e) => setPicked(iters[+e.target.value])} className="w-64" />
        <button className="rounded border border-neutral-300 px-1.5" onClick={() => setPicked(iters[Math.min(iters.length - 1, k + 1)])}>
          ›
        </button>
        <b className="w-10 tabular-nums">{it}</b>
        {logRow && (
          <span className="tabular-nums text-neutral-500">
            batch mean length {Number(logRow.ret).toFixed(1)} · mean |A| {Number(logRow.absA).toFixed(2)}
          </span>
        )}
        <button
          className={`ml-auto rounded border px-2 py-0.5 ${it === biggest ? "border-neutral-900 bg-neutral-900 text-white" : "border-neutral-300"}`}
          onClick={() => setPicked(biggest)}
        >
          biggest move (iteration {biggest})
        </button>
      </div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[1.4fr_1fr_1fr]">
        <div>
          <div className="text-[11.5px] font-medium text-neutral-700">Ratio r = π(a|s) / π_old(a|s) after each epoch</div>
          <UChart x={chart.x} ys={chart.r.ys} series={chart.r.series} bands={[{ hi: 6, lo: 7, fill: "rgba(27,175,122,0.10)" }]} height={180} xLabel="epoch" />
          <div className="text-[11px] text-neutral-400">
            Solid: the 10 % of steps pushed furthest (90th percentile of r where A &gt; 0, 10th where A &lt; 0). Dashed: the single most extreme step. Dotted: the mean, which stays near 1 because similar states with different actions pull against each other. Green band: the clip range [0.8, 1.2].
          </div>
        </div>
        <div>
          <div className="text-[11.5px] font-medium text-neutral-700">KL(π_old ‖ π) after each epoch</div>
          <UChart x={chart.x} ys={chart.kl.ys} series={chart.kl.series} height={180} yLog xLabel="epoch" />
          <div className="text-[11px] text-neutral-400">How different the whole policy is from π_old, on the batch states. Log scale.</div>
        </div>
        <div>
          <div className="text-[11.5px] font-medium text-neutral-700">Share of steps past the clip limit</div>
          <UChart x={chart.x} ys={chart.past.ys} series={chart.past.series} height={180} yRange={[0, null]} xLabel="epoch" />
          <div className="text-[11px] text-neutral-400">A &gt; 0 and r &gt; 1.2, or A &lt; 0 and r &lt; 0.8.</div>
        </div>
      </div>
    </Panel>
  );
}

function AlgoDetail({ data }: { data: RunData }) {
  const env = useEnv();
  const rows = data.rows;
  return (
    <>
      <div className="grid grid-cols-1 items-start gap-3 lg:grid-cols-2">
          <UpdatePanel c={data.settings.config} />
          <Panel title="Environment" help="What the agent observes and does. The state is fully observed; each number is divided by its range before going into the networks.">
            {env.state
              .filter((d) => !d.derived)
              .map((d) => (
              <Kv key={d.name} k={d.name.replace("pole angular velocity", "ang. velocity")}>
                {d.unit} <span className="text-neutral-400">{d.range}</span>
              </Kv>
            ))}
            <Kv k="actions">
              {env.actions.map((a, k) => (
                <span key={a}>
                  {k > 0 && " · "}
                  <span style={{ color: ACTION_COLORS[a] }}>
                    {ACTION_GLYPH[a]} {a === "none" ? "no push" : a}
                  </span>
                </span>
              ))}
            </Kv>
            <Kv k="reward">{env.reward}</Kv>
            <Kv k="ends">{env.ends}</Kv>
          </Panel>
      </div>

      {rows.length > 0 && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          <Chart
            title={env.success === "score" ? "Agent: score" : "Agent: episode length"}
            help={
              env.success === "score"
                ? "Batch: mean score of the episodes that ended in this batch (sampled actions). Eval: the most likely action, 10 fixed starts. 200 = solved; random pushes score about −210."
                : `Batch: mean length of the episodes that ended in this batch (sampled actions). Eval: the most likely action, 10 fixed starts. ${env.success === "flag" ? "Steps to the flag: lower is better, 200 = never reached." : "500 = perfect."}`
            }
            rows={rows}
            series={
              env.success === "score"
                ? [
                    { key: "score", label: "batch", color: C.ink },
                    { key: "eval_score", label: "eval", color: C.pos },
                  ]
                : [
                    { key: "ret", label: "batch", color: C.ink },
                    { key: "eval", label: "eval", color: C.pos },
                  ]
            }
          />
          {env.success === "flag" && (
            <Chart
              title="Agent: how far right does it get?"
              help="The highest position each episode (or episode piece, for PPO) reached, averaged over the batch. The flag is at x = 0.5; random pushes reach about −0.37. It shows progress before any episode reaches the flag."
              rows={rows}
              series={[{ key: "max_x", label: "highest x", color: C.ink }]}
              range={[-1.2, 0.6]}
            />
          )}
          <Chart
            title="Actor: is the policy right?"
            help={`P(good action): probability given to the good policy's action (${env.goodPolicy}); ${(1 / env.actions.length).toFixed(2)} = random, 1 = perfect. Agreement: how often the most likely action is the good one. Entropy: ${Math.log(env.actions.length).toFixed(2)} = all actions equally likely, 0 = always the same action.`}
            rows={rows}
            series={[
              { key: "p_good", label: "P(good)", color: C.actor },
              { key: "agree", label: "agree", color: C.actor, dash: [5, 3] },
              { key: "H", label: "entropy", color: C.muted },
            ]}
            range={[0, Math.max(1, Math.log(env.actions.length))]}
          />
          <Chart
            title="Actor: how far did this update move the policy?"
            help="KL(π_old ‖ π_new): the difference between the policy that collected the batch and the policy after all the gradient steps on it (log scale). Moved > 20%: share of the batch steps where π(a|s) changed by more than ±20 %, the limit the clip sets. More gradient steps per batch move the policy further; the clip holds it back."
            rows={rows}
            series={[
              { key: "KL", label: "KL", color: C.actor },
              { key: "clip_frac", label: "moved > 20%", color: C.marker2, dash: [5, 3] },
            ]}
            log
          />
          <Chart
            title="Critic: loss ½(target − V)²"
            help="The raw loss on each new batch, before the update. It grows when episodes get longer (bigger returns), so it does not show by itself whether the critic learns."
            rows={rows}
            series={[{ key: "Lv", label: "critic loss", color: C.critic }]}
            log
          />
          <Chart
            title="Critic: error vs true value"
            help="Scale-free. True error: critic vs the real average return (10 states × 10 episodes). Noise floor: scatter no critic can remove. Falling true error = the critic is learning."
            rows={rows}
            series={[
              { key: "true_err", label: "true error", color: C.critic },
              { key: "noise", label: "noise floor", color: C.muted, dash: [5, 3] },
              { key: "rel", label: "batch error", color: C.critic, dash: [2, 3] },
            ]}
            range={[0, 1.2]}
          />
          <Chart
            title="Critic: does V rank the states right?"
            help="corr(V, target) on the new batch: 1 = V orders the states like the targets do. Explained variance: 1 = V predicts the target, 0 = no better than a constant."
            rows={rows}
            series={[
              { key: "corr", label: "corr", color: C.critic },
              { key: "EV", label: "explained var.", color: C.critic, dash: [5, 3] },
            ]}
            range={[-0.2, 1]}
          />
        </div>
      )}

      {data.per_epoch?.length > 0 && <InsideIteration key={data.algo} data={data} />}

      {rows.length > 0 && (
        <Panel title="Log" help={`The numbers the terminal printed (runs/cli/${data.algo}/log.csv).`}>
          <Logs rows={rows} />
        </Panel>
      )}
    </>
  );
}

// ------------------------------------------------------------------ page
const isLive = (s?: Status) => s?.state === "running" || s?.state === "testing";

export default function RunPage({ envKey = "cartpole" }: { envKey?: EnvCfg["key"] }) {
  const env = ENVS[envKey];
  const [algos, setAlgos] = useState<AlgoSummary[] | null>(null);
  const [run, setRun] = useState<RunData | null>(null);
  const [picked, setPicked] = useState<AlgoKey | null>(null);
  const [err, setErr] = useState("");
  const [copied, setCopied] = useState("");

  // the list of runs, every 3 s (it is small); picks up new runs while `--algo all` works through the ladder
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const load = async () => {
      try {
        const res = await fetch(cliUrl("algos", env.key));
        if (!res.ok) throw new Error(res.statusText);
        const d: AlgoSummary[] = await res.json();
        if (!alive) return;
        setAlgos(d);
        setErr("");
      } catch (e) {
        if (alive) setErr(STATIC ? "Could not load the saved runs." : e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Backend not reachable on :8000.");
      }
      if (alive && !STATIC) timer = window.setTimeout(load, 3000); // the hosted copy never changes
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [env]);

  // default: the algorithm training right now, else the highest rung of the ladder that has a run
  const fallback = algos?.find((a) => isLive(a.status))?.algo ?? [...LADDER].reverse().find((l) => algos?.some((a) => a.algo === l.key))?.key ?? null;
  const algo: AlgoKey = picked ?? fallback ?? "batch_ac";
  const summary = algos?.find((a) => a.algo === algo);
  const version = summary ? `${summary.status.state}|${summary.status.started}|${summary.status.iter}` : "";

  // the selected run in full; refetched whenever its status in the list changes
  useEffect(() => {
    if (!version) return;
    let alive = true;
    fetch(cliUrl(`run/${algo}`, env.key))
      .then((r) => (r.ok ? r.json() : null))
      .then((d: RunData | null) => alive && d && setRun(d))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [algo, version, env]);

  const data = run?.algo === algo ? run : null;
  const st: Status = data?.status ?? summary?.status ?? {};
  const cfg = data?.settings.config ?? summary?.settings.config;
  const L = LADDER_BY[algo];
  const commandOne = env.command(algo);
  const commandAll = `${env.command("all")} --render-every 0`;
  const copy = (text: string, key: string) => navigator.clipboard?.writeText(text).then(() => setCopied(key)).catch(() => {});
  const progress = st.budget && st.env_steps != null ? Math.min(1, st.env_steps / st.budget) : null;

  return (
    <EnvCtx.Provider value={env}>
    <LegendToggles>
    <div className="min-h-screen bg-[#fafaf9]">
      <NavBar
        active="run"
        right={
          <div className="flex items-center gap-1 rounded-md bg-neutral-100 p-0.5 text-[12px]">
            {Object.values(ENVS).map((e) => (
              <Link
                key={e.key}
                href={e.href}
                className={`rounded px-2.5 py-1 ${e.key === env.key ? "bg-white font-medium text-neutral-900 shadow-sm" : "text-neutral-500 hover:text-neutral-900"}`}
              >
                {e.tab}
              </Link>
            ))}
          </div>
        }
      />
      <main className="mx-auto max-w-[1400px] space-y-3 p-3">
        {/* the ladder */}
        <div className="rounded-lg border border-neutral-200 bg-white px-4 py-3">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h1 className="text-[18px] font-semibold text-neutral-900">{env.title}: from Batch Actor-Critic to PPO</h1>
            <span className="text-[13px] text-neutral-500">one change at a time · the same budget of environment steps for each · {env.envId}</span>
            <button
              className="ml-auto rounded border border-neutral-300 px-2 py-0.5 font-mono text-[11px] hover:bg-neutral-50"
              title={commandAll}
              onClick={() => copy(commandAll, "all")}
            >
              {copied === "all" ? "copied ✓" : `copy command: train all ${LADDER.length}`}
            </button>
          </div>
          <div className="mt-3 flex flex-wrap items-stretch gap-1.5">
            {LADDER.map((l, k) => {
              const a = algos?.find((x) => x.algo === l.key);
              const on = l.key === algo;
              const speed = a?.scorecard?.speed_stability.eval500_steps;
              return (
                <div key={l.key} className="flex items-center gap-1.5">
                  {k > 0 && <span className="text-neutral-300">→</span>}
                  <button
                    onClick={() => setPicked(l.key)}
                    className={`min-w-[112px] rounded-md border px-2.5 py-1.5 text-left ${on ? "border-neutral-900 bg-neutral-900 text-white" : "border-neutral-200 bg-white hover:border-neutral-400"}`}
                  >
                    <div className="flex items-center gap-1.5 text-[12.5px] font-medium">
                      <span className="h-2 w-2 rounded-full" style={{ background: l.color === C.ink && on ? "#fff" : l.color }} />
                      {l.tab}
                    </div>
                    <div className={`text-[11px] tabular-nums ${on ? "text-neutral-300" : "text-neutral-500"}`}>
                      {!a
                        ? "not run yet"
                        : isLive(a.status)
                          ? `${a.status.state}… ${num(a.status.env_steps)} steps`
                          : a.scorecard
                            ? speed != null
                              ? `${env.success === "flag" ? "solved" : env.success === "score" ? "200" : "500"} after ${num(speed)} steps`
                              : env.success === "score"
                                ? "never reached 200"
                                : env.success === "flag"
                                ? "never solved"
                                : "never reached 500"
                            : (a.status.state ?? "")}
                    </div>
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {err && <div className="rounded-lg border border-neutral-200 bg-white px-4 py-3 text-[12px] text-neutral-600">{err}</div>}

        {algos && algos.length > 0 && <Compare algos={algos} selected={algo} onSelect={setPicked} />}

        {algos && algos.length > 0 && <SideBySide algos={algos} selected={algo} />}

        {/* the selected algorithm */}
        <div className="rounded-lg border border-neutral-200 bg-white px-4 py-3">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="h-2.5 w-2.5 self-center rounded-full" style={{ background: L.color }} />
            <h2 className="text-[17px] font-semibold text-neutral-900">{L.name}</h2>
            {cfg && <span className="text-[12.5px] text-neutral-500">{tags(cfg).join(" · ")}</span>}
          </div>
          <div className="mt-1 text-[12.5px] text-neutral-600">{L.key === "batch_ac" ? L.adds : `adds: ${L.adds}`}</div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[12px] text-neutral-600">
            {summary ? (
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: isLive(st) ? C.pos : st.state === "finished" ? C.actor : C.marker2 }} />
                <b className="font-medium text-neutral-800">{st.state ?? "…"}</b>
                {st.iter != null && (
                  <>
                    · iteration {st.iter} · {num(st.env_steps)}
                    {st.budget ? ` / ${num(st.budget)}` : ""} environment steps · started {st.started}
                  </>
                )}
              </span>
            ) : (
              <span>
                not run yet: <span className="font-mono text-[11.5px]">{commandOne}</span>
              </span>
            )}
            {progress != null && isLive(st) && (
              <span className="h-1.5 w-28 overflow-hidden rounded bg-neutral-100">
                <span className="block h-full bg-[#1baf7a]" style={{ width: `${progress * 100}%` }} />
              </span>
            )}
            {cfg &&
              (["hidden", "actor_lr", "critic_lr", "gamma", "seed"] as const).map((k) => (
                <span key={k} className="rounded bg-neutral-100 px-1.5 py-0.5 tabular-nums">
                  {k} {String(cfg[k])}
                </span>
              ))}
            {cfg?.env_kwargs &&
              Object.entries(cfg.env_kwargs).map(([k, v]) => (
                <span key={k} className="rounded bg-[#eef5fd] px-1.5 py-0.5 tabular-nums text-[#1d4f8f]" title="environment setting">
                  {k.replace("_", " ")} {String(v)}
                </span>
              ))}
            <button
              className="ml-auto rounded border border-neutral-300 px-2 py-0.5 font-mono text-[11px] hover:bg-neutral-50"
              title={commandOne}
              onClick={() => copy(commandOne, "one")}
            >
              {copied === "one" ? "copied ✓" : `copy command: --algo ${algo}`}
            </button>
          </div>
        </div>

        {data ? (
          <AlgoDetail key={algo} data={data} />
        ) : (
          summary && <div className="py-10 text-center text-[12px] text-neutral-400">loading…</div>
        )}
      </main>
    </div>
    </LegendToggles>
    </EnvCtx.Provider>
  );
}
