"use client";

import Link from "next/link";
import { ReactNode, useEffect, useMemo, useState } from "react";
import { BACKEND } from "@/lib/api";
import { ACTION_COLORS, C } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { StepRec } from "@/lib/types";
import EnvView from "./EnvView";
import NavBar from "./NavBar";
import Panel from "./Panel";
import UChart, { Marker, SeriesSpec } from "./UChart";

// ------------------------------------------------------------------ data
type Row = Record<string, number | null>;
type AlgoKey = "batch_ac" | "epochs" | "clip" | "gae" | "ppo";

interface AlgoConfig {
  batch_episodes?: number | null;
  batch_steps?: number | null;
  epochs?: number;
  minibatch?: number | null;
  clip?: number | null;
  gae_lambda?: number;
  adv_norm?: boolean;
  hidden: number;
  actor_lr: number;
  critic_lr: number;
  gamma: number;
  optimizer?: string;
  seed: number;
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
  speed: { iter: number; env_steps: number; ret: number; eval: number | null; KL?: number | null; clip_frac?: number | null }[];
}
interface EpisodeInfo {
  name: string;
  iter: number;
  steps: number;
  terminated: boolean;
}
interface RunData {
  algo: AlgoKey;
  settings: Settings;
  status: Status;
  rows: Row[];
  episodes: EpisodeInfo[];
  scorecard: Scorecard | null;
}
interface Episode extends EpisodeInfo {
  states: number[][];
  actions: number[];
  probs: number[][];
}

/** From our batch actor-critic to PPO, one switch at a time (backend/rl/ppo.py). */
const LADDER: { key: AlgoKey; tab: string; name: string; adds: string; color: string }[] = [
  {
    key: "batch_ac",
    tab: "Batch AC",
    name: "Batch Actor-Critic",
    adds: "the start: 5 episodes → 1 gradient step → throw the batch away",
    color: C.muted,
  },
  {
    key: "epochs",
    tab: "+ epochs",
    name: "Batch Actor-Critic + 10 epochs",
    adds: "use every batch 10 times: 10 gradient steps instead of 1, nothing stops the policy from moving too far",
    color: "#6250d6",
  },
  {
    key: "clip",
    tab: "+ clip",
    name: "+ importance ratio and clip",
    adds: "r = π/π_old; once π(a|s) has moved more than ±20 % in the direction A pushes, that step stops pushing",
    color: "#eda100",
  },
  {
    key: "gae",
    tab: "+ GAE",
    name: "+ GAE (λ = 0.95)",
    adds: "advantage from the critic's one-step errors δ, weighted by (γλ)ᵏ, instead of the full return G − V",
    color: "#e87ba4",
  },
  {
    key: "ppo",
    tab: "PPO",
    name: "PPO",
    adds: "fixed 2,048-step batches, minibatches of 64, advantages normalised per minibatch, smaller learning rates",
    color: C.ink,
  },
];
const LADDER_BY = Object.fromEntries(LADDER.map((l) => [l.key, l])) as Record<AlgoKey, (typeof LADDER)[number]>;
const GOOD_COLOR = C.pos;

/** Everything that differs between environments. MountainCar plugs in here next. */
const ENVS = {
  "CartPole-v1": {
    ready: true,
    command: (algo: string) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py --algo ${algo}`,
    actions: ["left", "right"],
    state: [
      { name: "cart position", unit: "m", range: "±2.4 (end)", scale: 2.4 },
      { name: "cart velocity", unit: "m/s", range: "÷3.0", scale: 3.0 },
      { name: "pole angle", unit: "rad", range: "±0.21 = ±12° (end)", scale: 0.2095 },
      { name: "pole angular velocity", unit: "rad/s", range: "÷3.5", scale: 3.5 },
    ],
    reward: "+1 per step alive",
    ends: "terminated: pole > 12° or cart off track · truncated: 500 steps",
    goodPolicy: "angle + 0.5·angular velocity > 0 → right",
  },
  "MountainCar-v0": { ready: false },
} as const;
type EnvKey = keyof typeof ENVS;
const CARTPOLE = ENVS["CartPole-v1"];

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
    c.clip ? `clip ε ${c.clip}` : "no clip",
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
      const pts = by[l.key]!.speed.filter((p) => speedY === "ret" || p.eval != null);
      return { x: pts.map((p) => p.env_steps), y: pts.map((p) => p[speedY]) };
    });
    return { ...overlay(curves), series: have.map((l) => ({ label: l.tab, color: l.color, width: l.key === "ppo" ? 2 : 1.75 })) };
  }, [have, by, speedY]);

  const kl = useMemo(() => {
    const curves = have.map((l) => {
      const pts = by[l.key]!.speed.filter((p) => p.KL != null && p.KL > 0);
      return { x: pts.map((p) => p.env_steps), y: pts.map((p) => p.KL!) };
    });
    return { ...overlay(curves), series: have.map((l) => ({ label: l.tab, color: l.color, width: 1.5 })) };
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
  const bestMedian = best(cards.map((c) => c.quality.learned.median_len), true);

  const cell = "px-2 py-1.5 text-right tabular-nums";
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
        <B on={bold && q.median_len === bestMedian}>{Math.round(q.median_len).toLocaleString()}</B>
      </td>
      <td className={cell}>{q.mean_abs_x.toFixed(2)} m</td>
      <td className={cell}>{q.mean_abs_theta.toFixed(3)} rad</td>
      <td className={cell}>{(q.switch_rate * 100).toFixed(0)}%</td>
    </>
  );

  return (
    <Panel
      title="Compare the algorithms"
      help="Every algorithm gets the same budget of environment steps (200,000). Their iterations are NOT comparable (a PPO iteration is 2,048 steps, a batch actor-critic iteration is 5 episodes), so the x-axis here is environment steps. Speed: steps until the greedy policy first scores 500. Stability: after the batch first averages ≥ 475, how often it stays there. Quality: the FINAL policy, most likely action, tested for up to 10,000 steps from 20 start states (training stops at 500). Click a row to open that algorithm below."
    >
      <div className="space-y-3">
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead className="text-[11px] text-neutral-500">
              <tr>
                <th className="px-2 py-1 text-left font-normal">algorithm</th>
                <th className={head}>gradient steps per batch</th>
                <th className={head}>speed: steps to 500</th>
                <th className={head}>first batch ≥ 475</th>
                <th className={head}>stability: stays ≥ 475</th>
                <th className={head}>survives {maxSteps.toLocaleString()} steps</th>
                <th className={head}>median survival</th>
                <th className={head}>mean |cart position|</th>
                <th className={head}>mean |pole angle|</th>
                <th className={head}>action switches</th>
              </tr>
            </thead>
            <tbody>
              {LADDER.map((l) => {
                const a = by[l.key];
                const ss = a?.scorecard?.speed_stability;
                const u = a ? updates(a.settings.config) : null;
                return (
                  <tr
                    key={l.key}
                    onClick={() => a && onSelect(l.key)}
                    className={`border-t border-neutral-100 ${a ? "cursor-pointer hover:bg-neutral-50" : "text-neutral-400"} ${selected === l.key ? "bg-neutral-50" : ""}`}
                  >
                    <td className="whitespace-nowrap px-2 py-1.5 text-left">
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
                );
              })}
              {good && (
                <tr className="border-t border-neutral-100">
                  <td className="whitespace-nowrap px-2 py-1.5 text-left">
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
            </tbody>
          </table>
        </div>

        {have.length > 0 && (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            <div>
              <div className="flex items-center gap-2 text-[11.5px] font-medium text-neutral-700">
                Speed: episode length vs environment steps
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
              <UChart x={speed.x} ys={speed.ys} series={speed.series} height={180} yRange={[0, 520]} xLabel="environment steps" />
              <div className="text-[11px] text-neutral-400">
                {speedY === "ret" ? "batch: mean length of the episodes played while learning (sampled actions)" : "greedy: most likely action, 10 fixed starts, every 10 iterations"}
              </div>
            </div>
            <div>
              <div className="text-[11.5px] font-medium text-neutral-700">Update size: how much one batch changes the policy</div>
              <UChart x={kl.x} ys={kl.ys} series={kl.series} height={180} yLog xLabel="environment steps" />
              <div className="text-[11px] text-neutral-400">KL(π_old ‖ π_new) on the batch states, log scale. Big = the update moved the policy far from the one that collected the data.</div>
            </div>
            <div>
              <div className="text-[11.5px] font-medium text-neutral-700">Quality: share of the 20 starts still balancing (final policy, long test)</div>
              {surv.ys.length ? (
                <UChart x={surv.x} ys={surv.ys} series={surv.series} height={180} yRange={[0, 1]} xLabel="step" />
              ) : (
                <div className="py-10 text-center text-[12px] text-neutral-400">appears when a run finishes</div>
              )}
            </div>
          </div>
        )}
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ episode replay
function Player({ ep, algo }: { ep: Episode; algo: AlgoKey }) {
  const [i, setI] = useState(0);
  const [want, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(50);
  const n = ep.actions.length;
  const playing = want && i < n - 1;
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setI((k) => Math.min(n - 1, k + 1)), 1000 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, n]);
  const s = ep.states[i];
  const step = { s, a: ep.actions[i], p: ep.probs[i] } as unknown as StepRec;
  const good = s[2] + 0.5 * s[3] > 0 ? 1 : 0;
  const trace = useMemo(
    () => ({ x: ep.states.map((_, k) => k + 1), ys: [ep.states.map((st) => st[0] / 2.4), ep.states.map((st) => st[2] / 0.2095)] }),
    [ep],
  );
  const markers: Marker[] = [{ x: i + 1, label: "", color: C.ink2 }];
  const actions = CARTPOLE.actions as readonly string[];
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.2fr_1fr]">
      <div>
        <EnvView envId="CartPole-v1" step={step} trail={[]} actionNames={[...actions]} />
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
          <a className="text-[#2a78d6] hover:underline" href={`${BACKEND}/api/cli/run/${algo}/gif/${ep.name}`} target="_blank" rel="noreferrer">
            GIF
          </a>
        </div>
      </div>
      <div className="space-y-3">
        <table className="w-full text-[12px] tabular-nums">
          <tbody>
            {CARTPOLE.state.map((d, k) => (
              <tr key={d.name} className="border-b border-neutral-100">
                <td className="py-0.5 text-neutral-500">{d.name}</td>
                <td className="py-0.5 text-right text-neutral-900">
                  {fmt(s[k], 3)} {d.unit}
                  {k === 2 ? ` (${((s[k] * 180) / Math.PI).toFixed(1)}°)` : ""}
                </td>
                <td className="py-0.5 text-right text-neutral-400" title="network input (scaled)">
                  {(s[k] / d.scale).toFixed(2)}
                </td>
              </tr>
            ))}
            <tr>
              <td className="py-0.5 text-neutral-500">{ep.name === "good" ? "good policy" : "agent"} pushes</td>
              <td className="py-0.5 text-right" style={{ color: ACTION_COLORS[actions[ep.actions[i]]] }}>
                {actions[ep.actions[i]]}
              </td>
              <td className="py-0.5 text-right text-neutral-400">P {ep.probs[i][ep.actions[i]].toFixed(2)}</td>
            </tr>
            {ep.name !== "good" && (
              <tr>
                <td className="py-0.5 text-neutral-500">good policy</td>
                <td className="py-0.5 text-right" style={{ color: ACTION_COLORS[actions[good]] }}>
                  {actions[good]}
                </td>
                <td className="py-0.5 text-right text-neutral-400">{good === ep.actions[i] ? "same" : "differs"}</td>
              </tr>
            )}
          </tbody>
        </table>
        <div>
          <div className="text-[11px] text-neutral-500">cart position / 2.4 and pole angle / 12° (±1 = episode ends)</div>
          <UChart
            x={trace.x}
            ys={trace.ys}
            height={120}
            yRange={[-1, 1]}
            markers={markers}
            xLabel="step"
            series={[
              { label: "cart position", color: C.ink, width: 1.5 },
              { label: "pole angle", color: C.critic, width: 1.5 },
            ]}
          />
        </div>
      </div>
    </div>
  );
}

function Episodes({ data }: { data: RunData }) {
  const learned = data.episodes.filter((e) => e.name !== "good");
  const good = data.episodes.find((e) => e.name === "good");
  const finalIter = data.status.state !== "running" ? learned.at(-1)?.iter : undefined;
  const [sel, setSel] = useState<string | null>(null);
  const current = sel ?? learned.at(-1)?.name ?? null;
  const [ep, setEp] = useState<Episode | null>(null);
  useEffect(() => {
    if (!current) return;
    let alive = true;
    fetch(`${BACKEND}/api/cli/run/${data.algo}/episode/${current}`)
      .then((r) => r.json())
      .then((e: Episode) => alive && setEp(e))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [current, data.algo]);
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        {learned.map((e) => {
          const isFinal = e.iter === finalIter;
          return (
            <button
              key={e.name}
              onClick={() => setSel(e.name)}
              className={`rounded border px-2 py-0.5 text-[12px] tabular-nums ${e.name === current ? "border-neutral-900 bg-neutral-900 text-white" : isFinal ? "border-[#2a78d6] bg-white text-[#2a78d6]" : "border-neutral-300 bg-white"}`}
            >
              {isFinal ? `final ${e.iter}` : e.iter}: {e.steps}
            </button>
          );
        })}
        {good && (
          <button
            onClick={() => setSel(good.name)}
            className={`ml-2 rounded border px-2 py-0.5 text-[12px] tabular-nums ${good.name === current ? "border-[#1baf7a] bg-[#1baf7a] text-white" : "border-[#1baf7a] bg-white text-[#0b7a52]"}`}
          >
            good policy: {good.steps}
          </button>
        )}
        <span className="ml-auto text-[11px] text-neutral-400">iteration: steps balanced</span>
      </div>
      {ep && ep.name === current && <Player key={ep.name} ep={ep} algo={data.algo} />}
    </div>
  );
}

// ------------------------------------------------------------------ the selected algorithm
function UpdatePanel({ c }: { c: AlgoConfig }) {
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
        ) : (
          <Mono>L_π = −A·log π(a|s)</Mono>
        )}
      </Kv>
      {c.adv_norm && <Kv k="normalise">A → mean 0, std 1 in each minibatch</Kv>}
      <Kv k="steps">
        {u.text}
        {u.n != null && u.n > 1 && <span className="text-neutral-500"> = {u.n} gradient steps</span>} (critic and actor each)
      </Kv>
      <Kv k="networks">
        actor 4→{c.hidden}→2 · critic 4→{c.hidden}→1 · {c.optimizer ?? "adam"} · lr {c.actor_lr} / {c.critic_lr}
      </Kv>
      <div className="pt-1.5 text-right">
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

function AlgoDetail({ data }: { data: RunData }) {
  const rows = data.rows;
  return (
    <>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_380px]">
        <Panel
          title="Recorded episodes"
          help="The policy at iteration 1, every 10 iterations and at the end, always from the same start state and always taking its most likely action. 'good policy' = the hand-written rule on the same start."
        >
          {data.episodes.length > 0 ? <Episodes key={data.algo} data={data} /> : <div className="py-10 text-center text-[12px] text-neutral-400">no episodes recorded yet</div>}
        </Panel>
        <div className="space-y-3">
          <UpdatePanel c={data.settings.config} />
          <Panel title="Environment" help="What the agent observes and does. The state is fully observed; each number is divided by its range before going into the networks.">
            {CARTPOLE.state.map((d) => (
              <Kv key={d.name} k={d.name.replace("pole angular velocity", "ang. velocity")}>
                {d.unit} <span className="text-neutral-400">{d.range}</span>
              </Kv>
            ))}
            <Kv k="actions">
              <span style={{ color: ACTION_COLORS.left }}>← left</span> · <span style={{ color: ACTION_COLORS.right }}>→ right</span>
            </Kv>
            <Kv k="reward">{CARTPOLE.reward}</Kv>
            <Kv k="ends">{CARTPOLE.ends}</Kv>
          </Panel>
        </div>
      </div>

      {rows.length > 0 && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          <Chart
            title="Agent: episode length"
            help="Batch: mean length of the episodes that ended in this batch (sampled actions). Eval: the most likely action, 10 fixed starts. 500 = perfect."
            rows={rows}
            series={[
              { key: "ret", label: "batch", color: C.ink },
              { key: "eval", label: "eval", color: C.pos },
            ]}
          />
          <Chart
            title="Actor: is the policy right?"
            help={`P(good action): probability given to the good policy's action (${CARTPOLE.goodPolicy}); 0.5 = coin flip, 1 = perfect. Agreement: how often the most likely action is the good one. Entropy: 0.69 = coin flip, 0 = always the same action.`}
            rows={rows}
            series={[
              { key: "p_good", label: "P(good)", color: C.actor },
              { key: "agree", label: "agree", color: C.actor, dash: [5, 3] },
              { key: "H", label: "entropy", color: C.muted },
            ]}
            range={[0, 1]}
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

export default function RunPage() {
  const [envKey, setEnvKey] = useState<EnvKey>("CartPole-v1");
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
        const res = await fetch(`${BACKEND}/api/cli/algos`);
        if (!res.ok) throw new Error(res.statusText);
        const d: AlgoSummary[] = await res.json();
        if (!alive) return;
        setAlgos(d);
        setErr("");
      } catch (e) {
        if (alive) setErr(e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Backend not reachable on :8000.");
      }
      if (alive) timer = window.setTimeout(load, 3000);
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  // default: the algorithm training right now, else the highest rung of the ladder that has a run
  const fallback = algos?.find((a) => isLive(a.status))?.algo ?? [...LADDER].reverse().find((l) => algos?.some((a) => a.algo === l.key))?.key ?? null;
  const algo: AlgoKey = picked ?? fallback ?? "batch_ac";
  const summary = algos?.find((a) => a.algo === algo);
  const version = summary ? `${summary.status.state}|${summary.status.started}|${summary.status.iter}` : "";

  // the selected run in full; refetched whenever its status in the list changes
  useEffect(() => {
    if (!version) return;
    let alive = true;
    fetch(`${BACKEND}/api/cli/run/${algo}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: RunData | null) => alive && d && setRun(d))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [algo, version]);

  const data = run?.algo === algo ? run : null;
  const st: Status = data?.status ?? summary?.status ?? {};
  const cfg = data?.settings.config ?? summary?.settings.config;
  const L = LADDER_BY[algo];
  const commandOne = CARTPOLE.command(algo);
  const commandAll = `${CARTPOLE.command("all")} --render-every 0`;
  const copy = (text: string, key: string) => navigator.clipboard?.writeText(text).then(() => setCopied(key)).catch(() => {});
  const progress = st.budget && st.env_steps != null ? Math.min(1, st.env_steps / st.budget) : null;

  return (
    <div className="min-h-screen bg-[#fafaf9]">
      <NavBar
        active="run"
        right={
          <div className="flex items-center gap-1 rounded-md bg-neutral-100 p-0.5 text-[12px]">
            {(Object.keys(ENVS) as EnvKey[]).map((k) => (
              <button
                key={k}
                disabled={!ENVS[k].ready}
                onClick={() => setEnvKey(k)}
                title={ENVS[k].ready ? "" : "next"}
                className={`rounded px-2.5 py-1 ${envKey === k ? "bg-white font-medium text-neutral-900 shadow-sm" : "text-neutral-400"} disabled:cursor-not-allowed`}
              >
                {k}
                {!ENVS[k].ready && <span className="ml-1 text-[10px]">(next)</span>}
              </button>
            ))}
          </div>
        }
      />
      <main className="mx-auto max-w-[1400px] space-y-3 p-3">
        {/* the ladder */}
        <div className="rounded-lg border border-neutral-200 bg-white px-4 py-3">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h1 className="text-[18px] font-semibold text-neutral-900">From Batch Actor-Critic to PPO</h1>
            <span className="text-[13px] text-neutral-500">one change at a time · same 200,000 environment steps each · {envKey}</span>
            <button
              className="ml-auto rounded border border-neutral-300 px-2 py-0.5 font-mono text-[11px] hover:bg-neutral-50"
              title={commandAll}
              onClick={() => copy(commandAll, "all")}
            >
              {copied === "all" ? "copied ✓" : "copy command: train all 5"}
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
                              ? `500 after ${num(speed)} steps`
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
  );
}
