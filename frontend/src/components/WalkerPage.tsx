"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import NavBar from "@/components/NavBar";
import Panel, { ChartTitle } from "@/components/Panel";
import UChart, { LegendToggles, SeriesSpec } from "@/components/UChart";
import { ENVS } from "@/components/RunPage";
import { BACKEND, STATIC, cliUrl } from "@/lib/api";
import { C } from "@/lib/colors";

// Walker2d (MuJoCo): continuous actions. Reads runs/walker2d/<algo>/ written by scripts/train_continuous.py.
const ENV = "walker2d";
const COMMAND = (algo: string) => `cd ~/my_deeprl_playground/backend && uv run python scripts/train_continuous.py --algo ${algo}`;
// The continuous-control ladder: built ones first; the rest appear when they have a run.
const LADDER: { key: string; name: string; what: string }[] = [
  { key: "ppo", name: "PPO (Gaussian)", what: "on-policy: the actor outputs a mean torque per joint, σ learned per joint" },
  { key: "ddpg", name: "DDPG", what: "off-policy: DQN's replay buffer and target network + an actor that learns the argmax" },
  { key: "td3", name: "TD3", what: "+ two critics (take the smaller), delayed actor updates, target smoothing" },
  { key: "sac", name: "SAC", what: "+ maximum entropy: a squashed Gaussian that keeps exploring" },
];
const nameOf = (algo: string) => LADDER.find((l) => l.key === algo)?.name ?? algo;
const JOINTS = ["hip R", "knee R", "ankle R", "hip L", "knee L", "ankle L"];
const RIGHT = "#cc9966"; // MuJoCo's colours: right leg and torso tan, left leg purple
const LEFT = "#b34d99";

type Row = Record<string, number | null>;
interface Status {
  state?: string;
  iter?: number;
  env_steps?: number;
  budget?: number;
  started?: string;
}
interface EpInfo {
  name: string;
  iter: number | null;
  steps: number;
  terminated: boolean | null;
  score: number | null;
  distance: number | null;
}
interface Scorecard {
  median_score: number;
  falls: number;
  stands_still: number;
  median_dist: number;
  scores: number[];
  first_eval_steps: Record<string, number | null>;
}
interface Run {
  algo: string;
  settings: { algo_name: string; budget: number; seed: number; config: Record<string, unknown> };
  status: Status;
  rows: Row[];
  episodes: EpInfo[];
  scorecard: Scorecard | null;
}
interface Episode extends EpInfo {
  dt: number;
  every: number;
  env_steps?: number;
  geoms: { name: string; radius: number }[];
  frames: number[][];
  actions: number[][];
}

const colorOf = (algo: string) => C.series[Math.max(0, LADDER.findIndex((l) => l.key === algo)) % C.series.length];
const fmt = (v: number | null | undefined, d = 0) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(d));
const kfmt = (v: number | null | undefined) => (v == null ? "never" : v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : `${Math.round(v / 1e3)}k`);

// ------------------------------------------------------------------ data
function useRuns() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const load = async () => {
      try {
        const res = await fetch(cliUrl("algos", ENV));
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        const list: { algo: string }[] = await res.json();
        const full = await Promise.all(
          list.map((a) => fetch(cliUrl(`run/${a.algo}`, ENV)).then((r) => (r.ok ? (r.json() as Promise<Run>) : null))),
        );
        if (alive) {
          setRuns(full.filter((r): r is Run => r !== null).sort((a, b) => order(a.algo) - order(b.algo)));
          setErr(null);
        }
      } catch (e) {
        if (alive) setErr(STATIC ? "Could not load the saved runs." : e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Backend not reachable on :8000.");
      }
      if (alive && !STATIC) timer = window.setTimeout(load, 3000);
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);
  return { runs, err };
}
const order = (algo: string) => {
  const k = LADDER.findIndex((l) => l.key === algo);
  return k < 0 ? 99 : k;
};

/** One x axis (million env steps) shared by every algorithm; each series is null where its algorithm has no row. */
function align(runs: Run[], pick: (r: Row) => number | null | undefined) {
  const xs = Array.from(new Set(runs.flatMap((r) => r.rows.map((row) => row.env_steps as number)))).sort((a, b) => a - b);
  const at = new Map(xs.map((x, i) => [x, i]));
  const ys = runs.map((run) => {
    const col: (number | null)[] = new Array(xs.length).fill(null);
    run.rows.forEach((row) => {
      const v = pick(row);
      if (v != null && Number.isFinite(v)) col[at.get(row.env_steps as number)!] = v;
    });
    return col;
  });
  return { x: xs.map((v) => v / 1e6), ys };
}

function MultiChart({ runs, pick, title, help, yRange, yLog, points }: {
  runs: Run[];
  pick: (r: Row) => number | null | undefined;
  title: string;
  help: string;
  yRange?: [number | null, number | null];
  yLog?: boolean;
  points?: boolean;
}) {
  const { x, ys } = useMemo(() => align(runs, pick), [runs, pick]);
  const series: SeriesSpec[] = runs.map((r) => ({ label: nameOf(r.algo), color: colorOf(r.algo), points, width: 1.6 }));
  return (
    <div>
      <ChartTitle help={help}>{title}</ChartTitle>
      <UChart x={x} ys={ys} series={series} height={170} yRange={yRange} yLog={yLog} sync="walker" xLabel="million env steps" />
    </div>
  );
}

// ------------------------------------------------------------------ the walker
function WalkerSvg({ ep, frame }: { ep: Episode; frame: number }) {
  const f = ep.frames[Math.min(frame, ep.frames.length - 1)];
  const torso = ep.geoms.findIndex((g) => g.name === "torso_geom");
  const cx = (f[4 * torso] + f[4 * torso + 2]) / 2; // the camera follows the torso
  const W = 6; // metres shown; the panel crops the sides on narrow screens (the camera keeps the torso centred)
  const x0 = cx - W / 2;
  const ticks: number[] = [];
  for (let t = Math.ceil(x0 * 2) / 2; t <= x0 + W; t += 0.5) ticks.push(t);
  return (
    <svg viewBox={`${x0} -1.85 ${W} 2.05`} preserveAspectRatio="xMidYMid slice" className="h-[230px] w-full rounded-md bg-[#f6f7f4]">
      <rect x={x0 - 10} y={0} width={W + 20} height={0.3} fill="#e3e6dc" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={t} x2={t} y1={0} y2={Number.isInteger(t) ? 0.08 : 0.04} stroke="#9aa08e" strokeWidth={0.008} />
          {Number.isInteger(t) && (
            <text x={t} y={0.17} fontSize={0.07} textAnchor="middle" fill="#7b8070">
              {t} m
            </text>
          )}
        </g>
      ))}
      <line x1={x0 - 10} x2={x0 + W + 10} y1={0} y2={0} stroke="#9aa08e" strokeWidth={0.01} />
      {ep.geoms.map((g, i) => (
        <line
          key={g.name}
          x1={f[4 * i]}
          y1={-f[4 * i + 1]}
          x2={f[4 * i + 2]}
          y2={-f[4 * i + 3]}
          stroke={g.name.includes("left") ? LEFT : RIGHT}
          strokeWidth={2 * g.radius}
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}

function Torques({ a }: { a: number[] }) {
  return (
    <div className="grid grid-cols-6 gap-1.5">
      {a.map((v, j) => (
        <div key={j} className="text-center">
          <div className="relative h-14 rounded bg-neutral-100">
            <div className="absolute left-0 right-0 top-1/2 h-px bg-neutral-300" />
            <div
              className="absolute left-1 right-1"
              style={{
                background: j < 3 ? RIGHT : LEFT,
                top: v >= 0 ? `${50 - 50 * Math.min(1, v)}%` : "50%",
                height: `${50 * Math.min(1, Math.abs(v))}%`,
              }}
            />
          </div>
          <div className="mt-0.5 text-[10px] leading-tight text-neutral-500">{JOINTS[j]}</div>
          <div className="font-mono text-[10px] text-neutral-700">{v >= 0 ? "+" : ""}{v.toFixed(2)}</div>
        </div>
      ))}
    </div>
  );
}

function useEpisode(algo: string | null, name: string | null) {
  const [ep, setEp] = useState<Episode | null>(null);
  useEffect(() => {
    if (!algo || !name) return;
    let alive = true;
    fetch(cliUrl(`run/${algo}/episode/${name}`, ENV))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => alive && setEp(d));
    return () => {
      alive = false;
    };
  }, [algo, name]);
  return ep && ep.name === name ? ep : null;
}

function Viewer({ runs, side, algo, setAlgo, name, setName, ep, time }: {
  runs: Run[];
  side: string;
  algo: string | null;
  setAlgo: (a: string) => void;
  name: string | null;
  setName: (n: string) => void;
  ep: Episode | null;
  time: number;
}) {
  const run = runs.find((r) => r.algo === algo);
  const recs = (run?.episodes ?? []).filter((e) => e.name.startsWith("iter"));
  const frame = ep ? Math.floor(time / (ep.dt * ep.every)) : 0;
  const done = ep ? frame >= ep.frames.length - 1 : false;
  const k = ep ? Math.min(frame, ep.frames.length - 1) : 0;
  const steps = ep ? Math.min(ep.steps, k * ep.every) : 0;
  return (
    <div className="min-w-0 flex-1 space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        <span className="font-medium text-neutral-800">{side}</span>
        <select className="rounded border border-neutral-300 px-1 py-0.5" value={algo ?? ""} onChange={(e) => setAlgo(e.target.value)}>
          {runs.map((r) => (
            <option key={r.algo} value={r.algo}>
              {nameOf(r.algo)}
            </option>
          ))}
        </select>
        <select className="rounded border border-neutral-300 px-1 py-0.5" value={name ?? ""} onChange={(e) => setName(e.target.value)}>
          {recs.map((e) => (
            <option key={e.name} value={e.name}>
              iteration {e.iter} · score {fmt(e.score)} · {fmt(e.distance, 1)} m
            </option>
          ))}
        </select>
      </div>
      {ep && ep.frames ? (
        <>
          <div className="relative">
            <WalkerSvg ep={ep} frame={frame} />
            <div className="absolute left-2 top-2 rounded bg-white/85 px-1.5 py-0.5 font-mono text-[11px] text-neutral-700">
              {ep.env_steps != null ? `${kfmt(ep.env_steps)} env steps of training · ` : ""}step {steps} / {ep.steps}
            </div>
            {done && (
              <div className={`absolute right-2 top-2 rounded px-1.5 py-0.5 text-[11px] font-medium ${ep.terminated ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-700"}`}>
                {ep.terminated ? `fell after ${ep.steps} steps` : "still up at the 1,000-step limit"}
              </div>
            )}
          </div>
          <div className="flex gap-4 text-[12px] text-neutral-600">
            <span>score <b className="text-neutral-900">{fmt(ep.score)}</b></span>
            <span>walked <b className="text-neutral-900">{fmt(ep.distance, 1)} m</b></span>
            <span>{ep.terminated ? "fell" : "time limit"}</span>
          </div>
          <Torques a={ep.actions[k]} />
        </>
      ) : ep && algo && name ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img alt="" src={`${BACKEND}/api/cli/run/${algo}/gif/${name}?env=${ENV}`} className="w-full rounded-md" />
      ) : (
        <div className="flex h-[230px] items-center justify-center rounded-md bg-neutral-50 text-[12px] text-neutral-400">
          {recs.length ? "loading …" : "no recording yet"}
        </div>
      )}
    </div>
  );
}

function Replay({ runs }: { runs: Run[] }) {
  const first = runs[0]?.algo ?? null;
  const [algoA, setAlgoA] = useState<string | null>(null);
  const [algoB, setAlgoB] = useState<string | null>(null);
  const [nameA, setNameA] = useState<string | null>(null);
  const [nameB, setNameB] = useState<string | null>(null);
  const aA = algoA ?? first;
  const aB = algoB ?? first;
  const recsOf = (algo: string | null) => (runs.find((r) => r.algo === algo)?.episodes ?? []).filter((e) => e.name.startsWith("iter"));
  // defaults: A = the first recording, B = the latest one (follows the run while it trains)
  const nA = nameA ?? recsOf(aA)[0]?.name ?? null;
  const nB = nameB ?? recsOf(aB).at(-1)?.name ?? null;
  const epA = useEpisode(aA, nA);
  const epB = useEpisode(aB, nB);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const dur = Math.max(epA ? epA.steps * epA.dt : 0, epB ? epB.steps * epB.dt : 0);
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || !dur) return;
    let raf = 0;
    const tick = (now: number) => {
      const step = last.current == null ? 0 : ((now - last.current) / 1000) * speed;
      last.current = now;
      setTime((t) => (t + step > dur + 1 ? 0 : t + step)); // loop, with a 1 s pause on the final pose
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      last.current = null;
    };
  }, [playing, speed, dur]);
  const pickA = (n: string) => (setNameA(n), setTime(0));
  const pickB = (n: string) => (setNameB(n), setTime(0));
  return (
    <Panel
      title="Watch it walk: two recordings side by side"
      help="Every 10 iterations the training records the current policy (its mean action, no sampling) from the same start. Pick any two recordings, of the same algorithm at different points of training or of two algorithms, and play them together. The camera follows the torso; the ticks on the ground are every 0.5 m. Below each: the 6 torques at this moment (tan = right leg, purple = left leg), each between −1 and +1."
      right={
        <div className="flex items-center gap-2 text-[12px]">
          <button className="rounded border border-neutral-300 px-2 py-0.5 hover:bg-neutral-50" onClick={() => setPlaying((p) => !p)}>
            {playing ? "pause" : "play"}
          </button>
          <button className="rounded border border-neutral-300 px-2 py-0.5 hover:bg-neutral-50" onClick={() => setTime(0)}>
            restart
          </button>
          {[0.25, 0.5, 1, 2].map((s) => (
            <button key={s} onClick={() => setSpeed(s)} className={`rounded px-1.5 py-0.5 ${s === speed ? "bg-neutral-900 text-white" : "text-neutral-500 hover:text-neutral-900"}`}>
              {s}×
            </button>
          ))}
        </div>
      }
    >
      <input
        type="range"
        min={0}
        max={dur || 1}
        step={0.008}
        value={Math.min(time, dur)}
        onChange={(e) => (setPlaying(false), setTime(+e.target.value))}
        className="mb-2 w-full"
      />
      <div className="flex flex-col gap-4 md:flex-row">
        <Viewer runs={runs} side="A" algo={aA} setAlgo={(a) => (setAlgoA(a), setNameA(null))} name={nA} setName={pickA} ep={epA} time={time} />
        <Viewer runs={runs} side="B" algo={aB} setAlgo={(a) => (setAlgoB(a), setNameB(null))} name={nB} setName={pickB} ep={epB} time={time} />
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ page
const pick = {
  eval: (r: Row) => r.eval_score,
  score: (r: Row) => r.score,
  dist: (r: Row) => r.dist,
  len: (r: Row) => r.len,
  fell: (r: Row) => r.fell,
  std: (r: Row) => r.std,
  clipped: (r: Row) => r.act_clipped,
  ev: (r: Row) => r.EV,
  kl: (r: Row) => r.KL,
};

function StatusLine({ run }: { run: Run }) {
  const st = run.status;
  const p = st.budget && st.env_steps != null ? Math.min(1, st.env_steps / st.budget) : null;
  const last = run.rows.at(-1);
  const secs = (last?.seconds as number | undefined) ?? null;
  const eta = p && secs && p < 1 ? (secs / p) * (1 - p) : null;
  const evals = run.rows.filter((r) => r.eval_score != null);
  const best = evals.length ? Math.max(...evals.map((r) => r.eval_score as number)) : null;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: colorOf(run.algo) }} />
      <span className="w-28 font-medium text-neutral-900">{nameOf(run.algo)}</span>
      <span className={`rounded px-1.5 py-0.5 text-[11px] ${st.state === "running" ? "bg-blue-50 text-blue-700" : st.state === "finished" ? "bg-emerald-50 text-emerald-700" : "bg-neutral-100 text-neutral-600"}`}>
        {st.state ?? "?"}
      </span>
      <div className="h-1.5 w-40 overflow-hidden rounded bg-neutral-100">
        <div className="h-full" style={{ width: `${100 * (p ?? 0)}%`, background: colorOf(run.algo) }} />
      </div>
      <span className="font-mono text-neutral-600">
        {(st.env_steps ?? 0).toLocaleString()} / {(st.budget ?? 0).toLocaleString()} steps
      </span>
      {secs != null && <span className="text-neutral-500">{(secs / 60).toFixed(1)} min{eta != null ? ` · ~${Math.ceil(eta / 60)} min left` : ""}</span>}
      <span className="text-neutral-600">
        last eval <b className="text-neutral-900">{fmt(evals.at(-1)?.eval_score)}</b> · best {fmt(best)} · walked {fmt(evals.at(-1)?.eval_dist, 1)} m
      </span>
    </div>
  );
}

function Scorecards({ runs }: { runs: Run[] }) {
  const done = runs.filter((r) => r.scorecard);
  if (!done.length) return null;
  return (
    <Panel
      title="Final test: the trained policy from 20 new start states"
      help="After training, the mean action (no sampling) plays 20 episodes from start states it never trained on, up to 1,000 steps each. 'Stood still' counts episodes that never fell but moved less than 1 m: standing earns +1 per step, so a policy can score ~1,000 without walking. 'First reached' is when the 5-start evaluation first averaged that score during training."
    >
      <table className="w-full text-[12px]">
        <thead className="text-left text-neutral-500">
          <tr>
            <th className="py-1 font-normal">algorithm</th>
            <th className="font-normal">median score</th>
            <th className="font-normal">median distance</th>
            <th className="font-normal">fell</th>
            <th className="font-normal">stood still</th>
            <th className="font-normal">eval first reached 1,000 / 2,000 / 3,000 / 4,000</th>
          </tr>
        </thead>
        <tbody>
          {done.map((r) => {
            const s = r.scorecard!;
            return (
              <tr key={r.algo} className="border-t border-neutral-100">
                <td className="py-1 font-medium" style={{ color: colorOf(r.algo) }}>{nameOf(r.algo)}</td>
                <td className="font-mono">{fmt(s.median_score)}</td>
                <td className="font-mono">{fmt(s.median_dist, 1)} m</td>
                <td className="font-mono">{s.falls}/{s.scores.length}</td>
                <td className="font-mono">{s.stands_still}/{s.scores.length}</td>
                <td className="font-mono">{["1000", "2000", "3000", "4000"].map((t) => kfmt(s.first_eval_steps[t])).join(" / ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}

export default function WalkerPage() {
  const { runs, err } = useRuns();
  const [copied, setCopied] = useState<string | null>(null);
  const copy = (text: string, key: string) => navigator.clipboard?.writeText(text).then(() => setCopied(key)).catch(() => {});
  const have = new Set(runs?.map((r) => r.algo));
  return (
    <LegendToggles>
      <div className="min-h-screen bg-[#fafaf9]">
        <NavBar
          active="run"
          right={
            <div className="flex items-center gap-1 rounded-md bg-neutral-100 p-0.5 text-[12px]">
              {Object.values(ENVS).map((e) => (
                <Link key={e.key} href={e.href} className="rounded px-2.5 py-1 text-neutral-500 hover:text-neutral-900">
                  {e.tab}
                </Link>
              ))}
              <span className="rounded bg-white px-2.5 py-1 font-medium text-neutral-900 shadow-sm">Walker2d</span>
            </div>
          }
        />
        <main className="mx-auto max-w-[1400px] space-y-3 p-3">
          <div className="rounded-lg border border-neutral-200 bg-white px-4 py-3">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h1 className="text-[18px] font-semibold text-neutral-900">Walker2d: continuous actions</h1>
              <span className="text-[13px] text-neutral-500">
                6 torques in [−1, 1] instead of a few buttons · 17 state numbers · Walker2d-v5 (MuJoCo) · reward: forward speed + 1 per step upright − 0.001·Σ torque²
              </span>
            </div>
            <div className="mt-3 flex flex-wrap items-stretch gap-1.5">
              {LADDER.map((l, k) => (
                <div key={l.key} className="flex items-center gap-1.5">
                  {k > 0 && <span className="text-neutral-300">→</span>}
                  <div className={`rounded-md border px-2.5 py-1.5 text-[12px] ${have.has(l.key) ? "border-neutral-300 bg-white" : "border-dashed border-neutral-200 bg-neutral-50 text-neutral-400"}`} title={l.what}>
                    <div className="font-medium" style={have.has(l.key) ? { color: colorOf(l.key) } : undefined}>{l.name}</div>
                    <div className="max-w-[220px] text-[11px] leading-tight text-neutral-500">{l.what}</div>
                    {k === 0 || have.has(l.key) ? (
                      <button className="mt-1 rounded border border-neutral-300 px-1.5 py-0.5 font-mono text-[10px] hover:bg-neutral-50" onClick={() => copy(COMMAND(l.key), l.key)}>
                        {copied === l.key ? "copied ✓" : "copy train command"}
                      </button>
                    ) : (
                      <div className="mt-1 text-[10px] italic">not built yet</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 space-y-1.5">
              {err && <div className="text-[12px] text-red-600">{err}</div>}
              {runs && !runs.length && (
                <div className="text-[12px] text-neutral-500">
                  No run yet. Start one: <code className="rounded bg-neutral-100 px-1">{COMMAND("ppo")}</code>. This page follows it live.
                </div>
              )}
              {runs?.map((r) => <StatusLine key={r.algo} run={r} />)}
            </div>
          </div>

          {runs && runs.length > 0 && (
            <>
              <Replay runs={runs} />
              <Panel
                title="Learning curves (every algorithm on the same axis: environment steps)"
                help="One point per iteration. For PPO, an iteration is one batch of 2,048 steps plus its update. Click a legend entry to hide that algorithm on every chart."
              >
                <div className="grid grid-cols-1 gap-x-4 gap-y-3 md:grid-cols-2 xl:grid-cols-3">
                  <MultiChart runs={runs} pick={pick.eval} points title="Score of the mean action (5 fixed starts)" help="Every 10 iterations: the policy without sampling (it always takes the mean torque) plays 5 episodes from fixed starts. Real reward. About 1,000 = stands for the whole episode; 3,000–4,000 = walks well." />
                  <MultiChart runs={runs} pick={pick.score} title="Score while training (sampled actions)" help="The mean return of the episodes that ended in each batch, with the actions sampled from the Gaussian as in training. Lower than the mean action's score while σ is large: the noise makes it stumble." />
                  <MultiChart runs={runs} pick={pick.dist} title="Distance walked per episode (m)" help="Where the torso was when the episode ended. Separates walking from standing: a standing policy can score ~1,000 with almost 0 m." />
                  <MultiChart runs={runs} pick={pick.len} yRange={[0, 1000]} title="Episode length (1,000 = never fell)" help="Steps until the walker fell (torso too low, too high or tilted more than 1 rad) or reached the 1,000-step limit." />
                  <MultiChart runs={runs} pick={pick.std} title="σ: the spread of the Gaussian (mean of the 6 joints)" help="How much the policy explores. It starts at 1 and shrinks as the policy becomes sure. PPO learns σ through the clipped objective; there is no entropy bonus here." />
                  <MultiChart runs={runs} pick={pick.clipped} yRange={[0, 1]} title="Share of sampled torques clipped to ±1" help="The Gaussian can sample torques outside [−1, 1]; they are cut to the limit before reaching the robot, but log π still uses the uncut value. A large share means the policy pushes against the limits. SAC avoids this with a tanh squash." />
                  <MultiChart runs={runs} pick={pick.ev} yRange={[-0.2, 1]} title="Critic: explained variance" help="1 = the critic predicts its targets (GAE targets, scaled rewards); 0 = no better than a constant." />
                  <MultiChart runs={runs} pick={pick.kl} yLog title="KL(π_old ‖ π_new) per update" help="How far one update moved the Gaussian, averaged over the batch states (closed form for two Gaussians). The learning rate falls linearly to 0, so it shrinks towards the end." />
                  <MultiChart runs={runs} pick={pick.fell} yRange={[0, 1]} title="Share of episodes that fell" help="Of the episodes that ended in each batch, the share that fell (the rest reached the 1,000-step limit)." />
                </div>
              </Panel>
              <Scorecards runs={runs} />
            </>
          )}
        </main>
      </div>
    </LegendToggles>
  );
}
