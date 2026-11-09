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
interface EpisodeInfo {
  name: string;
  iter: number;
  steps: number;
  terminated: boolean;
}
interface Latest {
  settings: Record<string, number | string | boolean>;
  status: { state?: string; iter?: number; iters?: number; started?: string };
  rows: Row[];
  episodes: EpisodeInfo[];
}
interface Episode extends EpisodeInfo {
  states: number[][];
  actions: number[];
  probs: number[][];
}

/** Everything that differs between environments. MountainCar plugs in here next. */
const ENVS = {
  "CartPole-v1": {
    ready: true,
    command: "cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py",
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
  "MountainCar-v0": { ready: false, command: "", actions: ["left", "none", "right"], state: [], reward: "", ends: "", goodPolicy: "" },
} as const;
type EnvKey = keyof typeof ENVS;

// ------------------------------------------------------------------ small pieces
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
    const ys = series.map((s) => rows.map((r) => r[s.key] ?? null));
    const specs: SeriesSpec[] = series.map((s) => ({ label: s.label, color: s.color, width: 1.75, dash: s.dash }));
    return { x, ys, specs };
  }, [rows, series]);
  return (
    <Panel title={title} help={help}>
      <UChart x={x} ys={ys} series={specs} yLog={log} yRange={range} height={170} sync="run" xLabel="iteration" />
    </Panel>
  );
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

// ------------------------------------------------------------------ episode replay
function Player({ ep, env }: { ep: Episode; env: (typeof ENVS)["CartPole-v1"] }) {
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
  const actions = env.actions as readonly string[];
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
          <a className="text-[#2a78d6] hover:underline" href={`${BACKEND}/api/cli/latest/gif/${ep.name}`} target="_blank" rel="noreferrer">
            GIF
          </a>
        </div>
      </div>
      <div className="space-y-3">
        <table className="w-full text-[12px] tabular-nums">
          <tbody>
            {env.state.map((d, k) => (
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

function Episodes({ data, env }: { data: Latest; env: (typeof ENVS)["CartPole-v1"] }) {
  const learned = data.episodes.filter((e) => e.name !== "good");
  const good = data.episodes.find((e) => e.name === "good");
  const finalIter = data.status.state !== "running" ? learned.at(-1)?.iter : undefined;
  const [sel, setSel] = useState<string | null>(null);
  const current = sel ?? learned.at(-1)?.name ?? null;
  const [ep, setEp] = useState<Episode | null>(null);
  useEffect(() => {
    if (!current) return;
    let alive = true;
    fetch(`${BACKEND}/api/cli/latest/episode/${current}`)
      .then((r) => r.json())
      .then((e: Episode) => alive && setEp(e))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [current]);
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
      {ep && ep.name === current && <Player key={ep.name} ep={ep} env={env} />}
    </div>
  );
}

const TABLE: [string, string, number][] = [
  ["iter", "iter", 0],
  ["ret", "length", 1],
  ["eval", "eval", 0],
  ["p_good", "P(good)", 2],
  ["H", "entropy", 2],
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

// ------------------------------------------------------------------ page
export default function RunPage() {
  const [envKey, setEnvKey] = useState<EnvKey>("CartPole-v1");
  const [data, setData] = useState<Latest | null>(null);
  const [err, setErr] = useState("");
  const [copied, setCopied] = useState(false);
  const env = ENVS["CartPole-v1"];

  useEffect(() => {
    let alive = true;
    let timer = 0;
    const load = async () => {
      try {
        const res = await fetch(`${BACKEND}/api/cli/latest`);
        if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
        const d: Latest = await res.json();
        if (!alive) return;
        setData(d);
        setErr("");
        if (d.status.state === "running") timer = window.setTimeout(load, 2000); // live while training
      } catch (e) {
        if (alive) setErr(e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Backend not reachable on :8000.");
      }
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  const rows = data?.rows ?? [];
  const st = data?.status ?? {};
  const S = data?.settings ?? {};

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
        {/* title + run bar */}
        <div className="rounded-lg border border-neutral-200 bg-white px-4 py-3">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h1 className="text-[18px] font-semibold text-neutral-900">Batch Actor-Critic</h1>
            <span className="text-[13px] text-neutral-500">on-policy · Monte Carlo returns · {envKey}</span>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[12px] text-neutral-600">
            {err ? (
              <span>{err}</span>
            ) : (
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: st.state === "running" ? C.pos : st.state === "finished" ? C.actor : C.marker2 }} />
                <b className="font-medium text-neutral-800">{st.state ?? "…"}</b>
                {st.iter != null && (
                  <>
                    · iteration {st.iter}/{st.iters} · started {st.started}
                  </>
                )}
              </span>
            )}
            {["batch", "hidden", "actor_lr", "critic_lr", "gamma", "optimizer", "seed"].map((k) =>
              S[k] != null ? (
                <span key={k} className="rounded bg-neutral-100 px-1.5 py-0.5 tabular-nums">
                  {k} {String(S[k])}
                </span>
              ) : null,
            )}
            <button
              className="ml-auto rounded border border-neutral-300 px-2 py-0.5 font-mono text-[11px] hover:bg-neutral-50"
              title={env.command}
              onClick={() => navigator.clipboard?.writeText(env.command).then(() => setCopied(true)).catch(() => {})}
            >
              {copied ? "copied ✓" : "copy train command"}
            </button>
          </div>
        </div>

        {/* replay + setup */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_340px]">
          <Panel
            title="Recorded episodes"
            help="The policy every 10 iterations (and at the end), always from the same start state and always taking its most likely action. 'good policy' = the hand-written rule on the same start."
          >
            {data && data.episodes.length > 0 ? <Episodes data={data} env={env} /> : <div className="py-10 text-center text-[12px] text-neutral-400">No run yet: copy the train command and run it in a terminal.</div>}
          </Panel>
          <div className="space-y-3">
            <Panel title="Environment" help="What the agent observes and does. The state is fully observed; each number is divided by its range before going into the networks.">
              {env.state.map((d) => (
                <Kv key={d.name} k={d.name.replace("pole angular velocity", "ang. velocity")}>
                  {d.unit} <span className="text-neutral-400">{d.range}</span>
                </Kv>
              ))}
              <Kv k="actions">
                <span style={{ color: ACTION_COLORS.left }}>← left</span> · <span style={{ color: ACTION_COLORS.right }}>→ right</span>
              </Kv>
              <Kv k="reward">{env.reward}</Kv>
              <Kv k="ends">{env.ends}</Kv>
            </Panel>
            <Panel title="Update per iteration" help="One iteration = collect a batch with the frozen policy, one critic step, one actor step; then the batch is thrown away (on-policy).">
              <Kv k="batch">
                {String(S.batch ?? 5)} full episodes, frozen π → rows (s, a, r)
              </Kv>
              <Kv k="critic">
                <Mono>G = r + γ·G′ · L_V = ½(G − V)²</Mono>
              </Kv>
              <Kv k="actor">
                <Mono>A = G − V · L_π = −A·log π(a|s)</Mono>
              </Kv>
              <Kv k="objective">
                <Mono>max J = E[Σ γᵗ rₜ]</Mono>
              </Kv>
              <Kv k="networks">
                actor 4→{String(S.hidden ?? 8)}→2 · critic 4→{String(S.hidden ?? 8)}→1 · {String(S.optimizer ?? "adam")}
              </Kv>
              <div className="pt-1.5 text-right">
                <Link href="/learn" className="text-[12px] text-[#2a78d6] hover:underline">
                  step through one update →
                </Link>
              </div>
            </Panel>
          </div>
        </div>

        {/* learning */}
        {rows.length > 0 && (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <Chart
              title="Agent: episode length"
              help="Batch: mean length of the episodes played with sampled actions. Eval: the most likely action, 10 fixed starts. 500 = perfect."
              rows={rows}
              series={[
                { key: "ret", label: "batch", color: C.ink },
                { key: "eval", label: "eval", color: C.pos },
              ]}
            />
            <Chart
              title="Actor: is the policy right?"
              help={`P(good action): probability given to the good policy's action (${env.goodPolicy}); 0.5 = coin flip, 1 = perfect. Agreement: how often the most likely action is the good one. Entropy: 0.69 = coin flip, 0 = always the same action.`}
              rows={rows}
              series={[
                { key: "p_good", label: "P(good)", color: C.actor },
                { key: "agree", label: "agree", color: C.actor, dash: [5, 3] },
                { key: "H", label: "entropy", color: C.muted },
              ]}
              range={[0, 1]}
            />
            <Chart
              title="Critic: loss ½(G − V)²"
              help="The raw loss on each new batch. It grows when episodes get longer (bigger returns), so it does not show by itself whether the critic learns."
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
          </div>
        )}

        {rows.length > 0 && (
          <Panel title="Log" help="The numbers the terminal printed (runs/cli/latest/log.csv).">
            <Logs rows={rows} />
          </Panel>
        )}
      </main>
    </div>
  );
}
