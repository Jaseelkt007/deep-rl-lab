"use client";

import Link from "next/link";
import { ReactNode, useEffect, useMemo, useState } from "react";
import { BACKEND } from "@/lib/api";
import { ACTION_COLORS, C } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { StepRec } from "@/lib/types";
import EnvView from "./EnvView";
import { Help } from "./Panel";
import UChart, { Marker, SeriesSpec } from "./UChart";

// ------------------------------------------------------------------ data from the backend
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

const ACTIONS = ["left", "right"];
const COMMAND = "cd ~/my_deeprl_playground/backend && uv run python scripts/train_cartpole.py";

// CartPole state: what each number is, and the range mapped to [−1, 1] before the networks see it
const STATE = [
  { name: "cart position", unit: "m", limit: "±2.4 (track end: episode ends)", scale: 2.4 },
  { name: "cart velocity", unit: "m/s", limit: "no limit (scaled by ±3.0)", scale: 3.0 },
  { name: "pole angle", unit: "rad", limit: "±0.21 = ±12° (pole falls: episode ends)", scale: 0.2095 },
  { name: "pole angular velocity", unit: "rad/s", limit: "no limit (scaled by ±3.5)", scale: 3.5 },
];

// ------------------------------------------------------------------ layout pieces
function Section({ n, title, help, children }: { n: number; title: string; help?: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <header className="mb-3 flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-neutral-900 text-[11px] font-semibold text-white">{n}</span>
        <h2 className="text-[15px] font-semibold text-neutral-900">{title}</h2>
        {help && <Help text={help} />}
      </header>
      {children}
    </section>
  );
}

function Card({ title, color, children }: { title: string; color?: string; children: ReactNode }) {
  return (
    <div className="rounded-md border border-neutral-200 p-3">
      <div className="mb-1.5 text-[12px] font-semibold" style={{ color: color ?? C.ink }}>
        {title}
      </div>
      <div className="space-y-1.5 text-[12.5px] leading-relaxed text-neutral-700">{children}</div>
    </div>
  );
}

const Eq = ({ children }: { children: ReactNode }) => (
  <div className="rounded bg-neutral-50 px-2 py-1 font-mono text-[12px] text-neutral-900">{children}</div>
);

function Chart({ title, rows, series, log, range, note }: {
  title: string;
  rows: Row[];
  series: { key: string; label: string; color: string; dash?: number[] }[];
  log?: boolean;
  range?: [number | null, number | null];
  note: ReactNode;
}) {
  const { x, ys, specs } = useMemo(() => {
    const x = rows.map((r) => r.iter as number);
    const ys = series.map((s) => rows.map((r) => r[s.key] ?? null));
    const specs: SeriesSpec[] = series.map((s) => ({ label: s.label, color: s.color, width: 1.75, dash: s.dash }));
    return { x, ys, specs };
  }, [rows, series]);
  return (
    <div className="min-w-0">
      <div className="text-[12px] font-medium text-neutral-800">{title}</div>
      <UChart x={x} ys={ys} series={specs} yLog={log} yRange={range} height={180} sync="cli" xLabel="iteration (one batch + one update)" />
      <div className="mt-1 text-[11.5px] leading-snug text-neutral-500">{note}</div>
    </div>
  );
}

// ------------------------------------------------------------------ 1. the problem
function Problem() {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.4fr_1fr]">
      <div>
        <div className="mb-1 text-[12px] font-medium text-neutral-800">The state s: 4 numbers the agent observes every step</div>
        <table className="w-full text-left text-[12.5px]">
          <thead className="text-[11px] text-neutral-500">
            <tr>
              <th className="py-1 font-normal">number</th>
              <th className="py-1 font-normal">unit</th>
              <th className="py-1 font-normal">limit</th>
              <th className="py-1 font-normal">network input</th>
            </tr>
          </thead>
          <tbody>
            {STATE.map((s) => (
              <tr key={s.name} className="border-t border-neutral-100">
                <td className="py-1 pr-3 text-neutral-900">{s.name}</td>
                <td className="py-1 pr-3 text-neutral-500">{s.unit}</td>
                <td className="py-1 pr-3 text-neutral-600">{s.limit}</td>
                <td className="py-1 font-mono text-[11.5px] text-neutral-600">value / {s.scale}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-2 text-[12px] text-neutral-500">
          The observation is the full state (nothing hidden). Each number is divided by its range so the networks get inputs around [−1, 1].
        </div>
      </div>
      <div className="grid grid-cols-1 gap-2 text-[12.5px]">
        <Card title="Actions a">
          <span style={{ color: ACTION_COLORS.left }}>← push left</span> or <span style={{ color: ACTION_COLORS.right }}>→ push right</span> (a fixed force; no
          &quot;do nothing&quot;).
        </Card>
        <Card title="Reward r">+1 for every step the pole is still up and the cart is still on the track. Nothing else counts: not the cart&apos;s position, not the wobble.</Card>
        <Card title="Episode end">
          <b>terminated</b>: the pole tips past ±12° or the cart leaves the track (a real end). <b>truncated</b>: 500 steps reached (time limit). Best possible return: 500.
        </Card>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 2. how it learns
function HowItLearns({ data }: { data: Latest | null }) {
  const batch = Number(data?.settings.batch ?? 5);
  const first = data?.rows[0];
  const firstSteps = first?.ret != null ? Math.round((first.ret as number) * batch) : null;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Card title="① Collect a batch">
          <p>
            The actor is <b>frozen</b> and plays <b>{batch} complete episodes</b>. Every step becomes one row (s<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>).
          </p>
          {firstSteps != null && (
            <p className="text-neutral-500">
              Iteration 1: {batch} episodes × {fmt(first?.ret, 1)} steps on average = {firstSteps} rows. Later, as episodes get longer, the batch grows (up to {batch * 500}).
            </p>
          )}
          <p className="text-neutral-500">On-policy: after one update the batch is thrown away and a new one is played with the new actor.</p>
        </Card>
        <Card title="② Critic: predicts the return" color={C.critic}>
          <p>
            Network 4 → {String(data?.settings.hidden ?? 8)} tanh → 1. Output V(s) = how much discounted reward is still to come from state s.
          </p>
          <p>Its target is the return that actually followed, computed backwards:</p>
          <Eq>G_t = r_t + γ·G_(t+1)</Eq>
          <p className="text-neutral-500">starting from 0 if the pole fell, or V(s_last) at the 500-step limit. γ = {String(data?.settings.gamma ?? 0.99)}.</p>
          <Eq>L_V = mean ½ (G_t − V(s_t))²</Eq>
        </Card>
        <Card title="③ Actor: learns from the advantage" color={C.actor}>
          <p>Network 4 → {String(data?.settings.hidden ?? 8)} tanh → 2 → softmax = π(a|s), the probability of each action. Actions are sampled from it.</p>
          <Eq>A_t = G_t − V(s_t)</Eq>
          <p className="text-neutral-500">A &gt; 0: better than the critic expected; A &lt; 0: worse.</p>
          <Eq>L_π = −mean A_t · log π(a_t|s_t)</Eq>
          <p className="text-neutral-500">A cross-entropy whose &quot;label&quot; is the action it took, weighted by A: push up what went better than expected, down what went worse.</p>
        </Card>
        <Card title="④ The objective">
          <p>Maximise the expected discounted return:</p>
          <Eq>J(θ) = E[ Σ_t γ^t r_t ]</Eq>
          <Eq>∇J = E[ A · ∇ log π(a|s) ]</Eq>
          <p className="text-neutral-500">
            So stepping down L_π steps up J. The <i>value</i> of L_π is not J (it is ≈ |A| × entropy). Every policy that survives 500 steps has the same J: staying
            centred is not rewarded.
          </p>
        </Card>
      </div>
      <div className="text-[12px] text-neutral-500">
        One iteration = ① collect → ② one critic gradient step → ③ one actor gradient step (both with Adam, from the same batch). Want to see one update weight by weight?{" "}
        <Link href="/learn" className="text-[#2a78d6] hover:underline">
          Step through it slowly →
        </Link>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ 3. terminal run
function RunInfo({ data, err }: { data: Latest | null; err: string }) {
  const [copied, setCopied] = useState(false);
  const st = data?.status ?? {};
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <code className="rounded bg-neutral-900 px-2.5 py-1.5 font-mono text-[12px] text-neutral-100">{COMMAND}</code>
        <button
          className="rounded border border-neutral-300 px-2 py-1 text-[12px] hover:bg-neutral-50"
          onClick={() => {
            navigator.clipboard?.writeText(COMMAND).then(() => setCopied(true)).catch(() => {});
          }}
        >
          {copied ? "copied" : "copy"}
        </button>
        <span className="text-[12px] text-neutral-500">logs in the terminal, the policy in a window every 25 iterations; this page shows the latest run (updates while it trains).</span>
      </div>
      {err ? (
        <div className="text-[12.5px] text-neutral-600">{err}</div>
      ) : data ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-neutral-600">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: st.state === "running" ? C.pos : st.state === "finished" ? C.actor : C.marker2 }} />
            <b className="font-medium text-neutral-800">{st.state}</b> · iteration {st.iter}/{st.iters} · started {st.started}
          </span>
          {["batch", "hidden", "actor_lr", "critic_lr", "gamma", "optimizer", "seed"].map((k) => (
            <span key={k} className="rounded bg-neutral-100 px-1.5 py-0.5 tabular-nums">
              {k} {String(data.settings[k])}
            </span>
          ))}
        </div>
      ) : (
        <div className="text-[12px] text-neutral-400">loading…</div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ 6. recorded episodes
function Player({ ep }: { ep: Episode }) {
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
  const trace = useMemo(() => {
    const x = ep.states.map((_, k) => k + 1);
    return { x, ys: [ep.states.map((st) => st[0] / 2.4), ep.states.map((st) => st[2] / 0.2095)] };
  }, [ep]);
  const markers: Marker[] = [{ x: i + 1, label: "", color: C.ink2 }];
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr_1fr]">
      <div>
        <EnvView envId="CartPole-v1" step={step} trail={[]} actionNames={ACTIONS} />
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
                {v} steps/s
              </option>
            ))}
          </select>
          <a className="text-[#2a78d6] hover:underline" href={`${BACKEND}/api/cli/latest/gif/${ep.name}`} target="_blank" rel="noreferrer">
            GIF
          </a>
        </div>
      </div>
      <div className="space-y-3">
        <table className="w-full text-[12.5px] tabular-nums">
          <thead className="text-[11px] text-neutral-500">
            <tr>
              <th className="text-left font-normal">state (step {i + 1})</th>
              <th className="text-right font-normal">value</th>
              <th className="text-right font-normal">network input</th>
            </tr>
          </thead>
          <tbody>
            {STATE.map((d, k) => (
              <tr key={d.name} className="border-t border-neutral-100">
                <td className="py-0.5 text-neutral-700">{d.name}</td>
                <td className="py-0.5 text-right">
                  {fmt(s[k], 3)} {d.unit}
                  {k === 2 ? ` (${((s[k] * 180) / Math.PI).toFixed(1)}°)` : ""}
                </td>
                <td className="py-0.5 text-right text-neutral-500">{(s[k] / d.scale).toFixed(2)}</td>
              </tr>
            ))}
            <tr className="border-t border-neutral-200">
              <td className="py-0.5 text-neutral-700">{ep.name === "good" ? "good policy pushes" : "agent pushes"}</td>
              <td className="py-0.5 text-right" style={{ color: ACTION_COLORS[ACTIONS[ep.actions[i]]] }}>
                {ACTIONS[ep.actions[i]]}
              </td>
              <td className="py-0.5 text-right text-neutral-500">P = {ep.probs[i][ep.actions[i]].toFixed(2)}</td>
            </tr>
            {ep.name !== "good" && (
              <tr>
                <td className="py-0.5 text-neutral-700">good policy would push</td>
                <td className="py-0.5 text-right" style={{ color: ACTION_COLORS[ACTIONS[good]] }}>
                  {ACTIONS[good]}
                </td>
                <td className="py-0.5 text-right text-neutral-500">{good === ep.actions[i] ? "same" : "different"}</td>
              </tr>
            )}
          </tbody>
        </table>
        <div>
          <div className="text-[11.5px] font-medium text-neutral-700">How close to failing, over the whole episode</div>
          <UChart
            x={trace.x}
            ys={trace.ys}
            height={140}
            yRange={[-1, 1]}
            markers={markers}
            xLabel="step"
            series={[
              { label: "cart position / 2.4", color: C.ink, width: 1.5 },
              { label: "pole angle / 12°", color: C.critic, width: 1.5 },
            ]}
          />
          <div className="text-[11px] text-neutral-500">±1 = the episode ends (track end or pole falls). A line that wanders off 0 = drift.</div>
        </div>
      </div>
    </div>
  );
}

function Episodes({ data }: { data: Latest }) {
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
              {isFinal ? `final (iter ${e.iter})` : `iter ${e.iter}`}: {e.steps}
            </button>
          );
        })}
        {good && (
          <>
            <span className="mx-1 text-neutral-300">|</span>
            <button
              onClick={() => setSel(good.name)}
              className={`rounded border px-2 py-0.5 text-[12px] tabular-nums ${good.name === current ? "border-[#1baf7a] bg-[#1baf7a] text-white" : "border-[#1baf7a] bg-white text-[#0b7a52]"}`}
            >
              good policy (hand-written): {good.steps}
            </button>
          </>
        )}
      </div>
      {ep && ep.name === current && <Player key={ep.name} ep={ep} />}
    </div>
  );
}

// ------------------------------------------------------------------ 7. logs
const TABLE: [string, string, number][] = [
  ["iter", "iter", 0],
  ["ret", "episode length", 1],
  ["eval", "eval", 0],
  ["p_good", "P(good action)", 2],
  ["H", "entropy", 2],
  ["Lpi", "actor loss", 2],
  ["Lv", "critic loss", 1],
  ["rel", "critic rel. error", 2],
  ["true_err", "critic true error", 2],
  ["noise", "noise floor", 2],
];

function Logs({ rows }: { rows: Row[] }) {
  return (
    <div className="max-h-96 overflow-auto">
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
export default function CartPolePage() {
  const [data, setData] = useState<Latest | null>(null);
  const [err, setErr] = useState("");

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
        if (alive) setErr(e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Backend not reachable on :8000 (start it first).");
      }
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  const rows = data?.rows ?? [];
  return (
    <div className="min-h-screen bg-[#fafaf9]">
      <header className="border-b border-neutral-200 bg-white px-4 py-3">
        <div className="mx-auto flex max-w-[1300px] items-baseline gap-3">
          <h1 className="text-[16px] font-semibold text-neutral-900">CartPole · batch actor-critic</h1>
          <span className="text-[12px] text-neutral-500">RL playground</span>
        </div>
      </header>
      <main className="mx-auto max-w-[1300px] space-y-3 p-3">
        <Section n={1} title="The problem: states, observations, actions, reward">
          <Problem />
        </Section>

        <Section n={2} title="How it learns: batch → critic → policy, and the objective">
          <HowItLearns data={data} />
        </Section>

        <Section n={3} title="Terminal run">
          <RunInfo data={data} err={err} />
        </Section>

        {rows.length > 0 && (
          <>
            <Section n={4} title="Is the agent learning?">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <Chart
                  title="Episode length (500 = perfect)"
                  rows={rows}
                  series={[
                    { key: "ret", label: "batch: mean episode length", color: C.ink },
                    { key: "eval", label: "eval: most likely action, 10 fixed starts", color: C.pos },
                  ]}
                  note="The batch plays sampled actions (exploring); eval always takes the most likely action, so it is usually higher."
                />
                <Chart
                  title="Is the policy right?  (does not grow with episode length)"
                  rows={rows}
                  series={[
                    { key: "p_good", label: "P(good action)", color: C.actor },
                    { key: "agree", label: "most likely = good action", color: C.actor, dash: [5, 3] },
                    { key: "H", label: "entropy", color: C.muted },
                  ]}
                  range={[0, 1]}
                  note={
                    <>
                      Good action = the hand-written policy (pole angle + 0.5·angular velocity &gt; 0 → right), which scores 500. P(good action): 0.5 coin flip → 1 perfect.
                      Entropy: 0.69 = coin flip, 0 = always the same action.
                    </>
                  }
                />
              </div>
            </Section>

            <Section n={5} title="Is the critic learning?">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <Chart
                  title="Critic loss ½(G − V)²  (raw)"
                  rows={rows}
                  series={[{ key: "Lv", label: "critic loss on the new batch", color: C.critic }]}
                  log
                  note="Grows when episodes get longer (bigger returns to predict), so it cannot tell by itself whether the critic is learning."
                />
                <Chart
                  title="Critic error as a fraction of the value  (scale-free)"
                  rows={rows}
                  series={[
                    { key: "true_err", label: "TRUE error: |V − real average return|", color: C.critic },
                    { key: "noise", label: "noise floor (no critic can go below)", color: C.muted, dash: [5, 3] },
                    { key: "rel", label: "error on the batch", color: C.critic, dash: [2, 3] },
                  ]}
                  range={[0, 1.2]}
                  note="True error: the critic against the real average return from 10 states × 10 episodes. Falling = the critic is learning, whatever the episode length."
                />
              </div>
            </Section>
          </>
        )}

        {data && data.episodes.length > 0 && (
          <Section
            n={6}
            title="Recorded episodes: the policy at that iteration"
            help="One episode every 10 iterations, always from the same start state, always taking the most likely action. The good policy plays the same start for comparison."
          >
            <Episodes data={data} />
          </Section>
        )}

        {rows.length > 0 && (
          <Section n={7} title="Logs" help="The numbers the terminal printed (runs/cli/latest/log.csv), newest at the bottom.">
            <Logs rows={rows} />
          </Section>
        )}
      </main>
    </div>
  );
}
