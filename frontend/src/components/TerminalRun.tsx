"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { BACKEND } from "@/lib/api";
import { C } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { StepRec } from "@/lib/types";
import EnvView from "./EnvView";
import Panel from "./Panel";
import UChart, { SeriesSpec } from "./UChart";

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
      <UChart x={x} ys={ys} series={specs} yLog={log} yRange={range} height={170} sync="cli" xLabel="iteration" />
    </Panel>
  );
}

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
  const step = { s: ep.states[i], a: ep.actions[i], p: ep.probs[i] } as unknown as StepRec;
  return (
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
      <div className="mt-1 text-[11px] tabular-nums text-neutral-500">
        angle {fmt(ep.states[i][2], 3)} rad · angular velocity {fmt(ep.states[i][3], 3)} ·{" "}
        {ep.name === "good" ? (
          <>the good policy pushes {ACTIONS[ep.actions[i]]}</>
        ) : (
          <>
            the good policy would push {ep.states[i][2] + 0.5 * ep.states[i][3] > 0 ? "right" : "left"}, the agent pushes {ACTIONS[ep.actions[i]]}
          </>
        )}
      </div>
    </div>
  );
}

const TABLE: [string, string, number][] = [
  ["iter", "iter", 0],
  ["ret", "length", 1],
  ["Lv", "critic loss", 1],
  ["rel", "critic rel. error", 2],
  ["true_err", "critic true error", 2],
  ["Lpi", "actor loss", 2],
  ["H", "entropy", 2],
  ["p_good", "P(good action)", 2],
  ["ce_good", "CE to good policy", 3],
  ["dir", "update direction", 2],
  ["eval", "eval", 0],
];

export default function TerminalRun() {
  const [data, setData] = useState<Latest | null>(null);
  const [err, setErr] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  const [ep, setEp] = useState<Episode | null>(null);

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
        if (alive) setErr(String(e instanceof Error ? e.message : e));
      }
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  const learned = (data?.episodes ?? []).filter((e) => e.name !== "good");
  const good = data?.episodes.find((e) => e.name === "good");
  const finalIter = data?.status.state !== "running" ? learned.at(-1)?.iter : undefined;
  const current = sel ?? learned.at(-1)?.name ?? null;
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

  const rows = data?.rows ?? [];
  const st = data?.status ?? {};
  return (
    <div className="min-h-screen bg-[#fafaf9]">
      <header className="sticky top-0 z-20 flex items-center gap-4 border-b border-neutral-200 bg-white/90 px-4 py-2 backdrop-blur">
        <h1 className="text-[14px] font-semibold text-neutral-900">RL playground</h1>
        <nav className="flex gap-3 text-[12px]">
          <Link href="/learn" className="text-neutral-500 hover:text-neutral-900">
            Learn
          </Link>
          <Link href="/terminal" className="font-medium text-neutral-900">
            Terminal run
          </Link>
          <Link href="/" className="text-neutral-500 hover:text-neutral-900">
            Live
          </Link>
          <Link href="/compare" className="text-neutral-500 hover:text-neutral-900">
            Compare
          </Link>
        </nav>
      </header>
      <div className="mx-auto max-w-[1400px] space-y-3 p-3">
        <Panel
          title="Latest terminal run: CartPole, batch actor-critic"
          help="The last run of  uv run python scripts/train_cartpole.py  (saved in runs/cli/latest/, replaced by the next run). The page updates every 2 s while it is training."
        >
          {err ? (
            <div className="text-[13px] text-neutral-600">{err}</div>
          ) : !data ? (
            <div className="text-[13px] text-neutral-400">loading…</div>
          ) : (
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-[12px] text-neutral-600">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: st.state === "running" ? C.pos : st.state === "finished" ? C.actor : C.marker2 }} />
                {st.state} · iteration {st.iter}/{st.iters}
              </span>
              <span>started {st.started}</span>
              {["batch", "hidden", "actor_lr", "critic_lr", "gamma", "optimizer", "seed"].map((k) => (
                <span key={k} className="rounded bg-neutral-100 px-1.5 py-0.5 tabular-nums">
                  {k} {String(data.settings[k])}
                </span>
              ))}
            </div>
          )}
        </Panel>

        {rows.length > 0 && (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <Chart
              title="1. Is the agent learning?  episode length (500 = perfect)"
              help="Batch: the average episode length of the 5 episodes played in that iteration. Eval: the policy always taking its most likely action, 10 fixed start states."
              rows={rows}
              series={[
                { key: "ret", label: "batch: mean episode length", color: C.ink },
                { key: "eval", label: "eval (most likely action)", color: C.pos },
              ]}
            />
            <Chart
              title="2. Is the actor learning?  (does not grow with episode length)"
              help="P(good action): the probability the actor gives to the action of the known good policy (angle + 0.5·angular velocity > 0 → right): 0.5 = coin flip, 1 = perfect. Agreement: how often its most likely action is the good one. Entropy: how undecided it is (0.69 = coin flip)."
              rows={rows}
              series={[
                { key: "p_good", label: "P(good action)", color: C.actor },
                { key: "agree", label: "most likely = good action", color: C.actor, dash: [5, 3] },
                { key: "H", label: "entropy", color: C.muted },
              ]}
              range={[0, 1]}
            />
            <Chart
              title="3. Critic loss ½(G−V)²  (grows with the returns)"
              help="The raw critic loss on each new batch. It grows when episodes get longer (bigger returns), so it does not show by itself whether the critic is learning: see panel 4."
              rows={rows}
              series={[{ key: "Lv", label: "critic loss", color: C.critic }]}
              log
            />
            <Chart
              title="4. Is the critic learning?  error as a fraction of the value"
              help="True error: |V − real average return| measured from 10 states × 10 episodes, as a fraction of the value. Noise floor: how much single returns scatter (no critic can go below). Rel: error on the batch. Falling true error = the critic is learning."
              rows={rows}
              series={[
                { key: "true_err", label: "TRUE error", color: C.critic },
                { key: "noise", label: "noise floor", color: C.muted, dash: [5, 3] },
                { key: "rel", label: "error on the batch (rel)", color: C.critic, dash: [2, 3] },
              ]}
              range={[0, 1.2]}
            />
          </div>
        )}

        {data && data.episodes.length > 0 && (
          <Panel
            title="Recorded episodes: the policy at that iteration, always taking its most likely action"
            help="Saved at iteration 1, every 25 iterations and at the end (the final policy), all from the same start state, so you can compare how the policy improved. 'Good policy' is the hand-written rule (angle + 0.5·angular velocity > 0 → right) on the same start, for comparison."
          >
            <div className="mb-2 flex flex-wrap items-center gap-1.5">
              {learned.map((e) => {
                const isFinal = e.iter === finalIter;
                return (
                  <button
                    key={e.name}
                    onClick={() => setSel(e.name)}
                    className={`rounded border px-2 py-0.5 text-[12px] tabular-nums ${e.name === current ? "border-neutral-900 bg-neutral-900 text-white" : isFinal ? "border-[#2a78d6] bg-white text-[#2a78d6]" : "border-neutral-300 bg-white"}`}
                  >
                    {isFinal ? `final policy (iteration ${e.iter})` : `iteration ${e.iter}`}: {e.steps} steps
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
                    good policy (hand-written): {good.steps} steps
                  </button>
                </>
              )}
            </div>
            {ep && ep.name === current && <Player key={ep.name} ep={ep} />}
          </Panel>
        )}

        {rows.length > 0 && (
          <Panel title="Log" help="The same numbers the terminal printed (log.csv). Newest at the bottom.">
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
          </Panel>
        )}
      </div>
    </div>
  );
}
