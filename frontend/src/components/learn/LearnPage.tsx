"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { C } from "@/lib/colors";
import { fmt } from "@/lib/math";
import type { Describe, LearnConfig, Summary, Trace } from "@/lib/learn";
import { learnApi } from "@/lib/learn";
import UChart from "../UChart";
import { ActorStepStage, AdvantageStage, Box, CollectStage, CriticOutStage, CriticStepStage, NetworksStage, NextStage, ProblemStage, ReturnsStage } from "./Stages";

const DEFAULTS: Record<LearnConfig["env_id"], LearnConfig> = {
  "CartPole-v1": { env_id: "CartPole-v1", seed: 0, batch_episodes: null, hidden: 8, optimizer: "adam", actor_lr: 0.01, critic_lr: 0.03, gamma: 0.99, shaping: "none" },
  "MountainCar-v0": { env_id: "MountainCar-v0", seed: 0, batch_episodes: null, hidden: 8, optimizer: "adam", actor_lr: 0.01, critic_lr: 0.03, gamma: 0.99, shaping: "none" },
};

const PRE = [
  { key: "problem", label: "The problem" },
  { key: "networks", label: "The agent: two networks" },
];
const ITER = [
  { key: "collect", label: "1 · Collect a batch" },
  { key: "critic", label: "2 · Critic judges it" },
  { key: "returns", label: "3 · Returns G" },
  { key: "advantage", label: "4 · Advantage A" },
  { key: "criticStep", label: "5 · Critic step" },
  { key: "actorStep", label: "6 · Actor step" },
  { key: "next", label: "7 · Repeat" },
];

function Settings({ cfg, setCfg, onApply }: { cfg: LearnConfig; setCfg: (c: LearnConfig) => void; onApply: () => void }) {
  const num = (k: keyof LearnConfig, label: string, step = "any") => (
    <label className="flex items-center gap-1.5">
      <span className="text-neutral-500">{label}</span>
      <input
        className="w-20 rounded border border-neutral-200 px-1 py-0.5 text-right tabular-nums"
        type="number"
        step={step}
        value={(cfg[k] as number | null) ?? ""}
        placeholder="auto"
        onChange={(e) => setCfg({ ...cfg, [k]: e.target.value === "" ? null : Number(e.target.value) })}
      />
    </label>
  );
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[12px]">
      {num("seed", "seed", "1")}
      {num("batch_episodes", "episodes per batch", "1")}
      <label className="flex items-center gap-1.5">
        <span className="text-neutral-500">hidden units</span>
        <select className="rounded border border-neutral-200 px-1 py-0.5" value={cfg.hidden} onChange={(e) => setCfg({ ...cfg, hidden: Number(e.target.value) })}>
          {[4, 8, 16, 64].map((h) => (
            <option key={h}>{h}</option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-1.5">
        <span className="text-neutral-500">optimizer</span>
        <select className="rounded border border-neutral-200 px-1 py-0.5" value={cfg.optimizer} onChange={(e) => setCfg({ ...cfg, optimizer: e.target.value as "adam" | "sgd" })}>
          <option value="adam">Adam</option>
          <option value="sgd">SGD</option>
        </select>
      </label>
      {num("actor_lr", "actor lr")}
      {num("critic_lr", "critic lr")}
      {num("gamma", "γ")}
      {cfg.env_id === "MountainCar-v0" && (
        <label className="flex items-center gap-1.5">
          <span className="text-neutral-500">reward shaping</span>
          <select className="rounded border border-neutral-200 px-1 py-0.5" value={cfg.shaping} onChange={(e) => setCfg({ ...cfg, shaping: e.target.value as "none" | "energy" })}>
            <option value="none">none</option>
            <option value="energy">energy</option>
          </select>
        </label>
      )}
      <button className="rounded-md border border-neutral-300 px-2.5 py-1 font-medium hover:bg-neutral-50" onClick={onApply}>
        Start over with these settings
      </button>
    </div>
  );
}

function Progress({ summaries, k }: { summaries: Summary[]; k: number }) {
  const data = useMemo(() => {
    const x = summaries.map((s) => s.k);
    return { x, ret: [summaries.map((s) => s.mean_return)], loss: [summaries.map((s) => s.critic_loss)], ev: [summaries.map((s) => (s.ev == null ? null : Math.max(-1, s.ev)))] };
  }, [summaries]);
  if (summaries.length < 2) return null;
  const marker = [{ x: k, label: "", color: C.ink2 }];
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      <div>
        <div className="text-[11px] font-medium text-neutral-600">mean episode return per batch</div>
        <UChart x={data.x} ys={data.ret} series={[{ label: "return", color: C.ink, width: 2 }]} markers={marker} height={110} xLabel="iteration" />
      </div>
      <div>
        <div className="text-[11px] font-medium text-neutral-600">critic loss (log)</div>
        <UChart x={data.x} ys={data.loss} series={[{ label: "critic loss", color: C.critic, width: 2 }]} yLog markers={marker} height={110} xLabel="iteration" />
      </div>
      <div>
        <div className="text-[11px] font-medium text-neutral-600">critic explained variance (1 = predicts the returns)</div>
        <UChart x={data.x} ys={data.ev} series={[{ label: "EV", color: C.critic, width: 2 }]} yRange={[-1, 1]} markers={marker} height={110} xLabel="iteration" />
      </div>
    </div>
  );
}

export default function LearnPage() {
  const [cfg, setCfg] = useState<LearnConfig>(DEFAULTS["CartPole-v1"]);
  const [d, setD] = useState<Describe | null>(null);
  const [tr, setTr] = useState<Trace | null>(null);
  const [summaries, setSummaries] = useState<Summary[]>([]);
  const [stage, setStage] = useState(0);
  const [focus, setFocus] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState("");

  const start = useCallback(async (c: LearnConfig) => {
    setBusy("starting");
    setErr("");
    try {
      const desc = await learnApi.create(c);
      setD(desc);
      setTr(null);
      setSummaries([]);
      setStage(0);
      setFocus(0);
    } catch (e) {
      setErr(`Could not reach the backend on :8000 (${String(e).slice(0, 80)})`);
    } finally {
      setBusy(null);
    }
  }, []);

  useEffect(() => {
    const id = window.setTimeout(() => start(DEFAULTS["CartPole-v1"]), 0);
    return () => window.clearTimeout(id);
  }, [start]);

  const stages = tr ? ITER : PRE;
  const run = async (what: "step" | number) => {
    if (!d) return;
    setBusy(what === "step" ? "collecting the next batch" : `running ${what} iterations`);
    setErr("");
    try {
      const r = what === "step" ? await learnApi.step(d.session) : await learnApi.forward(d.session, what);
      setTr(r.trace);
      setSummaries(r.summaries);
      setStage(0);
      setFocus(0);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      setErr(String(e).slice(0, 160));
    } finally {
      setBusy(null);
    }
  };
  const isLast = stage === stages.length - 1;
  const next = () => (isLast ? run("step") : setStage(stage + 1));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "ArrowRight" && !busy) next();
      if (e.key === "ArrowLeft" && stage > 0) setStage(stage - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const key = stages[stage]?.key;
  return (
    <div className="min-h-screen bg-[#fafaf9]">
      <header className="sticky top-0 z-20 flex items-center gap-4 border-b border-neutral-200 bg-white/90 px-4 py-2 backdrop-blur">
        <h1 className="text-[14px] font-semibold text-neutral-900">RL playground</h1>
        <nav className="flex gap-3 text-[12px]">
          <Link href="/learn" className="font-medium text-neutral-900">
            Learn
          </Link>
          <Link href="/terminal" className="text-neutral-500 hover:text-neutral-900">
            Terminal run
          </Link>
          <Link href="/" className="text-neutral-500 hover:text-neutral-900">
            Live
          </Link>
          <Link href="/compare" className="text-neutral-500 hover:text-neutral-900">
            Compare
          </Link>
        </nav>
        <div className="ml-auto flex items-center gap-1 rounded-md bg-neutral-100 p-0.5 text-[12px]">
          {(["CartPole-v1", "MountainCar-v0"] as const).map((env) => (
            <button
              key={env}
              onClick={() => {
                const c = { ...DEFAULTS[env], seed: cfg.seed };
                setCfg(c);
                start(c);
              }}
              className={`rounded px-2.5 py-1 ${cfg.env_id === env ? "bg-white font-medium text-neutral-900 shadow-sm" : "text-neutral-500"}`}
            >
              {env}
            </button>
          ))}
        </div>
      </header>

      <div className="mx-auto max-w-[1400px] space-y-3 p-3">
        <Box>
          <div className="flex flex-wrap items-center gap-3">
            <div className="text-[15px] font-semibold text-neutral-900">How actor-critic learns, one batch at a time</div>
            <div className="text-[12px] text-neutral-500">
              {tr ? (
                <>
                  iteration <b className="text-neutral-800">{tr.k}</b> · {tr.k * (d?.n_episodes ?? 0)} episodes played so far
                </>
              ) : (
                "before training"
              )}
            </div>
            <div className="ml-auto flex items-center gap-1.5 text-[12px]">
              <span className="text-neutral-500">fast-forward:</span>
              {[1, 10, 50, 150].map((n) => (
                <button key={n} disabled={!!busy || !d} onClick={() => (n === 1 ? run("step") : run(n))} className="rounded border border-neutral-300 px-2 py-0.5 hover:bg-neutral-50 disabled:opacity-40">
                  +{n}
                </button>
              ))}
            </div>
          </div>
          <details className="mt-2">
            <summary className="cursor-pointer text-[11px] text-neutral-500">settings (network size, learning rates, batch size, optimizer)</summary>
            <div className="mt-2">
              <Settings cfg={cfg} setCfg={setCfg} onApply={() => start(cfg)} />
            </div>
          </details>
          {summaries.length >= 2 && (
            <div className="mt-3">
              <Progress summaries={summaries} k={tr?.k ?? 0} />
            </div>
          )}
        </Box>

        {/* stage stepper */}
        <div className="flex flex-wrap items-center gap-1.5">
          {stages.map((s, i) => (
            <button
              key={s.key}
              onClick={() => setStage(i)}
              className={`rounded-full border px-3 py-1 text-[12px] ${i === stage ? "border-neutral-900 bg-neutral-900 text-white" : i < stage ? "border-neutral-300 bg-white text-neutral-700" : "border-neutral-200 bg-white text-neutral-400"}`}
            >
              {s.label}
            </button>
          ))}
        </div>

        {err && <div className="rounded-md border border-red-200 bg-red-50 p-2 text-[12px] text-red-800">{err}</div>}
        {busy && <div className="text-[12px] text-neutral-500">… {busy}</div>}

        {d && !busy && (
          <div key={`${d.session}-${tr?.k ?? 0}-${key}`}>
            {key === "problem" && <ProblemStage d={d} />}
            {key === "networks" && <NetworksStage d={d} weights={d.weights} />}
            {tr && key === "collect" && <CollectStage d={d} tr={tr} />}
            {tr && key === "critic" && <CriticOutStage d={d} tr={tr} focus={focus} setFocus={setFocus} />}
            {tr && key === "returns" && <ReturnsStage d={d} tr={tr} />}
            {tr && key === "advantage" && <AdvantageStage d={d} tr={tr} focus={focus} setFocus={setFocus} />}
            {tr && key === "criticStep" && <CriticStepStage d={d} tr={tr} focus={focus} />}
            {tr && key === "actorStep" && <ActorStepStage d={d} tr={tr} focus={focus} />}
            {tr && key === "next" && <NextStage d={d} tr={tr} />}
          </div>
        )}

        <div className="flex items-center gap-2 pb-6">
          <button disabled={stage === 0} onClick={() => setStage(stage - 1)} className="rounded-md border border-neutral-300 px-3 py-1.5 text-[13px] disabled:opacity-40">
            ← Back
          </button>
          <button disabled={!!busy || !d} onClick={next} className="rounded-md border border-[#2a78d6] bg-[#2a78d6] px-3 py-1.5 text-[13px] font-medium text-white hover:bg-[#256abf] disabled:opacity-40">
            {isLast ? (tr ? `Collect the next batch (iteration ${tr.k + 1}) →` : "Collect the first batch →") : "Next →"}
          </button>
          <span className="text-[11px] text-neutral-400">keyboard: ← →</span>
          {tr && (
            <span className="ml-auto text-[11px] tabular-nums text-neutral-500">
              this batch: {tr.episodes.length} episodes · {tr.summary.steps} steps · mean return {fmt(tr.summary.mean_return, 1)} · critic loss {fmt(tr.summary.critic_loss)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
