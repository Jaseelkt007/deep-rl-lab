"use client";

import Link from "next/link";
import { useEffect, useMemo } from "react";
import { C } from "@/lib/colors";
import { connectLive, currentView, useStore } from "@/lib/store";
import { fmt } from "@/lib/math";
import Controls from "./Controls";
import Curves, { noSignalValue } from "./Curves";
import EnvView from "./EnvView";
import Panel from "./Panel";
import { DeltaChart, StepNumbers, ValueVsReturn } from "./StepView";
import UpdateView from "./UpdateView";
import { LiveScrubber, ReplayBar, RunsList } from "./Replay";
import PhaseMaps from "./PhaseMaps";
import Diagnostics from "./Diagnostics";
import { EvChart } from "./CriticView";

const STATE_COLORS: Record<string, string> = {
  running: C.pos,
  paused: C.marker2,
  finished: C.actor,
  stopped: C.muted,
  error: C.neg,
  idle: C.axis,
};

function StatusBar() {
  const st = useStore();
  const s = st.status;
  const state = s?.state ?? "idle";
  return (
    <div className="flex items-center gap-3 text-[12px] text-neutral-600">
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full" style={{ background: st.connected ? STATE_COLORS[state] : C.neg }} />
        {st.connected ? state : "backend offline"}
      </span>
      {s?.run_id && (
        <>
          <span className="tabular-nums">
            ep {s.episode}/{s.num_episodes}
          </span>
          <span className="tabular-nums">{s.total_steps?.toLocaleString()} steps</span>
          <span className="tabular-nums">{fmt(s.steps_per_sec, 0)} steps/s</span>
          <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-[11px] text-neutral-700">{s.label}</span>
        </>
      )}
      {s && s.jobs_running > 0 && <span className="text-neutral-500">{s.jobs_running} background runs</span>}
    </div>
  );
}

export default function Dashboard() {
  const st = useStore();
  useEffect(() => connectLive(), []);
  const view = currentView();
  const rs = view.runStart;
  const envId = rs?.config.env_id ?? st.status?.env_id ?? "MountainCar-v0";
  const gamma = rs?.config.gamma ?? 0.99;
  const actionNames = rs?.action_names ?? (envId === "CartPole-v1" ? ["left", "right"] : ["left", "none", "right"]);
  const obsNames = rs?.obs_names ?? [];

  const steps = view.steps;
  const idx = st.cursor == null ? steps.length - 1 : Math.min(st.cursor, steps.length - 1);
  const step = idx >= 0 ? steps[idx] : null;
  const trail = useMemo(() => steps.slice(0, idx + 1).map((s) => s.s[0]), [steps, idx, steps.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // animate the one-update view only when steps come slowly enough to follow
  const animate = st.mode === "live" && (st.status?.state === "paused" || (st.status?.steps_per_sec ?? 0) < 4);
  const finished = view.lastSteps.length ? view.lastSteps : [];

  return (
    <div className="min-h-screen bg-[#fafaf9]">
      <header className="sticky top-0 z-20 flex items-center gap-4 border-b border-neutral-200 bg-white/90 px-4 py-2 backdrop-blur">
        <h1 className="text-[14px] font-semibold text-neutral-900">RL playground</h1>
        <nav className="flex gap-3 text-[12px]">
          <Link href="/" className="font-medium text-neutral-900">
            Live
          </Link>
          <Link href="/compare" className="text-neutral-500 hover:text-neutral-900">
            Compare
          </Link>
        </nav>
        <div className="ml-auto">
          <StatusBar />
        </div>
      </header>

      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-[270px_1fr]">
        <aside className="space-y-3 lg:sticky lg:top-12 lg:max-h-[calc(100vh-3.5rem)] lg:overflow-y-auto">
          <Panel title="Controls" help="Pick a task and switches, then Start. Every switch is off by default: that is the plain one-step actor-critic. Step does one environment step (and its update) while paused.">
            <Controls status={st.status} />
          </Panel>
          <Panel title="Saved runs" help="Every run is saved in runs/<id>/ (config.json, metrics.jsonl, checkpoints). Click one to replay it without training again: curves, maps and every logged episode step by step.">
            <RunsList />
          </Panel>
        </aside>

        <main className="min-w-0 space-y-3">
          <ReplayBar />
          {st.status?.error && (
            <pre className="overflow-auto rounded-md border border-red-200 bg-red-50 p-2 text-[11px] text-red-800">{st.status.error}</pre>
          )}
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(260px,1fr)_2.6fr]">
            <Panel
              title="Environment"
              help="The state the agent acted in, the action it took (arrow) and the actor's probabilities for each action. Faint dots: where this episode has been."
              right={step && <span className="text-[11px] tabular-nums text-neutral-500">ep {step.ep} · t {step.t}</span>}
            >
              <EnvView envId={envId} step={step} trail={envId === "MountainCar-v0" ? trail : []} actionNames={actionNames} />
            </Panel>
            <Panel
              title="One update"
              help="One step of learning, left to right: the actor proposes probabilities, the critic predicts V(s); the env returns r and s′; δ = r + γV(s′) − V(s) says whether things went better (+) or worse (−) than predicted; δ pushes the taken action's probability up or down, and moves V(s) towards the target."
            >
              <UpdateView step={step} gamma={gamma} actionNames={actionNames} obsNames={obsNames} animate={animate} />
            </Panel>
          </div>

          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            <Panel title="This step" help="The numbers of the step shown above, and δ over the steps of its episode.">
              <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_1.4fr]">
                <StepNumbers step={step} />
                <DeltaChart steps={steps} cursor={idx} />
              </div>
              <LiveScrubber />
            </Panel>
            <Panel
              title="Is the critic any good?"
              help="The actor can only be as good as its teacher. If the critic's V does not track the return, δ is noise and the actor learns noise."
            >
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <ValueVsReturn steps={finished} gamma={gamma} vStar={noSignalValue(rs?.config)} />
                <EvChart eps={view.episodes} />
              </div>
            </Panel>
          </div>

          <Panel
            title="Learning curves"
            help="One point per episode (faint) and a 20-episode average (bold). Dashed vertical lines: CartPole, critic saturation > 90% and entropy < 0.1 (the collapse chain); MountainCar, 'no signal' = V reached −1/(1−γ) everywhere so δ ≈ 0; both, the first success."
          >
            <Curves eps={view.episodes} cfg={rs?.config} />
          </Panel>

          <Panel
            title="Phase plane: what each network has learned, state by state"
            help="Maps over (position, velocity) for MountainCar, (pole angle, angular velocity) for CartPole, saved every K episodes. Drag the slider to watch them change during training. These are the key view: a successful MountainCar agent learns a spiral, swinging back and forth to build energy."
          >
            <PhaseMaps snapshots={view.snapshots} runStart={rs} vStar={noSignalValue(rs?.config)} />
          </Panel>

          <Panel
            title="Inside the networks"
            help="The same diagnostics we use on large agents: which layers still learn, whether hidden units are saturated or constant, whether the representation still tells states apart, and whether the actor uses the state at all."
          >
            <Diagnostics eps={view.episodes} probes={view.probes} cfg={rs?.config} />
          </Panel>
        </main>
      </div>
    </div>
  );
}


