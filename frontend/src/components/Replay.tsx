"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { C } from "@/lib/colors";
import { fmt } from "@/lib/math";
import { storeActions, useStore } from "@/lib/store";
import type { RunMeta } from "@/lib/types";

const ENV_SHORT: Record<string, string> = { "MountainCar-v0": "MC", "CartPole-v1": "CP" };

export async function openRun(runId: string, ep?: number) {
  const detail = await api.run(runId);
  storeActions.openReplay(detail);
  const eps = detail.episodes_with_steps;
  const pick = ep ?? eps.at(-1);
  if (pick != null) storeActions.setReplaySteps(pick, await api.steps(runId, pick));
}

/** Saved runs (runs/<id>/), newest first. Click one to replay it without training again. */
export function RunsList() {
  const st = useStore();
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .runs()
        .then((r) => alive && (setRuns(r), setError("")))
        .catch(() => alive && setError("backend not reachable"));
    load();
    const id = window.setInterval(load, 5000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  if (error) return <div className="text-[11px] text-neutral-400">{error}</div>;
  if (!runs.length) return <div className="text-[11px] text-neutral-400">No saved runs yet.</div>;
  return (
    <ul className="max-h-80 space-y-0.5 overflow-y-auto pr-1">
      {runs.map((r) => {
        const active = st.mode === "replay" && st.replay.runId === r.run_id;
        const running = r.summary.status.startsWith("running");
        return (
          <li key={r.run_id}>
            <button
              type="button"
              onClick={() => openRun(r.run_id)}
              title={`${r.run_id}${r.group ? `\n${r.group}` : ""}`}
              className={`w-full rounded px-1.5 py-1 text-left text-[11px] hover:bg-neutral-50 ${active ? "bg-blue-50 ring-1 ring-blue-200" : ""}`}
            >
              <div className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: running ? C.pos : r.summary.status === "finished" ? C.axis : C.marker2 }} />
                <span className="font-medium text-neutral-800">{ENV_SHORT[r.env_id]}</span>
                <span className="truncate text-neutral-700">{r.label}</span>
                <span className="ml-auto shrink-0 text-neutral-400">s{r.seed}</span>
              </div>
              {r.group && <div className="truncate pl-3 text-[10px] text-neutral-500">{r.group}</div>}
              <div className="flex gap-2 pl-3 text-[10px] text-neutral-500 tabular-nums">
                <span>{r.created.slice(5, 16)}</span>
                <span>{r.summary.episodes} ep</span>
                <span>last-100 return {fmt(r.summary.recent_return, 1)}</span>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** Episode picker and step player for a reopened run. */
export function ReplayBar() {
  const st = useStore();
  const rp = st.replay;
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(30);
  const n = rp.steps.length;
  const cursor = st.cursor ?? 0;

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const c = st.cursor ?? 0;
      if (c >= n - 1) setPlaying(false);
      else storeActions.setCursor(c + 1);
    }, 1000 / fps);
    return () => window.clearInterval(id);
  }, [playing, fps, n, st]);

  if (st.mode !== "replay" || !rp.meta) return null;
  const epIdx = rp.ep == null ? -1 : rp.withSteps.indexOf(rp.ep);
  const goEp = async (i: number) => {
    const ep = rp.withSteps[Math.max(0, Math.min(rp.withSteps.length - 1, i))];
    if (ep == null || !rp.runId) return;
    setPlaying(false);
    storeActions.setReplaySteps(ep, await api.steps(rp.runId, ep));
  };
  const btn = "rounded border border-neutral-300 px-2 py-0.5 text-[12px] hover:bg-neutral-50 disabled:opacity-40";

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-blue-200 bg-blue-50/50 px-3 py-2 text-[12px]">
      <div className="flex items-center gap-2">
        <span className="font-medium text-neutral-800">Replay</span>
        <span className="rounded bg-white px-1.5 py-0.5 text-[11px] text-neutral-700">{rp.meta.label}</span>
        <span className="text-neutral-500">
          {rp.meta.env_id} · seed {rp.meta.seed} · {rp.episodes.length} episodes
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-neutral-500">episode</span>
        <button className={btn} disabled={epIdx <= 0} onClick={() => goEp(epIdx - 1)}>
          ‹
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, rp.withSteps.length - 1)}
          value={Math.max(0, epIdx)}
          onChange={(e) => goEp(Number(e.target.value))}
          className="w-40"
        />
        <button className={btn} disabled={epIdx >= rp.withSteps.length - 1} onClick={() => goEp(epIdx + 1)}>
          ›
        </button>
        <span className="w-12 tabular-nums text-neutral-700">{rp.ep ?? "–"}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <button className={btn} disabled={!n} onClick={() => (cursor >= n - 1 ? (storeActions.setCursor(0), setPlaying(true)) : setPlaying(!playing))}>
          {playing ? "❚❚" : "▶"}
        </button>
        <input type="range" min={0} max={Math.max(0, n - 1)} value={cursor} onChange={(e) => storeActions.setCursor(Number(e.target.value))} className="w-56" />
        <span className="w-16 tabular-nums text-neutral-700">
          t {n ? cursor : "–"}/{n ? n - 1 : "–"}
        </span>
        <select className="rounded border border-neutral-300 bg-white px-1 py-0.5 text-[12px]" value={fps} onChange={(e) => setFps(Number(e.target.value))}>
          {[5, 15, 30, 60, 120].map((f) => (
            <option key={f} value={f}>
              {f} steps/s
            </option>
          ))}
        </select>
      </div>
      <button className={`${btn} ml-auto`} onClick={() => storeActions.setMode("live")}>
        Back to live
      </button>
    </div>
  );
}

/** In live mode: follow the newest step, or browse the episode in progress. */
export function LiveScrubber() {
  const st = useStore();
  if (st.mode !== "live") return null;
  const n = st.live.steps.length;
  const following = st.cursor == null;
  return (
    <div className="mt-2 flex items-center gap-2 text-[11px] text-neutral-500">
      <input
        type="range"
        min={0}
        max={Math.max(0, n - 1)}
        value={following ? Math.max(0, n - 1) : Math.min(st.cursor!, n - 1)}
        onChange={(e) => storeActions.setCursor(Number(e.target.value))}
        className="flex-1"
      />
      <button
        type="button"
        className={`rounded border px-1.5 py-0.5 ${following ? "border-blue-300 bg-blue-50 text-blue-700" : "border-neutral-300"}`}
        onClick={() => storeActions.setCursor(null)}
      >
        follow
      </button>
    </div>
  );
}
