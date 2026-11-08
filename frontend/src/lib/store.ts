"use client";

import { useSyncExternalStore } from "react";
import { WS_URL } from "./api";
import type { EpisodeRec, ProbeRec, RunDetail, RunStart, SnapshotRec, Status, StepRec } from "./types";

/** Everything one view needs about one run (the live one, or a reopened one). */
export interface RunView {
  runStart: RunStart | null;
  episodes: EpisodeRec[];
  snapshots: SnapshotRec[];
  probes: ProbeRec[];
  steps: StepRec[]; // live: the episode in progress; replay: the selected episode
  lastSteps: StepRec[]; // live: the last finished episode
}

const emptyView = (): RunView => ({ runStart: null, episodes: [], snapshots: [], probes: [], steps: [], lastSteps: [] });

interface State {
  mode: "live" | "replay";
  live: RunView;
  replay: RunView & { runId: string | null; meta: RunDetail["meta"] | null; withSteps: number[]; ep: number | null };
  status: Status | null;
  connected: boolean;
  cursor: number | null; // index into the shown steps; null = follow the latest
  version: number;
}

const state: State = {
  mode: "live",
  live: emptyView(),
  replay: { ...emptyView(), runId: null, meta: null, withSteps: [], ep: null },
  status: null,
  connected: false,
  cursor: null,
  version: 0,
};

// ---------------------------------------------------------------- notify
const listeners = new Set<() => void>();
let scheduled = false;
function notify() {
  if (scheduled) return;
  scheduled = true;
  const run = () => {
    scheduled = false;
    state.version++;
    listeners.forEach((l) => l());
  };
  if (typeof window !== "undefined") window.requestAnimationFrame(run);
  else run();
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** Re-render on every store change (rAF-throttled) and read the state directly. */
export function useStore(): State {
  useSyncExternalStore(subscribe, () => state.version, () => 0);
  return state;
}

export function currentView(): RunView {
  return state.mode === "live" ? state.live : state.replay;
}

export const storeActions = {
  setMode(mode: "live" | "replay") {
    state.mode = mode;
    state.cursor = null;
    notify();
  },
  setCursor(c: number | null) {
    state.cursor = c;
    notify();
  },
  openReplay(detail: RunDetail) {
    state.replay = {
      runStart: detail.run_start,
      episodes: detail.episodes,
      snapshots: detail.snapshots,
      probes: detail.probes,
      steps: [],
      lastSteps: [],
      runId: detail.meta.run_id,
      meta: detail.meta,
      withSteps: detail.episodes_with_steps,
      ep: null,
    };
    state.mode = "replay";
    state.cursor = 0;
    notify();
  },
  setReplaySteps(ep: number, steps: StepRec[]) {
    state.replay.ep = ep;
    state.replay.steps = steps;
    state.replay.lastSteps = steps;
    state.cursor = 0;
    notify();
  },
  clearLive() {
    state.live = emptyView();
    state.cursor = null;
    notify();
  },
};

// ---------------------------------------------------------------- WebSocket
type Msg =
  | { type: "reset"; run_start: RunStart | null; history: (EpisodeRec | SnapshotRec | ProbeRec | { type: "run_end" })[]; steps: StepRec[] }
  | { type: "records"; records: (EpisodeRec | SnapshotRec | ProbeRec | { type: "run_end" })[] }
  | { type: "steps"; steps: StepRec[]; replace: boolean }
  | Status;

function addRecords(v: RunView, recs: { type: string }[]) {
  for (const r of recs) {
    if (r.type === "episode") v.episodes.push(r as EpisodeRec);
    else if (r.type === "snapshot") v.snapshots.push(r as SnapshotRec);
    else if (r.type === "probe") v.probes.push(r as ProbeRec);
  }
}

function handle(msg: Msg) {
  const v = state.live;
  switch (msg.type) {
    case "reset": {
      state.live = { ...emptyView(), runStart: msg.run_start };
      addRecords(state.live, msg.history);
      state.live.steps = msg.steps;
      const last = msg.steps.at(-1);
      if (last && (last.term || last.trunc)) state.live.lastSteps = msg.steps; // a finished episode
      break;
    }
    case "records":
      addRecords(v, msg.records);
      break;
    case "steps":
      if (msg.replace) {
        v.lastSteps = msg.steps;
        v.steps = msg.steps;
      } else {
        for (const s of msg.steps) {
          if (v.steps.length && v.steps[0].ep !== s.ep) {
            v.lastSteps = v.steps;
            v.steps = [];
          }
          v.steps.push(s);
        }
      }
      break;
    case "status":
      state.status = msg;
      break;
  }
  notify();
}

let socket: WebSocket | null = null;
let retry = 0;
export function connectLive() {
  if (typeof window === "undefined" || socket) return;
  const open = () => {
    const ws = new WebSocket(WS_URL);
    socket = ws;
    ws.onopen = () => {
      retry = 0;
      state.connected = true;
      notify();
    };
    ws.onmessage = (e) => handle(JSON.parse(e.data));
    ws.onclose = () => {
      state.connected = false;
      socket = null;
      notify();
      retry = Math.min(retry + 1, 6);
      window.setTimeout(open, 300 * 2 ** retry);
    };
    ws.onerror = () => ws.close();
  };
  open();
}
