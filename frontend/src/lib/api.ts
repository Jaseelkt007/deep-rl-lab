import type { RunConfig, RunDetail, RunMeta, StepRec, Status } from "./types";

export const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/** The hosted site (Vercel) has no backend: the terminal runs are read from static JSON in
 *  public/data, written by backend/scripts/export_site_data.py. Pages that need the live
 *  backend (Learn, Compare) are hidden there. */
export const STATIC = process.env.NEXT_PUBLIC_STATIC === "1";

/** URL of a terminal-run API call: `/api/cli/<path>?env=…` locally, `/data/<env>/<path>.json` on the hosted site. */
export function cliUrl(path: string, env: string): string {
  if (STATIC) return `/data/${env}/${path}.json`;
  return `${BACKEND}/api/cli/${path}${env === "cartpole" ? "" : `?env=${env}`}`;
}
export const WS_URL = BACKEND.replace(/^http/, "ws") + "/ws";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BACKEND + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) => req<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  defaults: () => req<{ config: RunConfig }>("/api/config/default"),
  start: (config: RunConfig, speed: number | null, paused = false) =>
    post<{ run_id: string; label: string }>("/api/runs", { config, speed, paused }),
  control: (action: "pause" | "resume" | "step" | "stop" | "reset") => post<Status>("/api/control", { action }),
  speed: (steps_per_sec: number | null) => post<Status>("/api/speed", { steps_per_sec }),
  runs: () => req<RunMeta[]>("/api/runs"),
  run: (id: string, lite = false) => req<RunDetail>(`/api/runs/${encodeURIComponent(id)}${lite ? "?lite=true" : ""}`),
  steps: (id: string, ep: number) => req<StepRec[]>(`/api/runs/${encodeURIComponent(id)}/episodes/${ep}/steps`),
  batch: (config: RunConfig, seeds: number[]) => post<{ group: string; run_ids: string[] }>("/api/batch", { config, seeds }),
};
